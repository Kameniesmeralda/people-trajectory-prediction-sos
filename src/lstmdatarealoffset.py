import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ============================================================
# Reproductibilité
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Utils I/O
# ============================================================
def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_all_txt_files(split_dir: Path):
    txt_files = sorted([p for p in split_dir.glob("*.txt") if p.is_file()])
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in: {split_dir}")
    dfs = []
    for p in txt_files:
        df = pd.read_csv(
            p,
            sep=r"\s+",
            header=None,
            names=["frame", "pid", "x", "y"],
        )
        dfs.append(df)
    big = pd.concat(dfs, axis=0, ignore_index=True)
    return big, txt_files


# ============================================================
# Windowing
# ============================================================
def build_windows_from_df(df: pd.DataFrame, obs_len: int = 10, pred_horizon: int = 1):
    """
    Build (X, Y) where:
      X: (N, obs_len, 2) observed positions
      Y: (N, 2) absolute position at horizon steps after last obs

    pred_horizon:
      1 => predict the next step after the observation window
      h => predict h steps after the last observed position
    """
    if pred_horizon < 1:
        raise ValueError("--pred_horizon must be >= 1")

    df = df.copy()
    df = df.sort_values(["pid", "frame"])
    X_list, Y_list = [], []
    target_offset = obs_len + (pred_horizon - 1)

    for pid, g in df.groupby("pid"):
        coords = g[["x", "y"]].to_numpy(dtype=np.float32)
        if len(coords) <= target_offset:
            continue

        max_i = len(coords) - target_offset - 1
        for i in range(0, max_i + 1):
            x_seq = coords[i : i + obs_len]           # (obs_len,2)
            y_tgt = coords[i + target_offset]         # (2,)
            X_list.append(x_seq)
            Y_list.append(y_tgt)

    if len(X_list) == 0:
        raise ValueError("No windows were built. Check data formatting / obs_len / pred_horizon.")

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


# ============================================================
# Dataset OFFSET
# ============================================================
class OffsetTrajectoryDataset(Dataset):
    """
    Train in OFFSET space:
      input: offsets relative to last observed point
      target: delta to future point (Y - last_obs)

    Keep last_obs and Y_abs to compute metrics in absolute space.
    """
    def __init__(self, X_abs: np.ndarray, Y_abs: np.ndarray):
        super().__init__()
        self.X_abs = torch.from_numpy(X_abs)  # (N,T,2)
        self.Y_abs = torch.from_numpy(Y_abs)  # (N,2)

        last_obs = self.X_abs[:, -1, :]                     # (N,2)
        self.X_off = self.X_abs - last_obs.unsqueeze(1)     # (N,T,2)
        self.Y_delta = self.Y_abs - last_obs                # (N,2)
        self.last_obs = last_obs

    def __len__(self):
        return self.X_abs.shape[0]

    def __getitem__(self, idx):
        return {
            "X_off": self.X_off[idx],
            "Y_delta": self.Y_delta[idx],
            "last_obs": self.last_obs[idx],
            "Y_abs": self.Y_abs[idx],
        }


# ============================================================
# Model
# ============================================================
class LSTMOffset(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)


# ============================================================
# Metrics + Eval
# ============================================================
@torch.no_grad()
def compute_metrics_abs(y_pred_abs: torch.Tensor, y_true_abs: torch.Tensor, ks):
    diff = y_pred_abs - y_true_abs
    mse = torch.mean(diff ** 2).item()
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))
    ade = torch.mean(dist).item()
    out = {"mse": mse, "ade": ade}
    for k in ks:
        out[f"acc@{k}"] = torch.mean((dist <= float(k)).float()).item()
    return out


@torch.no_grad()
def evaluate(model, loader, device, ks):
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        X_off = batch["X_off"].to(device)
        last_obs = batch["last_obs"].to(device)
        Y_abs = batch["Y_abs"].to(device)

        pred_delta = model(X_off)
        pred_abs = last_obs + pred_delta

        all_pred.append(pred_abs.cpu())
        all_true.append(Y_abs.cpu())

    y_pred_abs = torch.cat(all_pred, dim=0)
    y_true_abs = torch.cat(all_true, dim=0)
    return compute_metrics_abs(y_pred_abs, y_true_abs, ks)


# ============================================================
# Plotting
# ============================================================
def plot_acc_curves(metrics_list, split_name: str, ks, outpath: Path):
    plt.figure()
    for k in ks:
        key = f"acc@{k}"
        series = [m[split_name].get(key, None) for m in metrics_list]
        if any(v is None for v in series):
            continue
        plt.plot(series, label=key)
    plt.title(f"{split_name.upper()} Accuracy@k over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {outpath}")


def plot_test_acc_bar(test_metrics: dict, outpath: Path):
    acc = {k: v for k, v in test_metrics.items() if k.startswith("acc@")}
    if not acc:
        print("ℹ️ No acc@k found in test metrics, skip bar plot.")
        return
    labels = list(acc.keys())
    values = [acc[k] for k in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("TEST Accuracy@k (Best Model)")
    plt.grid(axis="y")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {outpath}")


# ============================================================
# ✅ train_and_eval() : utilisé par PSO (VRAI entraînement)
# ============================================================
_DATA_CACHE = {}  # (data_root, scene, obs_len, pred_horizon) -> (train_ds, val_ds)

def _get_train_val_loaders(cfg: dict, device):
    key = (cfg["data_root"], cfg["scene"], cfg["obs_len"], cfg["pred_horizon"])
    if key not in _DATA_CACHE:
        scene_dir = Path(cfg["data_root"]) / cfg["scene"]
        train_df, _ = load_all_txt_files(scene_dir / "train")
        val_df, _ = load_all_txt_files(scene_dir / "val")
        Xtr, Ytr = build_windows_from_df(train_df, obs_len=cfg["obs_len"], pred_horizon=cfg["pred_horizon"])
        Xva, Yva = build_windows_from_df(val_df, obs_len=cfg["obs_len"], pred_horizon=cfg["pred_horizon"])
        train_ds = OffsetTrajectoryDataset(Xtr, Ytr)
        val_ds = OffsetTrajectoryDataset(Xva, Yva)
        _DATA_CACHE[key] = (train_ds, val_ds)

    train_ds, val_ds = _DATA_CACHE[key]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def train_and_eval(cfg: dict) -> float:
    """
    Objectif PSO : retourner la meilleure val_ADE (à MINIMISER).
    """
    # --- defaults nécessaires ---
    required = ["scene", "data_root", "obs_len", "pred_horizon", "epochs", "lr", "hidden_dim", "num_layers", "dropout"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"train_and_eval: missing cfg['{k}']")

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks = cfg.get("ks", [0.25, 0.5, 1.0, 2.0, 4.0])
    ks = [float(x) for x in ks]

    train_loader, val_loader = _get_train_val_loaders(cfg, device)

    model = LSTMOffset(
        input_dim=2,
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    best_val_ade = float("inf")
    patience = int(cfg.get("patience", 3))
    bad = 0

    # Option: limiter le nombre de batches par epoch (PSO = plus rapide)
    max_batches = cfg.get("max_batches", None)
    if max_batches is not None:
        max_batches = int(max_batches)

    for _epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        for b_idx, batch in enumerate(train_loader, start=1):
            X_off = batch["X_off"].to(device)
            Y_delta = batch["Y_delta"].to(device)

            optimizer.zero_grad()
            pred_delta = model(X_off)
            loss = criterion(pred_delta, Y_delta)
            loss.backward()
            optimizer.step()

            if max_batches is not None and b_idx >= max_batches:
                break

        val_metrics = evaluate(model, val_loader, device, ks)
        val_ade = float(val_metrics["ade"])

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_val_ade


# ============================================================
# Main training (ton script normal)
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="hotel",
                        choices=["eth", "hotel", "univ", "zara01", "zara02"])
    parser.add_argument("--data_root", type=str, default="data_real/raw")
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_horizon", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_test_each_epoch", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks = [float(x) for x in args.k]

    run_name = args.run_name or f"run_offset_{args.scene}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print(f"Run: {run_name}")
    print(f"Scene: {args.scene}")
    print("Mode: OFFSET + HUBER (abs metrics)")
    print(f"Device: {device}")
    print(f"Acc@k: {ks}")
    print(f"obs_len: {args.obs_len} | pred_horizon: {args.pred_horizon}")
    print(f"Eval test each epoch: {args.eval_test_each_epoch}")
    print("========================================")

    # Load data
    scene_dir = Path(args.data_root) / args.scene
    train_df, train_files = load_all_txt_files(scene_dir / "train")
    val_df, val_files = load_all_txt_files(scene_dir / "val")
    test_df, test_files = load_all_txt_files(scene_dir / "test")

    print(f"[{args.scene}/train] Loaded {len(train_files)} txt files, total rows={len(train_df):,}")
    print(f"[{args.scene}/val] Loaded {len(val_files)} txt files, total rows={len(val_df):,}")
    print(f"[{args.scene}/test] Loaded {len(test_files)} txt files, total rows={len(test_df):,}")

    Xtr, Ytr = build_windows_from_df(train_df, obs_len=args.obs_len, pred_horizon=args.pred_horizon)
    Xva, Yva = build_windows_from_df(val_df, obs_len=args.obs_len, pred_horizon=args.pred_horizon)
    Xte, Yte = build_windows_from_df(test_df, obs_len=args.obs_len, pred_horizon=args.pred_horizon)

    print(f"Train windows: {Xtr.shape} {Ytr.shape}")
    print(f"Val windows:   {Xva.shape} {Yva.shape}")
    print(f"Test windows:  {Xte.shape} {Yte.shape}")

    train_ds = OffsetTrajectoryDataset(Xtr, Ytr)
    val_ds = OffsetTrajectoryDataset(Xva, Yva)
    test_ds = OffsetTrajectoryDataset(Xte, Yte)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = LSTMOffset(2, args.hidden_dim, args.num_layers, args.dropout).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ade = float("inf")
    best_path = run_dir / "best_model.pth"
    last_path = run_dir / "last_model.pth"
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_pred_all, train_true_all = [], []

        for batch in train_loader:
            X_off = batch["X_off"].to(device)
            Y_delta = batch["Y_delta"].to(device)
            last_obs = batch["last_obs"].to(device)
            Y_abs = batch["Y_abs"].to(device)

            optimizer.zero_grad()
            pred_delta = model(X_off)
            loss = criterion(pred_delta, Y_delta)
            loss.backward()
            optimizer.step()

            pred_abs = (last_obs + pred_delta).detach().cpu()
            train_pred_all.append(pred_abs)
            train_true_all.append(Y_abs.detach().cpu())

        y_pred_tr = torch.cat(train_pred_all, dim=0)
        y_true_tr = torch.cat(train_true_all, dim=0)
        train_metrics = compute_metrics_abs(y_pred_tr, y_true_tr, ks)

        val_metrics = evaluate(model, val_loader, device, ks)
        test_metrics = evaluate(model, test_loader, device, ks) if args.eval_test_each_epoch else None

        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            torch.save(model.state_dict(), best_path)

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        if test_metrics is not None:
            row["test"] = test_metrics
        metrics_history.append(row)
        save_json(metrics_history, run_dir / "metrics.json")

        msg = (
            f"Epoch {epoch:03d} | "
            f"train ADE={train_metrics['ade']:.4f} MSE={train_metrics['mse']:.4f} || "
            f"val ADE={val_metrics['ade']:.4f} MSE={val_metrics['mse']:.4f} "
        )
        for k in ks:
            msg += f"| val Acc@{k}={val_metrics[f'acc@{k}']:.3f} "
        print(msg)

    torch.save(model.state_dict(), last_path)

    best_model = LSTMOffset(2, args.hidden_dim, args.num_layers, args.dropout).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    best_model.eval()

    test_best = evaluate(best_model, test_loader, device, ks)
    save_json(test_best, run_dir / "test_metrics_best.json")
    print("=== TEST (best model) ===")
    print(test_best)

    # Plots
    train_ade = [m["train"]["ade"] for m in metrics_history]
    val_ade = [m["val"]["ade"] for m in metrics_history]
    train_mse = [m["train"]["mse"] for m in metrics_history]
    val_mse = [m["val"]["mse"] for m in metrics_history]

    plt.figure()
    plt.plot(train_ade, label="train")
    plt.plot(val_ade, label="val")
    plt.title("ADE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("ADE")
    plt.grid(True)
    plt.legend()
    ade_path = run_dir / "ade_curve.png"
    plt.savefig(ade_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {ade_path}")

    plt.figure()
    plt.plot(train_mse, label="train")
    plt.plot(val_mse, label="val")
    plt.title("MSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    mse_path = run_dir / "mse_curve.png"
    plt.savefig(mse_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {mse_path}")

    plot_acc_curves(metrics_history, "train", ks, run_dir / "acc_at_k_train.png")
    plot_acc_curves(metrics_history, "val", ks, run_dir / "acc_at_k_val.png")
    if args.eval_test_each_epoch and all("test" in m for m in metrics_history):
        plot_acc_curves(metrics_history, "test", ks, run_dir / "acc_at_k_test.png")
    else:
        print("ℹ️ Test curves not plotted (use --eval_test_each_epoch).")

    plot_test_acc_bar(test_best, run_dir / "test_acc_at_k_bar.png")
    save_json(vars(args), run_dir / "config.json")

    print(f"✅ Saved best model: {best_path}")
    print(f"✅ Saved last model: {last_path}")
    print(f"📁 All artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
