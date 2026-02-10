import os
import json
import time
import argparse
from dataclasses import asdict, dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ----------------------------
# Repro / Utils
# ----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_str():
    return time.strftime("%Y-%m-%d_%H-%M-%S")


# ----------------------------
# Data Loading (ETH/UCY style)
# ----------------------------
def list_txt_files(scene_dir: str, split: str) -> List[str]:
    """
    Expected structure:
      data_root/<scene>/<split>/*.txt
    Example:
      data_real/raw/hotel/train/biwi_hotel_train.txt
    """
    split_dir = os.path.join(scene_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    files = []
    for f in os.listdir(split_dir):
        if f.lower().endswith(".txt"):
            files.append(os.path.join(split_dir, f))
    files.sort()
    if len(files) == 0:
        raise FileNotFoundError(f"No .txt files found in: {split_dir}")
    return files


def read_eth_ucy_txt(path: str) -> pd.DataFrame:
    """
    ETH/UCY standard format:
      frame_id pedestrian_id x y
    separated by whitespace.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["frame", "pid", "x", "y"])
    # Ensure numeric
    df["frame"] = df["frame"].astype(int)
    df["pid"] = df["pid"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def load_split_df(data_root: str, scene: str, split: str, verbose: bool = True) -> pd.DataFrame:
    scene_dir = os.path.join(data_root, scene)
    files = list_txt_files(scene_dir, split)
    dfs = [read_eth_ucy_txt(p) for p in files]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Sort for safety
    df = df.sort_values(["pid", "frame"]).reset_index(drop=True)

    if verbose:
        print(f"[{scene}/{split}] Loaded {len(files)} txt files, total rows={len(df):,}")
        print("  Example files:", [os.path.basename(x) for x in files[:3]])

    return df


# ----------------------------
# Windowing
# ----------------------------
def build_windows_from_df(df: pd.DataFrame, obs_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised windows:
      X: (N, obs_len, 2)
      Y: (N, 2)  next-step target (t+1)
    Done per pedestrian id, respecting frame order.
    """
    X_list = []
    Y_list = []

    for pid, g in df.groupby("pid"):
        g = g.sort_values("frame")
        xy = g[["x", "y"]].values.astype(np.float32)

        if len(xy) <= obs_len:
            continue

        # sliding windows: [t-obs_len ... t-1] -> target t
        for start in range(0, len(xy) - obs_len):
            end = start + obs_len
            if end >= len(xy):
                break
            x_seq = xy[start:end]          # obs_len
            y_next = xy[end]              # next point
            X_list.append(x_seq)
            Y_list.append(y_next)

    if len(X_list) == 0:
        raise RuntimeError("No windows created. Check data or obs_len.")

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


# ----------------------------
# Normalization
# ----------------------------
def compute_xy_stats(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_df[["x", "y"]].mean().values.astype(np.float32)
    sigma = train_df[["x", "y"]].std().values.astype(np.float32)
    sigma = np.maximum(sigma, 1e-6)  # avoid divide-by-zero
    return mu, sigma


def apply_xy_norm_df(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df[["x", "y"]] = (df[["x", "y"]] - mu) / sigma
    return df


def denorm_xy(arr_xy: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return arr_xy * sigma + mu


# ----------------------------
# Dataset / Model
# ----------------------------
class TrajDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LSTMNext(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, T, 2)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # (B, H)
        pred = self.fc(last)           # (B, 2)
        return pred


# ----------------------------
# Metrics
# ----------------------------
def mse_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))


def ade_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    # Euclidean distance averaged
    d = np.linalg.norm(pred - gt, axis=1)
    return float(np.mean(d))


def acc_at_k(pred: np.ndarray, gt: np.ndarray, ks: List[float]) -> Dict[str, float]:
    d = np.linalg.norm(pred - gt, axis=1)
    out = {}
    for k in ks:
        out[f"acc@{k}"] = float(np.mean(d <= k))
    return out


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             ks: List[float],
             mu: np.ndarray = None,
             sigma: np.ndarray = None) -> Dict[str, float]:
    model.eval()
    preds = []
    gts = []

    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        Pb = model(Xb)

        preds.append(Pb.detach().cpu().numpy())
        gts.append(Yb.detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)

    # If normalized training, denormalize for metrics
    if (mu is not None) and (sigma is not None):
        pred_real = denorm_xy(pred, mu, sigma)
        gt_real = denorm_xy(gt, mu, sigma)
    else:
        pred_real = pred
        gt_real = gt

    metrics = {
        "mse": mse_metric(pred_real, gt_real),
        "ade": ade_metric(pred_real, gt_real),
    }
    metrics.update(acc_at_k(pred_real, gt_real, ks))
    return metrics


# ----------------------------
# Plotting
# ----------------------------
def plot_metrics(history: List[Dict[str, float]], out_path: str, keys: List[str], title: str):
    if len(history) == 0:
        return
    epochs = np.arange(1, len(history) + 1)
    for k in keys:
        vals = [h.get(k, np.nan) for h in history]
        plt.plot(epochs, vals, label=k)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Config
# ----------------------------
@dataclass
class RunConfig:
    scene: str
    data_root: str
    obs_len: int
    epochs: int
    lr: float
    batch_size: int
    hidden_dim: int
    num_layers: int
    dropout: float
    normalize: str
    ks: List[float]
    run_name: str
    device: str
    seed: int


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, choices=["eth", "hotel", "univ", "zara01", "zara02"])
    parser.add_argument("--data_root", type=str, default="data_real/raw")
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize", type=str, default="scene", choices=["none", "scene"])
    parser.add_argument("--k", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0], help="Acc@k thresholds")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"run_{args.scene}_ep{args.epochs}_lr{args.lr}_{now_str()}"
    run_dir = os.path.join("runs", run_name)
    ensure_dir(run_dir)

    cfg = RunConfig(
        scene=args.scene,
        data_root=args.data_root,
        obs_len=args.obs_len,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        normalize=args.normalize,
        ks=list(args.k),
        run_name=run_name,
        device=str(device),
        seed=args.seed
    )

    print("===================================")
    print(f"Run: {cfg.run_name}")
    print(f"Scene: {cfg.scene}")
    print(f"Data root: {cfg.data_root}")
    print(f"Device: {cfg.device}")
    print(f"Normalization: {cfg.normalize}")
    print(f"Acc@k: {cfg.ks}")
    print("===================================")

    # Load split dataframes (concat all txt in each split)
    train_df = load_split_df(cfg.data_root, cfg.scene, "train", verbose=True)
    val_df   = load_split_df(cfg.data_root, cfg.scene, "val", verbose=True)
    test_df  = load_split_df(cfg.data_root, cfg.scene, "test", verbose=True)

    # Normalization (scene-level z-score using train stats only)
    mu, sigma = None, None
    if cfg.normalize == "scene":
        mu, sigma = compute_xy_stats(train_df)
        train_df = apply_xy_norm_df(train_df, mu, sigma)
        val_df   = apply_xy_norm_df(val_df, mu, sigma)
        test_df  = apply_xy_norm_df(test_df, mu, sigma)

        np.save(os.path.join(run_dir, "norm_mu.npy"), mu)
        np.save(os.path.join(run_dir, "norm_sigma.npy"), sigma)

        print(f"[Norm] mu={mu.tolist()} sigma={sigma.tolist()}")
    else:
        print("[Norm] none")

    # Build windows
    X_train, Y_train = build_windows_from_df(train_df, obs_len=cfg.obs_len)
    X_val, Y_val     = build_windows_from_df(val_df, obs_len=cfg.obs_len)
    X_test, Y_test   = build_windows_from_df(test_df, obs_len=cfg.obs_len)

    print(f"Train windows: {X_train.shape} {Y_train.shape}")
    print(f"Val windows:   {X_val.shape} {Y_val.shape}")
    print(f"Test windows:  {X_test.shape} {Y_test.shape}")

    # Dataloaders
    train_loader = DataLoader(TrajDataset(X_train, Y_train), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TrajDataset(X_val, Y_val), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TrajDataset(X_test, Y_test), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Model
    model = LSTMNext(input_dim=2, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    history = []
    best_val_ade = float("inf")
    best_path = os.path.join(run_dir, "best_model.pth")
    last_path = os.path.join(run_dir, "last_model.pth")

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for Xb, Yb in train_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, Yb)  # training loss in normalized space if normalize=scene
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Train metrics (in real scale for reporting)
        train_metrics = evaluate(model, train_loader, device, cfg.ks, mu=mu, sigma=sigma)
        val_metrics   = evaluate(model, val_loader, device, cfg.ks, mu=mu, sigma=sigma)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_mse": train_metrics["mse"],
            "train_ade": train_metrics["ade"],
            "val_mse": val_metrics["mse"],
            "val_ade": val_metrics["ade"],
        }
        for k in cfg.ks:
            row[f"val_acc@{k}"] = val_metrics[f"acc@{k}"]

        history.append(row)

        # Print
        acc_str = " | ".join([f"val Acc@{k}={val_metrics[f'acc@{k}']:.3f}" for k in cfg.ks])
        print(f"Epoch {epoch:03d} | train MSE={train_metrics['mse']:.4f} ADE={train_metrics['ade']:.4f} "
              f"|| val MSE={val_metrics['mse']:.4f} ADE={val_metrics['ade']:.4f} | {acc_str}")

        # Save best
        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best model -> {best_path} (best val ADE={best_val_ade:.4f})")

    # Save last
    torch.save(model.state_dict(), last_path)
    print(f"✅ Saved last model -> {last_path}")

    # Save metrics csv/json
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    pd.DataFrame(history).to_csv(metrics_csv, index=False)

    summary = {
        "best_val_ade": best_val_ade,
        "run_dir": run_dir,
        "best_model": best_path,
        "last_model": last_path,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot curves
    # MSE/ADE curves
    plot_metrics(
        [{"train_ade": h["train_ade"], "val_ade": h["val_ade"]} for h in history],
        os.path.join(run_dir, "ade_curve.png"),
        keys=["train_ade", "val_ade"],
        title=f"ADE curve ({cfg.scene})"
    )
    plot_metrics(
        [{"train_mse": h["train_mse"], "val_mse": h["val_mse"]} for h in history],
        os.path.join(run_dir, "mse_curve.png"),
        keys=["train_mse", "val_mse"],
        title=f"MSE curve ({cfg.scene})"
    )

    # Acc@k curves (val)
    acc_hist = []
    for h in history:
        row_acc = {}
        for k in cfg.ks:
            row_acc[f"val_acc@{k}"] = h[f"val_acc@{k}"]
        acc_hist.append(row_acc)

    plot_metrics(
        acc_hist,
        os.path.join(run_dir, "acc_at_k_curve.png"),
        keys=[f"val_acc@{k}" for k in cfg.ks],
        title=f"Val Acc@k ({cfg.scene})"
    )

    # TEST using best model
    best_model = LSTMNext(input_dim=2, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(best_model, test_loader, device, cfg.ks, mu=mu, sigma=sigma)

    print("=== TEST (best model) ===")
    print(test_metrics)

    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"✅ All artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
