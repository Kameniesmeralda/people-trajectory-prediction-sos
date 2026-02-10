import os
import glob
import json
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Data loading utilities
# ----------------------------
def load_split(data_root: str, scene: str, split: str) -> pd.DataFrame:
    """
    Reads all .txt files from: data_root/scene/split/*.txt
    Each line expected: frame pid x y (whitespace separated)
    Returns a concatenated DataFrame.
    """
    split_dir = os.path.join(data_root, scene, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    files = sorted(glob.glob(os.path.join(split_dir, "*.txt")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .txt files found in: {split_dir}")

    dfs = []
    for path in files:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["frame", "pid", "x", "y"],
            engine="python"
        )
        df["source_file"] = os.path.basename(path)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def build_traj_dict(df: pd.DataFrame) -> dict:
    """
    Converts DataFrame to dict: pid -> array of shape (T,3) columns [frame,x,y]
    Ensures numeric and sorted by pid/frame.
    """
    df = df.copy()

    # Force numeric
    for col in ["frame", "pid", "x", "y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame", "pid", "x", "y"])

    # Sort
    df = df.sort_values(["pid", "frame"]).reset_index(drop=True)

    traj_by_pid = {
        int(pid): g[["frame", "x", "y"]].to_numpy(dtype=np.float32)
        for pid, g in df.groupby("pid")
    }
    return traj_by_pid


def build_windows(traj_by_pid: dict, obs_len: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds supervised dataset:
      X: (N, obs_len, 2)  -> consecutive (x,y)
      Y: (N, 2)           -> next (x,y)
    """
    X_list, Y_list = [], []

    for traj in traj_by_pid.values():
        coords = traj[:, 1:3]  # keep x,y (T,2)
        if len(coords) <= obs_len:
            continue

        for t in range(len(coords) - obs_len):
            X_list.append(coords[t:t + obs_len])
            Y_list.append(coords[t + obs_len])

    if len(X_list) == 0:
        X = np.zeros((0, obs_len, 2), dtype=np.float32)
        Y = np.zeros((0, 2), dtype=np.float32)
    else:
        X = np.stack(X_list).astype(np.float32)
        Y = np.stack(Y_list).astype(np.float32)

    return X, Y


# ----------------------------
# PyTorch dataset
# ----------------------------
class TrajDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ----------------------------
# Model
# ----------------------------
class NextStepLSTM(nn.Module):
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
        out, _ = self.lstm(x)      # out: (B, T, H)
        last = out[:, -1, :]       # (B, H)
        pred = self.fc(last)       # (B, 2)
        return pred


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def compute_batch_metrics(pred, target, k_list):
    """
    pred, target: (B,2)
    Returns: mse(float), ade(float), acc_dict
    """
    mse = torch.mean((pred - target) ** 2)
    dist = torch.norm(pred - target, dim=1)   # (B,)
    ade = dist.mean()
    acc = {k: (dist <= k).float().mean() for k in k_list}
    return mse.item(), ade.item(), {k: acc[k].item() for k in k_list}


def run_epoch(model, loader, device, optimizer=None, k_list=None):
    """
    If optimizer is None -> eval mode
    else -> train mode
    Returns aggregated metrics dict
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    mse_sum, ade_sum, n = 0.0, 0.0, 0
    acc_sum = {k: 0.0 for k in k_list}

    for Xb, Yb in loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(Xb)

        loss = torch.mean((pred - Yb) ** 2)  # MSE
        if is_train:
            loss.backward()
            optimizer.step()

        # metrics
        mse_b, ade_b, acc_b = compute_batch_metrics(pred, Yb, k_list)
        bsz = Xb.shape[0]
        mse_sum += mse_b * bsz
        ade_sum += ade_b * bsz
        for k in k_list:
            acc_sum[k] += acc_b[k] * bsz
        n += bsz

    if n == 0:
        # avoid divide by zero
        out = {"mse": float("nan"), "ade": float("nan")}
        for k in k_list:
            out[f"acc@{k}"] = float("nan")
        return out

    out = {"mse": mse_sum / n, "ade": ade_sum / n}
    for k in k_list:
        out[f"acc@{k}"] = acc_sum[k] / n
    return out


# ----------------------------
# Plotting & saving
# ----------------------------
def save_plots(df_metrics: pd.DataFrame, run_dir: str, k_list):
    # Loss curves
    plt.figure()
    plt.plot(df_metrics["epoch"], df_metrics["train_mse"], label="train MSE")
    plt.plot(df_metrics["epoch"], df_metrics["val_mse"], label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss curves (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"))
    plt.close()

    # ADE curves
    plt.figure()
    plt.plot(df_metrics["epoch"], df_metrics["train_ade"], label="train ADE")
    plt.plot(df_metrics["epoch"], df_metrics["val_ade"], label="val ADE")
    plt.xlabel("Epoch")
    plt.ylabel("ADE")
    plt.title("Error curves (ADE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "ade_curves.png"))
    plt.close()

    # Acc curves (one plot, multiple lines)
    plt.figure()
    for k in k_list:
        plt.plot(df_metrics["epoch"], df_metrics[f"train_acc@{k}"], label=f"train Acc@{k}")
        plt.plot(df_metrics["epoch"], df_metrics[f"val_acc@{k}"], label=f"val Acc@{k}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curves (Acc@k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "acc_curves.png"))
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data_real/raw",
                        help="Path containing scenes: eth, hotel, univ, zara01, zara02")
    parser.add_argument("--scene", type=str, default="hotel",
                        choices=["eth", "hotel", "univ", "zara01", "zara02"])
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Acc@k thresholds (space-separated)
    parser.add_argument("--k", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                        help="Thresholds for Acc@k in coordinate units")

    parser.add_argument("--run_name", type=str, default=None,
                        help="Run folder name inside runs/. Example: run_epochs100_lr1e-4_day1")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run directory
    os.makedirs("runs", exist_ok=True)
    if args.run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"run_scene{args.scene}_ep{args.epochs}_lr{args.lr}_{stamp}"
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["device"] = str(device)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("===================================")
    print("Run:", args.run_name)
    print("Scene:", args.scene)
    print("Data root:", args.data_root)
    print("Device:", device)
    print("Acc@k:", args.k)
    print("===================================")

    # Load splits
    df_train = load_split(args.data_root, args.scene, "train")
    df_val   = load_split(args.data_root, args.scene, "val")
    df_test  = load_split(args.data_root, args.scene, "test")

    # Build windows
    traj_train = build_traj_dict(df_train)
    traj_val   = build_traj_dict(df_val)
    traj_test  = build_traj_dict(df_test)

    X_train, Y_train = build_windows(traj_train, obs_len=args.obs_len)
    X_val,   Y_val   = build_windows(traj_val,   obs_len=args.obs_len)
    X_test,  Y_test  = build_windows(traj_test,  obs_len=args.obs_len)

    print(f"Train windows: {X_train.shape} {Y_train.shape}")
    print(f"Val windows:   {X_val.shape} {Y_val.shape}")
    print(f"Test windows:  {X_test.shape} {Y_test.shape}")

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise RuntimeError("No windows were created for train/val. Check data formatting and obs_len.")

    # DataLoaders
    train_loader = DataLoader(TrajDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TrajDataset(X_val,   Y_val),   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TrajDataset(X_test,  Y_test),  batch_size=args.batch_size, shuffle=False, drop_last=False) if X_test.shape[0] else None

    # Model
    model = NextStepLSTM(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ade = float("inf")
    best_path = os.path.join(run_dir, "best_model.pth")
    last_path = os.path.join(run_dir, "last_model.pth")

    rows = []
    k_list = args.k

    # Training loop
    for ep in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, k_list=k_list)
        val_metrics   = run_epoch(model, val_loader,   device, optimizer=None,      k_list=k_list)

        row = {"epoch": ep}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        rows.append(row)

        # Print compact log
        msg = f"Epoch {ep:03d} | train MSE={train_metrics['mse']:.4f} ADE={train_metrics['ade']:.4f} || " \
              f"val MSE={val_metrics['mse']:.4f} ADE={val_metrics['ade']:.4f}"
        # Add Acc@k
        for k in k_list:
            msg += f" | val Acc@{k}={val_metrics[f'acc@{k}']:.3f}"
        print(msg)

        # Save best (based on val ADE)
        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved best model -> {best_path} (best val ADE={best_val_ade:.4f})")

        # Save CSV each epoch (safe if crash)
        df_metrics = pd.DataFrame(rows)
        df_metrics.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # Save last
    torch.save(model.state_dict(), last_path)
    print(f"✅ Saved last model -> {last_path}")

    # Plots
    df_metrics = pd.DataFrame(rows)

    # Rename columns for plotting convenience
    # We want: train_mse/train_ade and val_mse/val_ade and train_acc@k/val_acc@k
    # Right now keys are train_mse, train_ade, train_acc@..., etc already (thanks to row mapping)
    # But we used train_{k} where k is 'mse'/'ade'/'acc@...'
    # So columns are train_mse, train_ade, train_acc@0.5 ...
    save_plots(df_metrics, run_dir, k_list)

    # Final test evaluation with BEST model (if test exists)
    if test_loader is not None:
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = run_epoch(model, test_loader, device, optimizer=None, k_list=k_list)
        with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print("=== TEST (best model) ===")
        print(test_metrics)

    print("✅ All artifacts saved in:", run_dir)


if __name__ == "__main__":
    main()
