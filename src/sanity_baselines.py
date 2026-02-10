import argparse, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Reuse your dataset loader from lstmdatarealoffset.py by importing if possible.
# If not, we implement minimal window loading via your existing script functions.
# ---- IMPORTANT ----
# This script expects that your run folder already contains the saved windows:
#   train_windows.npz / val_windows.npz / test_windows.npz
# If you don't have them, see A4 to enable saving windows.

def load_windows_npz(run_dir, split):
    path = os.path.join(run_dir, f"{split}_windows.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run with --save_windows first (see instructions).")
    data = np.load(path)
    X = data["X"]  # (N, obs_len, 2) absolute coords
    y = data["y"]  # (N, 2) absolute coords (target next point)
    return X, y

def acc_at_k(y_true, y_pred, ks):
    # y_true/y_pred: (N,2)
    d = np.linalg.norm(y_true - y_pred, axis=1)  # Euclidean distance
    out = {}
    for k in ks:
        out[f"acc@{k}"] = float((d <= k).mean())
    return out, float(d.mean()), float(((y_true - y_pred)**2).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--k", nargs="+", type=float, default=[0.5,1.0,2.0,4.0])
    args = ap.parse_args()

    X, y = load_windows_npz(args.run_dir, args.split)
    last_pos = X[:, -1, :]            # (N,2)
    y_pred = last_pos                 # baseline: no movement

    accs, ade, mse = acc_at_k(y, y_pred, args.k)

    print(f"=== ZERO-OFFSET BASELINE on {args.split.upper()} ===")
    print(f"ADE: {ade:.6f}")
    print(f"MSE: {mse:.6f}")
    for kk, vv in accs.items():
        print(f"{kk}: {vv:.6f}")

if __name__ == "__main__":
    main()
