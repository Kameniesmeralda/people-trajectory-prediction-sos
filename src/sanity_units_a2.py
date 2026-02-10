import argparse
import glob
import os
import numpy as np


def load_txt_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    all_rows = []
    for f in files:
        arr = np.loadtxt(f)  # expected columns: frame, ped_id, x, y (ETH/UCY style)
        if arr.ndim == 1:
            arr = arr[None, :]
        all_rows.append(arr)
    data = np.vstack(all_rows)
    return data, files


def build_windows(data, obs_len=10, pred_horizon=1):
    """
    Build windows per pedestrian track (sorted by frame).
    X: (N, obs_len, 2) absolute coords
    y: (N, 2) absolute coords at t+pred_horizon
    """
    # columns assumption: [frame, ped_id, x, y] or more; we take last two as x,y
    frame_col = 0
    pid_col = 1
    x_col = 2
    y_col = 3

    # group by ped_id
    pids = np.unique(data[:, pid_col]).astype(int)
    X_list, y_list = [], []

    for pid in pids:
        traj = data[data[:, pid_col] == pid]
        # sort by frame
        traj = traj[np.argsort(traj[:, frame_col])]
        coords = traj[:, [x_col, y_col]].astype(np.float32)

        T = len(coords)
        # need obs_len + pred_horizon steps
        for t in range(0, T - (obs_len + pred_horizon) + 1):
            X = coords[t : t + obs_len]
            y = coords[t + obs_len + pred_horizon - 1]  # next step (or horizon)
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        raise RuntimeError("No windows created. Check obs_len/pred_horizon and data format.")
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)


def acc_at_k(y_true, y_pred, ks):
    d = np.linalg.norm(y_true - y_pred, axis=1)
    out = {f"acc@{k}": float((d <= k).mean()) for k in ks}
    return out, float(d.mean()), float((d**2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="e.g. hotel")
    ap.add_argument("--data_root", default="data_real/raw", help="root containing scene folders")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--obs_len", type=int, default=10)
    ap.add_argument("--pred_horizon", type=int, default=1, help="1 means next-step prediction")
    ap.add_argument("--k", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    args = ap.parse_args()

    folder = os.path.join(args.data_root, args.scene, args.split)
    data, files = load_txt_files(folder)
    X, y = build_windows(data, obs_len=args.obs_len, pred_horizon=args.pred_horizon)

    last_pos = X[:, -1, :]
    step_dist = np.linalg.norm(y - last_pos, axis=1)

    # A2: unit / scale check
    mean_step = float(step_dist.mean())
    median_step = float(np.median(step_dist))
    p90_step = float(np.percentile(step_dist, 90))
    p99_step = float(np.percentile(step_dist, 99))

    # Baseline: predict last position
    y_pred = last_pos
    accs, ade, mse = acc_at_k(y, y_pred, args.k)

    print("===================================")
    print(f"Scene: {args.scene} | Split: {args.split}")
    print(f"Files: {len(files)}")
    print(f"Windows: {len(X)} | obs_len={args.obs_len} | horizon={args.pred_horizon}")
    print("===================================")
    print("A2) SCALE CHECK (distance between last obs and target)")
    print(f"mean_step   = {mean_step:.6f}")
    print(f"median_step = {median_step:.6f}")
    print(f"p90_step    = {p90_step:.6f}")
    print(f"p99_step    = {p99_step:.6f}")
    print()
    print("Baseline: predict last observed position (offset=0)")
    print(f"ADE  = {ade:.6f}")
    print(f"MSE  = {mse:.6f}")
    for kk, vv in accs.items():
        print(f"{kk} = {vv:.6f}")

    # Quick interpretation
    print("\nInterpretation:")
    if mean_step < min(args.k) / 10.0:
        print(f"- mean_step ({mean_step:.4f}) is MUCH smaller than smallest k ({min(args.k)}).")
        print("  => acc@k will naturally be very high. k thresholds are huge relative to your scale.")
    else:
        print("- mean_step is not extremely small vs k. High acc@k would be less likely from scale alone.")


if __name__ == "__main__":
    main()
