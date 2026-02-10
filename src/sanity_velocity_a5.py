import argparse
import glob
import os
import numpy as np

def load_txt(path: str) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data

def get_split_files(scene: str, split: str, data_root: str):
    p1 = os.path.join(data_root, scene, split, "*.txt")
    files = sorted(glob.glob(p1))
    if len(files) > 0:
        return files

    if split in ("train", "val"):
        p2 = os.path.join(data_root, f"*_{split}.txt")
        files = sorted(glob.glob(p2))
        if len(files) > 0:
            return files

    p3 = os.path.join(data_root, f"*{scene}*.txt")
    files = sorted(glob.glob(p3))
    files = [f for f in files if not f.endswith("_train.txt") and not f.endswith("_val.txt")]
    if len(files) > 0:
        return files

    raise FileNotFoundError(f"No files found for scene={scene}, split={split} under {data_root}")

def build_full_windows(files, obs_len: int, horizon: int):
    X_list, y_list = [], []

    for fp in files:
        arr = load_txt(fp)
        frame = arr[:, 0].astype(np.int64)
        ped = arr[:, 1].astype(np.int64)
        xy = arr[:, -2:].astype(np.float64)

        for pid in np.unique(ped):
            m = ped == pid
            f = frame[m]
            p = xy[m]
            order = np.argsort(f)
            f = f[order]
            p = p[order]

            T = len(p)
            max_start = T - (obs_len + horizon) + 1
            if max_start <= 0:
                continue

            for s in range(max_start):
                X = p[s:s+obs_len]
                y = p[s+obs_len+horizon-1]
                X_list.append(X)
                y_list.append(y)

    return np.asarray(X_list, dtype=np.float64), np.asarray(y_list, dtype=np.float64)

def compute_metrics(y_true, y_pred, ks):
    d = np.linalg.norm(y_true - y_pred, axis=1)
    ade = float(d.mean())
    mse = float(((y_true - y_pred) ** 2).mean())
    acc = {f"acc@{k}": float((d <= k).mean()) for k in ks}
    return ade, mse, acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=str)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--data_root", default="data_real/raw", type=str)
    ap.add_argument("--obs_len", default=10, type=int)
    ap.add_argument("--pred_horizon", default=1, type=int)
    ap.add_argument("--k", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    args = ap.parse_args()

    files = get_split_files(args.scene, args.split, args.data_root)
    X, y = build_full_windows(files, args.obs_len, args.pred_horizon)
    if len(X) == 0:
        raise RuntimeError("No windows built. Check obs_len/horizon and data.")

    last = X[:, -1, :]
    prev = X[:, -2, :] if args.obs_len >= 2 else X[:, -1, :]

    pred_last = last.copy()
    v = (last - prev)
    pred_cv = last + v * float(args.pred_horizon)

    ade1, mse1, acc1 = compute_metrics(y, pred_last, args.k)
    ade2, mse2, acc2 = compute_metrics(y, pred_cv, args.k)

    print("===================================")
    print(f"Scene: {args.scene} | Split: {args.split}")
    print(f"Files: {len(files)} | Windows: {len(X)}")
    print(f"obs_len={args.obs_len} horizon={args.pred_horizon} | k={args.k}")
    print("===================================")

    print("\nA5) BASELINES")
    print("\nBaseline: LAST POSITION")
    print(f"ADE = {ade1:.6f} | MSE = {mse1:.6f}")
    for k in args.k:
        print(f"acc@{k} = {acc1[f'acc@{k}']:.6f}")

    print("\nBaseline: CONSTANT VELOCITY (CV)")
    print(f"ADE = {ade2:.6f} | MSE = {mse2:.6f}")
    for k in args.k:
        print(f"acc@{k} = {acc2[f'acc@{k}']:.6f}")

    print("\nInterpretation:")
    print("- If CV is already very strong, the task/horizon is easy; acc@1/2/4 will saturate.")
    print("- The meaningful metric becomes ADE/MSE and acc@0.5 vs CV.")

if __name__ == "__main__":
    main()
