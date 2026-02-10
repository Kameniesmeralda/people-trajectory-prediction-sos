import argparse
import glob
import os
import numpy as np
import hashlib

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

def build_windows(files, obs_len: int, horizon: int):
    X_last_list, y_list = [], []

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
                X_last_list.append(X[-1])
                y_list.append(y)

    return np.asarray(X_last_list, dtype=np.float64), np.asarray(y_list, dtype=np.float64)

def hash_pairs(X_last: np.ndarray, y: np.ndarray, round_decimals: int = 4):
    Xr = np.round(X_last, round_decimals)
    yr = np.round(y, round_decimals)

    hashes = []
    for i in range(len(Xr)):
        s = f"{Xr[i,0]:.4f},{Xr[i,1]:.4f}|{yr[i,0]:.4f},{yr[i,1]:.4f}"
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        hashes.append(h)
    return set(hashes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=str)
    ap.add_argument("--data_root", default="data_real/raw", type=str)
    ap.add_argument("--obs_len", default=10, type=int)
    ap.add_argument("--pred_horizon", default=1, type=int)
    ap.add_argument("--round_decimals", default=4, type=int)
    args = ap.parse_args()

    splits = ["train", "val", "test"]
    sets = {}

    print("===================================")
    print(f"Scene: {args.scene}")
    print(f"Data root: {args.data_root}")
    print(f"obs_len={args.obs_len} horizon={args.pred_horizon}")
    print("===================================")

    for sp in splits:
        files = get_split_files(args.scene, sp, args.data_root)
        X_last, y = build_windows(files, args.obs_len, args.pred_horizon)
        hset = hash_pairs(X_last, y, args.round_decimals)
        sets[sp] = (hset, len(X_last), len(files))
        print(f"[{sp}] files={len(files)} windows={len(X_last)} unique_hashes={len(hset)}")

    def report(a, b):
        A, na, _ = sets[a]
        B, nb, _ = sets[b]
        inter = len(A.intersection(B))
        print(f"\nA3) Overlap check: {a} vs {b}")
        print(f"  overlap_hashes = {inter}")
        if min(len(A), len(B)) > 0:
            print(f"  overlap / min(unique) = {inter / min(len(A), len(B)):.6f}")
        print(f"  overlap / {a}_windows = {inter / max(1, na):.6f}")
        print(f"  overlap / {b}_windows = {inter / max(1, nb):.6f}")

    report("train", "test")
    report("val", "test")
    report("train", "val")

    print("\nInterpretation:")
    print("- Train/Test overlap should be 0. If not, strong sign of leakage.")
    print("- Tiny overlap can happen only due to rounding collisions; increase --round_decimals to 6 to be stricter.")

if __name__ == "__main__":
    main()
