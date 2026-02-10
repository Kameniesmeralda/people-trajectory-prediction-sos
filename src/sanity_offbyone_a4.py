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
    # ✅ Case 1 (your structure): data_root/scene/split/*.txt
    p1 = os.path.join(data_root, scene, split, "*.txt")
    files = sorted(glob.glob(p1))
    if len(files) > 0:
        return files

    # Fallback: data_root/*.txt (older structure)
    # train/val often have *_train.txt, *_val.txt
    if split in ("train", "val"):
        p2 = os.path.join(data_root, f"*_{split}.txt")
        files = sorted(glob.glob(p2))
        if len(files) > 0:
            return files

    # test: try *scene*.txt
    p3 = os.path.join(data_root, f"*{scene}*.txt")
    files = sorted(glob.glob(p3))
    files = [f for f in files if not f.endswith("_train.txt") and not f.endswith("_val.txt")]
    if len(files) > 0:
        return files

    raise FileNotFoundError(
        f"No files found. Tried:\n  {p1}\n  {p2 if split in ('train','val') else ''}\n  {p3}"
    )

def build_windows_with_frames(files, obs_len: int, horizon: int):
    X_last_list, y_list = [], []
    f_last_list, f_y_list = [], []

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
                f_last_list.append(f[s+obs_len-1])
                f_y_list.append(f[s+obs_len+horizon-1])

    return (np.asarray(X_last_list), np.asarray(y_list),
            np.asarray(f_last_list), np.asarray(f_y_list))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=str)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--data_root", default="data_real/raw", type=str)
    ap.add_argument("--obs_len", default=10, type=int)
    ap.add_argument("--pred_horizon", default=1, type=int)
    args = ap.parse_args()

    files = get_split_files(args.scene, args.split, args.data_root)
    X_last, y, f_last, f_y = build_windows_with_frames(files, args.obs_len, args.pred_horizon)

    if len(X_last) == 0:
        raise RuntimeError("No windows built. Check obs_len/horizon and data.")

    same = np.linalg.norm(y - X_last, axis=1) < 1e-12
    same_ratio = float(same.mean())

    gaps = f_y - f_last
    min_gap = int(gaps.min())
    bad_gap_ratio = float((gaps <= 0).mean())

    print("===================================")
    print(f"Scene: {args.scene} | Split: {args.split}")
    print(f"Files: {len(files)} | Windows: {len(X_last)}")
    print(f"obs_len={args.obs_len} horizon={args.pred_horizon}")
    print("===================================")

    print("A4) OFF-BY-ONE / WINDOW CONSISTENCY")
    print(f"y == last_obs ratio    = {same_ratio:.6f}")
    print(f"frame_gap min          = {min_gap}")
    print(f"frame_gap median       = {np.median(gaps):.3f}")
    print(f"frame_gap p90          = {np.percentile(gaps, 90):.3f}")
    print(f"non-positive gap ratio = {bad_gap_ratio:.6f}")

    print("\nInterpretation:")
    print("- y==last_obs should be ~0. If it's high, you might be predicting t instead of t+1.")
    print("- frame gaps must be >= 1. If you see <=0, indexing is broken.")
    print("- If median gap is huge, frames may be subsampled; that’s OK but note it.")

if __name__ == "__main__":
    main()
