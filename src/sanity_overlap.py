import argparse, os, hashlib
import numpy as np

def load_npz(run_dir, split):
    p = os.path.join(run_dir, f"{split}_windows.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Generate windows with --save_windows first.")
    d = np.load(p)
    return d["X"], d["y"]

def hash_window(Xi, yi):
    b = Xi.tobytes() + yi.tobytes()
    return hashlib.md5(b).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    Xtr, ytr = load_npz(args.run_dir, "train")
    Xva, yva = load_npz(args.run_dir, "val")
    Xte, yte = load_npz(args.run_dir, "test")

    def make_set(X, y, maxn=200000):
        n = min(len(X), maxn)
        s = set()
        for i in range(n):
            s.add(hash_window(X[i], y[i]))
        return s, n

    S_tr, ntr = make_set(Xtr, ytr)
    S_va, nva = make_set(Xva, yva)
    S_te, nte = make_set(Xte, yte)

    inter_tr_te = len(S_tr & S_te)
    inter_va_te = len(S_va & S_te)
    inter_tr_va = len(S_tr & S_va)

    print("=== OVERLAP CHECK (by exact window hash) ===")
    print(f"train windows hashed: {ntr}")
    print(f"val   windows hashed: {nva}")
    print(f"test  windows hashed: {nte}")
    print(f"overlap train-test: {inter_tr_te}")
    print(f"overlap val-test:   {inter_va_te}")
    print(f"overlap train-val:  {inter_tr_va}")

if __name__ == "__main__":
    main()
