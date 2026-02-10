import argparse
import numpy as np

from lstmdatarealoffset import train_and_eval


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def decode_particle(x):
    """
    x = [lr_log10, dropout, hidden_dim_cont]
    """
    lr_log10 = float(x[0])   # ex: -5 .. -3
    dropout = float(x[1])    # 0 .. 0.3
    hidden_c = float(x[2])   # 16 .. 128 (continu)

    lr = 10 ** lr_log10
    hidden_dim = int(np.round(hidden_c / 8) * 8)
    hidden_dim = int(np.clip(hidden_dim, 16, 128))
    dropout = float(np.clip(dropout, 0.0, 0.3))

    return {"lr": lr, "dropout": dropout, "hidden_dim": hidden_dim}


def objective(x, base_cfg, repeats=1):
    """
    Moyenne sur plusieurs runs (seed différentes) pour réduire le bruit.
    Retourne val_ADE moyen (à MINIMISER).
    """
    cfg_part = decode_particle(x)
    scores = []
    for r in range(int(repeats)):
        cfg = dict(base_cfg)
        cfg.update(cfg_part)
        cfg["seed"] = int(base_cfg["seed"]) + r
        scores.append(train_and_eval(cfg))
    return float(np.mean(scores)), cfg_part


def pso_search(
    base_cfg,
    bounds,
    n_particles=10,
    iters=25,
    w=0.72, c1=1.4, c2=1.4,
    repeats=1,
):
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float32)
    hi = np.array([b[1] for b in bounds], dtype=np.float32)

    rng = np.random.default_rng(int(base_cfg["seed"]))

    X = rng.uniform(lo, hi, size=(n_particles, dim)).astype(np.float32)
    V = rng.normal(0, 0.1, size=(n_particles, dim)).astype(np.float32)

    pbest_X = X.copy()
    pbest_S = np.full((n_particles,), np.inf, dtype=np.float32)

    gbest_X = None
    gbest_S = np.inf
    gbest_cfg = None

    for t in range(1, iters + 1):
        print(f"\n=== PSO iter {t}/{iters} ===")
        for i in range(n_particles):
            s, cfg_part = objective(X[i], base_cfg, repeats=repeats)

            if s < pbest_S[i]:
                pbest_S[i] = s
                pbest_X[i] = X[i].copy()

            if s < gbest_S:
                gbest_S = s
                gbest_X = X[i].copy()
                gbest_cfg = cfg_part

            print(f"p{i:02d}  score={s:.4f}  cfg={cfg_part}")

        r1 = rng.random(size=(n_particles, dim), dtype=np.float32)
        r2 = rng.random(size=(n_particles, dim), dtype=np.float32)
        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X - X)
        X = clip(X + V, lo, hi)

        print(f"--> Best so far: {gbest_S:.4f} with {gbest_cfg}")

    return float(gbest_S), gbest_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="univ", choices=["eth", "hotel", "univ", "zara01", "zara02"])
    p.add_argument("--data_root", default="data_real/raw")
    p.add_argument("--obs_len", type=int, default=10)
    p.add_argument("--pred_horizon", type=int, default=30)

    # objective training budget (PSO)
    p.add_argument("--obj_epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=80, help="speed-up per epoch (PSO). Use None to disable.")

    # PSO
    p.add_argument("--particles", type=int, default=10)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # search bounds
    p.add_argument("--lr_log10_min", type=float, default=-5.0)
    p.add_argument("--lr_log10_max", type=float, default=-3.0)
    p.add_argument("--dropout_min", type=float, default=0.0)
    p.add_argument("--dropout_max", type=float, default=0.3)
    p.add_argument("--hidden_min", type=float, default=16.0)
    p.add_argument("--hidden_max", type=float, default=128.0)

    args = p.parse_args()

    base_cfg = {
        "seed": args.seed,
        "scene": args.scene,
        "data_root": args.data_root,
        "obs_len": args.obs_len,
        "pred_horizon": args.pred_horizon,
        "epochs": args.obj_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "dropout": 0.1,     # overwritten by PSO particles
        "hidden_dim": 64,   # overwritten by PSO particles
        "lr": 1e-4,         # overwritten by PSO particles
        "ks": [0.25, 0.5, 1.0, 2.0, 4.0],
    }
    if args.max_batches is not None:
        base_cfg["max_batches"] = args.max_batches


    bounds = [
        (args.lr_log10_min, args.lr_log10_max),
        (args.dropout_min, args.dropout_max),
        (args.hidden_min, args.hidden_max),
    ]

    best_score, best_cfg = pso_search(
        base_cfg=base_cfg,
        bounds=bounds,
        n_particles=args.particles,
        iters=args.iters,
        repeats=args.repeats,
    )

    print("\n======================")
    print("BEST val_ADE:", best_score)
    print("BEST cfg:", best_cfg)
    print("======================")
    print("=> Training command example:")
    print(
        f"python src/lstmdatarealoffset.py --scene {args.scene} --data_root {args.data_root} "
        f"--obs_len {args.obs_len} --pred_horizon {args.pred_horizon} --epochs 100 "
        f"--lr {best_cfg['lr']:.8f} --hidden_dim {best_cfg['hidden_dim']} --dropout {best_cfg['dropout']:.3f} "
        f"--num_layers {args.num_layers} --k 0.25 0.5 1.0 2.0 4.0 --run_name final_pso_{args.scene}_h{args.pred_horizon}"
    )


if __name__ == "__main__":
    main()
