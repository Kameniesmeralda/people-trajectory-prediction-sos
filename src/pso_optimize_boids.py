import os
import numpy as np

from boid import Boid
from simulation_core import run_boids_simulation
from boids_metrics import boids_objective

# --- NEW imports for GIF ---
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def decode_particle(x):
    return {
        "w_separation": float(x[0]),
        "w_alignement": float(x[1]),
        "w_cohesion": float(x[2]),
        "r_separation": float(x[3]),
        "r_alignement": float(x[4]),
        "r_cohesion": float(x[5]),
        "max_speed": float(x[6]),
    }


# ---------- NEW: simple renderer -> GIF ----------
def save_trajectories_gif(
    trajectories: np.ndarray,
    out_path: str,
    width: float,
    height: float,
    fps: int = 20,
    stride: int = 2,
    title: str = "",
    point_size: int = 30,
):
    """
    trajectories: (n_frames, n_boids, 2)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_frames = trajectories.shape[0]
    n_boids = trajectories.shape[1]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    scat = ax.scatter([], [], s=point_size)

    with imageio.get_writer(out_path, mode="I", fps=fps) as writer:
        for t in range(0, n_frames, stride):
            pts = trajectories[t]  # (n_boids, 2)
            scat.set_offsets(pts)

            ax.set_xlabel(f"Frame {t}/{n_frames-1} | Boids={n_boids}")

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())  # shape (H, W, 4) already
            img = img[..., :3].copy()  # convert RGBA -> RGB
            writer.append_data(img)

    plt.close(fig)
    print(f"🎞️ GIF saved: {out_path}")


def main():
    width, height = 100, 100
    n_boids = 30
    n_frames = 300

    # bounds
    lo = np.array([0.0, 0.0, 0.0,  2.0,  5.0,  5.0, 0.5], dtype=np.float32)
    hi = np.array([5.0, 5.0, 5.0, 30.0, 80.0, 90.0, 6.0], dtype=np.float32)

    n_particles = 12
    n_iters = 25

    w = 0.7
    c1 = 1.4
    c2 = 1.4

    rng = np.random.default_rng(0)

    X = rng.uniform(lo, hi, size=(n_particles, lo.shape[0])).astype(np.float32)
    V = np.zeros_like(X)

    pbest_X = X.copy()
    pbest_score = np.full((n_particles,), np.inf, dtype=np.float32)

    gbest_X = None
    gbest_score = np.inf
    gbest_info = None
    gbest_traj = None

    # -------------------------
    # NEW: baseline simulation (before PSO)
    # -------------------------
    # Option A (safe): use midpoints of bounds as "reasonable defaults"
    baseline_vec = (lo + hi) / 2.0
    baseline_params = decode_particle(baseline_vec)

    baseline_traj = run_boids_simulation(
        BoidClass=Boid,
        width=width,
        height=height,
        n_boids=n_boids,
        n_frames=n_frames,
        params=baseline_params,
        seed=42,
    )
    baseline_score, baseline_info = boids_objective(baseline_traj, collision_radius=2.0)

    print("\n=== BASELINE (before PSO) ===")
    print("Baseline score:", baseline_score)
    print("Baseline info:", baseline_info)
    print("Baseline params:", baseline_params)

    print("\n=== PSO Boids (objectif: collisions↓, dispersion↓, polarization↑) ===\n")

    for it in range(1, n_iters + 1):
        print(f"--- Iter {it}/{n_iters} ---")
        for i in range(n_particles):
            params = decode_particle(X[i])
            traj = run_boids_simulation(
                BoidClass=Boid,
                width=width,
                height=height,
                n_boids=n_boids,
                n_frames=n_frames,
                params=params,
                seed=42,
            )

            score, info = boids_objective(traj, collision_radius=2.0)

            if score < pbest_score[i]:
                pbest_score[i] = score
                pbest_X[i] = X[i].copy()

            if score < gbest_score:
                gbest_score = score
                gbest_X = X[i].copy()
                gbest_info = info
                gbest_traj = traj

            print(
                f"  p{i+1:02d} score={score:.3f} | "
                f"col={info['collisions']:.3f} disp={info['dispersion']:.3f} pol={info['polarization']:.3f}"
            )

        r1 = rng.random(size=X.shape).astype(np.float32)
        r2 = rng.random(size=X.shape).astype(np.float32)

        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X[None, :] - X)
        X = X + V
        X = clamp(X, lo, hi)

        print(f">> Best global score: {gbest_score:.3f} | info={gbest_info}\n")

    os.makedirs("../models", exist_ok=True)
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    best_params = decode_particle(gbest_X)
    np.save("../models/best_boids_pso_params.npy", best_params, allow_pickle=True)
    np.save("../data/boids_trajectories_best_pso.npy", gbest_traj)

    # -------------------------
    # NEW: save two GIFs (before/after)
    # -------------------------
    save_trajectories_gif(
        baseline_traj,
        out_path="../results/boids_baseline.gif",
        width=width,
        height=height,
        fps=20,
        stride=2,
        title="Boids baseline (before PSO)",
    )
    save_trajectories_gif(
        gbest_traj,
        out_path="../results/boids_pso_optimized.gif",
        width=width,
        height=height,
        fps=20,
        stride=2,
        title="Boids optimized (after PSO)",
    )

    print("=== DONE ===")
    print("Best params saved to: ../models/best_boids_pso_params.npy")
    print("Best trajectories saved to: ../data/boids_trajectories_best_pso.npy")
    print("GIFs saved to: ../results/boids_baseline.gif and ../results/boids_pso_optimized.gif")
    print("Best score:", gbest_score)
    print("Best info:", gbest_info)
    print("Best params:", best_params)


if __name__ == "__main__":
    main()
