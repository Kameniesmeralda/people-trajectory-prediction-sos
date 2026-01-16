import numpy as np

def _pairwise_distances(positions):
    diff = positions[:, None, :] - positions[None, :, :]
    d2 = (diff ** 2).sum(axis=-1)
    return np.sqrt(d2 + 1e-12)

def collisions_per_frame(positions, collision_radius=2.0):
    D = _pairwise_distances(positions)
    n = positions.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.sum((D < collision_radius) & mask)

def dispersion(positions):
    center = positions.mean(axis=0)
    return np.mean(np.linalg.norm(positions - center, axis=1))

def polarization_from_velocities(velocities):
    speeds = np.linalg.norm(velocities, axis=1) + 1e-12
    vhat = velocities / speeds[:, None]
    return np.linalg.norm(vhat.mean(axis=0))

def estimate_velocities(trajectories):
    v = trajectories[1:] - trajectories[:-1]
    return v

def boids_objective(trajectories, collision_radius=2.0):
    # collisions + dispersion + (1 - alignment)
    T, N, _ = trajectories.shape

    col = 0.0
    disp = 0.0

    for t in range(T):
        pos = trajectories[t]
        col += collisions_per_frame(pos, collision_radius=collision_radius)
        disp += dispersion(pos)

    col /= T
    disp /= T

    v = estimate_velocities(trajectories)              # (T-1, N, 2)
    pol = np.mean([polarization_from_velocities(vt) for vt in v])

    # pondérations (tu peux ajuster)
    return 5.0 * col + 1.0 * disp + 50.0 * (1.0 - pol), {
        "collisions": float(col),
        "dispersion": float(disp),
        "polarization": float(pol),
    }
