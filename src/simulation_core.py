import numpy as np

def run_boids_simulation(BoidClass, width=100, height=100, n_boids=30, n_frames=300, params=None, seed=0):
    if params is None:
        params = {
            "w_separation": 1.5,
            "w_alignement": 1.0,
            "w_cohesion": 0.8,
            "r_separation": 15.0,
            "r_alignement": 40.0,
            "r_cohesion": 50.0,
            "max_speed": 3.0,
        }

    rng = np.random.default_rng(seed)

    boids = [
        BoidClass(
            position=rng.random(2) * np.array([width, height]),
            velocity=(rng.random(2) - 0.5) * 10.0,
        )
        for _ in range(n_boids)
    ]

    trajectories = np.zeros((n_frames, n_boids, 2), dtype=np.float32)

    for t in range(n_frames):
        for b in boids:
            b.update(boids, params)
            b.apply_boundaries(width, height)

        positions = np.array([b.pos for b in boids], dtype=np.float32)
        trajectories[t] = positions

    return trajectories
