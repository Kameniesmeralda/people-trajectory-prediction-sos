from boid import Boid
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------
# 1️⃣ PARAMÈTRES DE SIMULATION
# ------------------------------------------------

# Dimensions de l'espace
width, height = 100, 100

# Nombre d'agents
N_boids = 30
n_frames = 300       # nombre d'images / pas de temps

# Paramètres comportementaux
params = {
    # poids des trois forces
    'w_separation': 1.5,  # poids de la séparation
    'w_alignement': 1.0,  # poids de l'alignement
    'w_cohesion': 0.8,  # poids de la cohésion

    # rayons d'interaction
    'r_separation': 15.0,  # rayon d'évitement
    'r_alignement': 40.0,  # rayon d'alignement
    'r_cohesion': 50.0,  # rayon de cohésion

    # vitesse maximale
    'max_speed': 3.0
}

# ------------------------------------------------
# 2️⃣ INITIALISATION DES BOIDS
# ------------------------------------------------

boids = [
    Boid(
        position=np.random.rand(2) * [width, height],  # position aléatoire
        velocity=(np.random.rand(2) - 0.5) * 10  # vitesse aléatoire
    )
    for _ in range(N_boids)
]

# Tableau pour stocker les trajectoires
# shape: (n_frames, N_boids, 2)
trajectories = np.zeros((n_frames, N_boids, 2))


# ------------------------------------------------
# 3️⃣ CONFIGURATION DE LA FIGURE MATPLOTLIB
# ------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_title("Simulation du modèle Boids (Reynolds, 1987)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# On crée un nuage de points initial
positions = np.array([b.pos for b in boids])
scat = ax.scatter(positions[:, 0], positions[:, 1], color="royalblue", s=30)


# ------------------------------------------------
# 4️⃣ FONCTION DE MISE À JOUR (pour l’animation)
# ------------------------------------------------

def update(frame):
    """
    Fonction appelée à chaque frame de l’animation.
    Met à jour la position et la vitesse des boids,
    puis rafraîchit l’affichage.
    """
    for b in boids:
        b.update(boids, params)
        b.apply_boundaries(width, height)

    # enregistrement des positions dans le tableau de trajectoires
    positions = np.array([b.pos for b in boids])
    trajectories[frame] = positions

    # Mise à jour graphique
    scat.set_offsets(positions)
    ax.set_title(f"Simulation Boids - Frame {frame + 1}/{n_frames}")

    return scat,  # matplotlib demande de renvoyer l’objet modifié


# ------------------------------------------------
# 5️⃣ CRÉATION DE L’ANIMATION
# ------------------------------------------------

# On définit une animation avec 300 frames (≈ 10 secondes à 30 fps)
animation = FuncAnimation(
    fig,  # la figure matplotlib
    update,  # la fonction appelée à chaque frame
    frames=n_frames,  # nombre d'images
    interval=30,  # intervalle en ms entre frames (≈33 ms → 30 fps)
    blit=True  # pour optimiser les performances
)

# ------------------------------------------------
# 6️⃣ ENREGISTREMENT DE LA VIDÉO
# ------------------------------------------------

# Sauvegarde la vidéo sous forme d'un gif
# ⚠️ Nécessite que ffmpeg soit installé sur ton ordinateur
animation.save("boids_simulation.gif", writer="pillow", fps=25)
print("🎥 Vidéo enregistrée sous le nom 'boids_simulation.gif' ✅")

# ------------------------------------------------
# AFFICHAGE FINAL (facultatif si tu veux juste enregistrer)
# ------------------------------------------------

plt.show()
print("Simulation terminée ✅")

# ------------------------------------------------
# 7️⃣ SAUVEGARDE DES TRAJECTOIRES POUR LE LSTM
# ------------------------------------------------

os.makedirs("../data", exist_ok=True)   # dossier data à la racine du projet
np.save("../data/boids_trajectories.npy", trajectories)
print("💾 Trajectoires sauvegardées dans '../data/boids_trajectories.npy'")
print("Shape des données :", trajectories.shape)