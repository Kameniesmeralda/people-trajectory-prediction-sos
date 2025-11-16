"""
---------------------------------------------------
DATASET.PY
Génération du dataset LSTM à partir des trajectoires Boids
------------------------------------------------------------

Ce script :
1. Charge le fichier boids_trajectories.npy généré par simulation.py
2. Transforme les trajectoires en séquences temporelles pour le LSTM
3. Construit deux fichiers :
        - X.npy → séquences d'entrée
        - Y.npy → positions futures (cibles à prédire)
4. Sauvegarde le tout dans /data/
------------------------------------------------------------
"""

import numpy as np
import os


# =====================================================================
# 1. PARAMÈTRES DU DATASET
# =====================================================================

# Longueur de la séquence temporelle que tu veux donner au LSTM
# Exemple : avec SEQ_LEN = 10 → le LSTM voit les 10 positions précédentes
SEQ_LEN = 10    # (Tu pourras ajuster plus tard !)


# =====================================================================
# 2. CHARGER LE FICHIER DE TRAJECTOIRES
# =====================================================================

# Fichier généré par simulation.py
# Format : (n_frames, n_boids, 2)
# Exemple : (300, 30, 2)
trajectories = np.load("../data/boids_trajectories.npy")

# Récupérer les dimensions
n_frames, n_boids, coord_dim = trajectories.shape

print("------------------------------------------------")
print(" Trajectoires chargées depuis simulation.py")
print(" Shape :", trajectories.shape)
print(" n_frames =", n_frames)
print(" n_boids =", n_boids)
print(" coord_dim =", coord_dim)
print("------------------------------------------------")


# =====================================================================
# 3. INITIALISER LES LISTES POUR LES SÉQUENCES ET LES CIBLES
# =====================================================================

X = []   # Séquences d'entrée pour le LSTM → shape finale (N, SEQ_LEN, 2)
Y = []   # Cibles → position suivante → shape finale (N, 2)


# =====================================================================
# 4. CONSTRUCTION DU DATASET POUR CHAQUE BOID
# =====================================================================

"""
Rappel sur le principe d'un LSTM :

On lui donne une séquence temporelle :
    [pos_t, pos_t+1, ..., pos_t+SEQ_LEN-1]

Il doit prédire la position suivante :
    pos_t+SEQ_LEN

Donc on glisse une fenêtre temporelle de taille SEQ_LEN
sur la trajectoire de chaque boid.
"""

for boid_idx in range(n_boids):

    # Extraire la trajectoire complète du boid n° boid_idx
    # Shape : (n_frames, 2)
    boid_traj = trajectories[:, boid_idx, :]

    # On crée les séquences temporelles pour CE boid
    for t in range(n_frames - SEQ_LEN):

        # Séquence d'entrée → positions de t à t+SEQ_LEN (NON INCLUS)
        input_seq = boid_traj[t : t + SEQ_LEN]     # shape (SEQ_LEN, 2)

        # Cible → position juste après la séquence
        target = boid_traj[t + SEQ_LEN]            # shape (2,)

        # On ajoute à la liste globale
        X.append(input_seq)
        Y.append(target)


# =====================================================================
# 5. CONVERSION EN TABLEAUX NUMPY
# =====================================================================

X = np.array(X)
Y = np.array(Y)

print("\n------------------------------------------------")
print(" DATASET CONSTRUIT AVEC SUCCÈS !")
print(" Shape X =", X.shape, " → séquences (input)")
print(" Shape Y =", Y.shape, " → positions futures (target)")
print("------------------------------------------------")
print(" Exemple :")
print(" - X[i] contient une séquence de", SEQ_LEN, "positions")
print(" - Y[i] contient la position suivante")
print("------------------------------------------------\n")


# =====================================================================
# 6. SAUVEGARDE DU DATASET
# =====================================================================

# Création du dossier si besoin
os.makedirs("../data", exist_ok=True)

# Sauvegarde des fichiers
np.save("../data/X.npy", X)
np.save("../data/Y.npy", Y)

print("💾 Fichiers sauvegardés avec succès !")
print("   → ../data/X.npy")
print("   → ../data/Y.npy")
print("------------------------------------------------")
print("Tu peux maintenant passer à l'entraînement du LSTM 🔥!")
print("------------------------------------------------")
