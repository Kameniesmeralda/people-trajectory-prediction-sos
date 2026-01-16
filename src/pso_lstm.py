"""
pso_lstm.py
Optimisation des hyperparamètres du LSTM avec PSO.

Hyperparamètres optimisés :
- hidden_dim    (taille de l'état caché)
- learning_rate (log10)
- num_layers    (1, 2 ou 3)

La fonction objectif :
→ entraîner un petit LSTM quelques époques sur un sous-ensemble
   du dataset (X, Y) et mesurer la MSE sur un set de validation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from copy import deepcopy

# ============================================================
# 1. Chargement du dataset (X.npy, Y.npy)
# ============================================================

X = np.load("../data/X.npy")  # (N, SEQ_LEN, 2)
Y = np.load("../data/Y.npy")  # (N, 2)

X_torch = torch.tensor(X, dtype=torch.float32)
Y_torch = torch.tensor(Y, dtype=torch.float32)

N = X_torch.shape[0]
print("Données chargées :", X_torch.shape, Y_torch.shape)

# Split train / validation (80% / 20%)
indices = np.arange(N)
np.random.shuffle(indices)
split = int(0.8 * N)
train_idx = indices[:split]
val_idx = indices[split:]

X_train, Y_train = X_torch[train_idx], Y_torch[train_idx]
X_val,   Y_val   = X_torch[val_idx],   Y_torch[val_idx]

print(f"Train : {X_train.shape[0]} exemples - Val : {X_val.shape[0]} exemples")


# ============================================================
# 2. Dataset PyTorch
# ============================================================

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ============================================================
# 3. Modèle LSTM paramétrable
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        pred = self.fc(last_out)
        return pred


# ============================================================
# 4. Fonction objectif pour le PSO
#    → retourne la MSE sur validation
# ============================================================

def evaluate_hyperparams(hidden_dim, lr, num_layers,
                         train_subset_size=2000,
                         val_subset_size=1000,
                         epochs=5,
                         batch_size=64):
    """
    hidden_dim : int
    lr         : float
    num_layers : int
    """

    # Sous-échantillonnage pour garder un coût raisonnable
    train_size = min(train_subset_size, X_train.shape[0])
    val_size   = min(val_subset_size,   X_val.shape[0])

    train_indices = np.random.choice(X_train.shape[0], train_size, replace=False)
    val_indices   = np.random.choice(X_val.shape[0],   val_size,   replace=False)

    train_dataset = TrajectoryDataset(X_train[train_indices], Y_train[train_indices])
    val_dataset   = TrajectoryDataset(X_val[val_indices],   Y_val[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Modèle + perte + optimiseur
    model = LSTMModel(hidden_dim=hidden_dim, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Entraînement rapide
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()

    # Évaluation sur validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    return val_loss


# ============================================================
# 5. Implémentation PSO
# ============================================================

class Particle:
    def __init__(self, bounds):
        """
        bounds = [ (min_hidden, max_hidden),
                   (min_log_lr, max_log_lr),
                   (min_layers, max_layers) ]
        """
        self.dim = len(bounds)

        # Position initiale aléatoire dans les bornes
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        # Vitesse initiale
        self.velocity = np.zeros(self.dim)

        # Meilleur perso
        self.best_position = deepcopy(self.position)
        self.best_value = np.inf


def pso_optimize(num_particles=15, num_iterations=30):
    """
    PSO pour optimiser [hidden_dim, log10(lr), num_layers].
    """

    # Bornes de recherche
    # hidden_dim      : [16, 128]
    # log10(lr)       : [-4, -1]  → lr entre 1e-4 et 1e-1
    # num_layers      : [1, 3]
    bounds = [
        (40, 100),    # hidden_dim
        (-2, -0.7),     # log10(lr)
        (1, 2)        # num_layers
    ]

    # Paramètres PSO
    w  = 0.7   # inertie
    c1 = 1.5   # composante cognitive
    c2 = 1.5   # composante sociale

    # Initialisation du swarm
    particles = [Particle(bounds) for _ in range(num_particles)]

    global_best_position = None
    global_best_value = np.inf

    history_best = []

    for it in range(num_iterations):
        print(f"\n=== Itération PSO {it+1}/{num_iterations} ===")

        for i, p in enumerate(particles):
            # Décodage des hyperparamètres à partir de la position
            hidden_dim = int(np.round(p.position[0]))
            hidden_dim = int(np.clip(hidden_dim, bounds[0][0], bounds[0][1]))

            log_lr = p.position[1]
            log_lr = np.clip(log_lr, bounds[1][0], bounds[1][1])
            lr = 10 ** log_lr

            num_layers = int(np.round(p.position[2]))
            num_layers = int(np.clip(num_layers, bounds[2][0], bounds[2][1]))

            print(f"  Particule {i+1} → hidden={hidden_dim}, lr={lr:.2e}, layers={num_layers}")

            # Évaluation de la fonction objectif (MSE de validation)
            value = evaluate_hyperparams(hidden_dim, lr, num_layers)

            print(f"    → MSE val = {value:.4f}")

            # Mise à jour du meilleur personnel
            if value < p.best_value:
                p.best_value = value
                p.best_position = deepcopy(p.position)

            # Mise à jour du meilleur global
            if value < global_best_value:
                global_best_value = value
                global_best_position = deepcopy(p.position)

        history_best.append(global_best_value)
        print(f" >> Meilleur global après itération {it+1} : {global_best_value:.4f}")

        # Mise à jour des vitesses / positions
        for p in particles:
            r1 = np.random.rand(p.dim)
            r2 = np.random.rand(p.dim)

            cognitive = c1 * r1 * (p.best_position - p.position)
            social    = c2 * r2 * (global_best_position - p.position)

            p.velocity = w * p.velocity + cognitive + social
            p.position = p.position + p.velocity

    # Décodage final du meilleur global
    best_hidden = int(np.round(global_best_position[0]))
    best_hidden = int(np.clip(best_hidden, bounds[0][0], bounds[0][1]))

    best_log_lr = np.clip(global_best_position[1], bounds[1][0], bounds[1][1])
    best_lr = 10 ** best_log_lr

    best_layers = int(np.round(global_best_position[2]))
    best_layers = int(np.clip(best_layers, bounds[2][0], bounds[2][1]))

    print("\n=============================")
    print("   Résultat final du PSO")
    print("=============================")
    print(f"Best hidden_dim  = {best_hidden}")
    print(f"Best learning_rate = {best_lr:.2e}")
    print(f"Best num_layers  = {best_layers}")
    print(f"Best MSE val     = {global_best_value:.4f}")

    # Courbe de convergence
    plt.figure(figsize=(6,4))
    plt.plot(history_best, marker="o")
    plt.title("Convergence du PSO (meilleure MSE validation)")
    plt.xlabel("Itération PSO")
    plt.ylabel("MSE validation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_hidden, best_lr, best_layers, global_best_value


# ============================================================
# 6. Lancement du PSO
# ============================================================

if __name__ == "__main__":
    os.makedirs("../models", exist_ok=True)
    best_hidden, best_lr, best_layers, best_mse = pso_optimize()

    # Option : enregistrer les hyperparamètres trouvés
    config = {
        "hidden_dim": best_hidden,
        "learning_rate": best_lr,
        "num_layers": best_layers,
        "val_mse": best_mse
    }
    np.save("../models/best_lstm_pso_config_1.npy", config)
    print("\nConfig PSO sauvegardée dans ../models/best_lstm_pso_config.npy")
