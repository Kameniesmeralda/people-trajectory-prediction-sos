import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# =========================
# 0. Paramètres d'expérience
# =========================
HIDDEN_DIM = 61
NUM_LAYERS = 1
LEARNING_RATE = 1e-4   # change ici: 1e-2, 1e-3, 1e-4
EPOCHS = 100
BATCH_SIZE = 64

# "Accuracy" version trajectoire: % de points prédits à moins de tol
TOLERANCES = [2.0, 5.0, 10.0]  # en unités de ton espace (0..100)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", DEVICE)

# =========================
# 1. Chargement des données
# =========================
X = np.load("../data/X.npy")   # (N, 10, 2)
Y = np.load("../data/Y.npy")   # (N, 2)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

N = X.shape[0]
print("Données :", X.shape, Y.shape)

# Train / Val (80 / 20)
indices = np.arange(N)
np.random.shuffle(indices)
split = int(0.8 * N)
train_idx = indices[:split]
val_idx   = indices[split:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]

print(f"Train : {X_train.shape[0]}  |  Val : {X_val.shape[0]}")

# =========================
# 2. Dataset & DataLoader
# =========================
class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(TrajectoryDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TrajectoryDataset(X_val,   Y_val),   batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 3. Modèle LSTM
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=61, num_layers=1, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq_len, hidden_dim)
        last_out = out[:, -1, :]    # dernier pas
        pred = self.fc(last_out)    # (batch, 2)
        return pred

model = LSTMModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# 4. Fonctions métriques
# =========================
@torch.no_grad()
def compute_metrics(pred, target):
    """
    pred, target : (batch, 2)
    Retourne:
      mse (float),
      ade (float)  = mean L2 distance
      fde (float)  = mean L2 distance (ici identique à ADE car 1 seul point)
      acc@tol dict = % de distances <= tol
    """
    mse = torch.mean((pred - target) ** 2).item()

    dist = torch.sqrt(torch.sum((pred - target) ** 2, dim=1))  # (batch,)
    ade = torch.mean(dist).item()
    fde = ade

    acc = {}
    for tol in TOLERANCES:
        acc[tol] = torch.mean((dist <= tol).float()).item()
    return mse, ade, fde, acc

# =========================
# 5. Entraînement + Validation
# =========================
train_mse_hist, val_mse_hist = [], []
train_ade_hist, val_ade_hist = [], []
val_acc_hist = {tol: [] for tol in TOLERANCES}

best_val_mse = float("inf")
best_epoch = -1

# noms de fichiers (propres)
lr_str = f"{LEARNING_RATE:.0e}"  # ex: 1e-02
exp_name = f"hd{HIDDEN_DIM}_ly{NUM_LAYERS}_lr{lr_str}_ep{EPOCHS}"
os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

best_path  = f"../models/lstm_optimized_best_{exp_name}.pth"
final_path = f"../models/lstm_optimized_final_{exp_name}.pth"
fig_path   = f"../results/curves_{exp_name}.png"
npz_path   = f"../results/metrics_{exp_name}.npz"

print("\nDébut entraînement LSTM optimisé...\n")

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    running_mse, running_ade = 0.0, 0.0

    for batch_X, batch_Y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_Y = batch_Y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        loss.backward()
        optimizer.step()

        mse_b, ade_b, _, _ = compute_metrics(pred.detach(), batch_Y)
        running_mse += mse_b
        running_ade += ade_b

    train_mse = running_mse / len(train_loader)
    train_ade = running_ade / len(train_loader)

    # ---- Val ----
    model.eval()
    running_mse, running_ade = 0.0, 0.0
    running_acc = {tol: 0.0 for tol in TOLERANCES}

    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(DEVICE)
            batch_Y = batch_Y.to(DEVICE)
            pred = model(batch_X)

            mse_b, ade_b, _, acc_b = compute_metrics(pred, batch_Y)
            running_mse += mse_b
            running_ade += ade_b
            for tol in TOLERANCES:
                running_acc[tol] += acc_b[tol]

    val_mse = running_mse / len(val_loader)
    val_ade = running_ade / len(val_loader)
    val_acc = {tol: running_acc[tol] / len(val_loader) for tol in TOLERANCES}

    train_mse_hist.append(train_mse)
    val_mse_hist.append(val_mse)
    train_ade_hist.append(train_ade)
    val_ade_hist.append(val_ade)
    for tol in TOLERANCES:
        val_acc_hist[tol].append(val_acc[tol])

    # sauvegarde best sur Val MSE
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_epoch = epoch
        torch.save(model.state_dict(), best_path)

    # affichage
    acc_txt = " | ".join([f"Acc@{int(tol)}={100*val_acc[tol]:.1f}%" for tol in TOLERANCES])
    print(f"Epoch {epoch}/{EPOCHS} - "
          f"Train MSE={train_mse:.2f} ADE={train_ade:.2f} | "
          f"Val MSE={val_mse:.2f} ADE={val_ade:.2f} | {acc_txt}")

print("\nEntraînement terminé ✔️")
print(f"✅ Best Val MSE = {best_val_mse:.4f} (epoch {best_epoch})")
print(f"💾 Best modèle sauvegardé : {best_path}")

# Modèle final
torch.save(model.state_dict(), final_path)
print(f"💾 Modèle final sauvegardé : {final_path}")

# =========================
# 6. Figures + sauvegarde métriques
# =========================
plt.figure(figsize=(7, 4))
plt.plot(train_mse_hist, label="Train MSE")
plt.plot(val_mse_hist, label="Val MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title(f"Courbes MSE - {exp_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"🖼️ Figure sauvegardée : {fig_path}")

# Sauvegarde tout dans un npz
save_dict = {
    "train_mse": np.array(train_mse_hist),
    "val_mse": np.array(val_mse_hist),
    "train_ade": np.array(train_ade_hist),
    "val_ade": np.array(val_ade_hist),
}
for tol in TOLERANCES:
    save_dict[f"val_acc@{tol}"] = np.array(val_acc_hist[tol])

np.savez(npz_path, **save_dict)
print(f"📦 Métriques sauvegardées : {npz_path}")
