"""
------------------------------------------------------------
TRAIN_LSTM.PY
Apprentissage d’un modèle LSTM pour prédire la trajectoire
------------------------------------------------------------
Ce script :
1. Charge X.npy et Y.npy
2. Crée un dataset PyTorch
3. Définit un modèle LSTM simple
4. Entraîne le modèle
5. Affiche la perte
6. Sauvegarde le modèle entraîné
------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# =====================================================================
# 1. CHARGEMENT DU DATASET (X.npy, Y.npy)
# =====================================================================

X = np.load("../data/X.npy")   # shape (N, SEQ_LEN, 2)
Y = np.load("../data/Y.npy")   # shape (N, 2)

print("Shapes :")
print("X =", X.shape)
print("Y =", Y.shape)

# Conversion en tenseurs PyTorch
X_torch = torch.tensor(X, dtype=torch.float32)
Y_torch = torch.tensor(Y, dtype=torch.float32)


# =====================================================================
# 2. DATASET PYTORCH PERSONNALISÉ
# =====================================================================

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset = TrajectoryDataset(X_torch, Y_torch)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# =====================================================================
# 3. DÉFINITION DU MODÈLE LSTM
# =====================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Couche fully connected
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x : (batch_size, seq_len, 2)
        out, _ = self.lstm(x)
        # on garde la dernière prédiction de la séquence
        last_out = out[:, -1, :]     # shape (batch, hidden_dim)
        pred = self.fc(last_out)     # shape (batch, 2)
        return pred


model = LSTMModel()
print(model)

# Optimiseur Adam + MSE
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =====================================================================
# 4. BOUCLE D'ENTRAÎNEMENT
# =====================================================================

EPOCHS = 30
loss_history = []

print("\nDébut de l'entraînement...\n")

for epoch in range(EPOCHS):
    epoch_loss = 0

    for batch_X, batch_Y in dataloader:

        optimizer.zero_grad()       # reset gradient
        pred = model(batch_X)       # prédiction LSTM
        loss = criterion(pred, batch_Y)  # perte MSE

        loss.backward()             # backprop
        optimizer.step()            # update des poids

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss = {epoch_loss:.6f}")

print("\nEntraînement terminé ✔️")


# =====================================================================
# 5. VISUALISATION DE LA COURBE DE PERTE
# =====================================================================

plt.plot(loss_history)
plt.title("Courbe de perte (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()


# =====================================================================
# 6. SAUVEGARDE DU MODÈLE
# =====================================================================

os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/lstm_model.pth")

print("\n💾 Modèle sauvegardé dans '../models/lstm_model.pth'\n")
