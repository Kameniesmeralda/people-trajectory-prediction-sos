import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#----------1.Charger les données--------
X = np.load("../data/X.npy")   # (N, SEQ_LEN, 2)
Y = np.load("../data/Y.npy")   # (N, 2)

X_torch = torch.tensor(X, dtype=torch.float32)


# ---------- 2. Définir le même modèle LSTM que pour l'entraînement ----------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm= nn.LSTM(input_dim, hidden_dim,num_layers, batch_first=True)
        self.fc=nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        pred = self.fc(last_out)
        return pred

model=LSTMModel()
model_path="../models/lstm_model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

print("Modèle chargé depuis:", model_path)

# ---------- 3. Choisir un exemple et prédire ----------
# indice aléatoire ou fixe
idx = np.random.randint(0, X.shape[0])
# idx = 0  # si tu veux toujours le même

input_seq = X_torch[idx]          # (SEQ_LEN, 2)
target = Y[idx]                   # (2,)

with torch.no_grad():
    pred = model(input_seq.unsqueeze(0))   # (1, 2)
pred = pred.squeeze(0).numpy()             # (2,)

print("Exemple n°", idx)
print("Vraie position suivante :", target)
print("Position prédite        :", pred)

#-----------------4. Visualisation--------------
seq=input_seq.numpy()

plt.figure(figsize=(6,6))
#trajectoire d'entrée
plt.plot(seq[:, 0], seq[:, 1], "-o", label="Séquence d'entrée (10 positions)")
# vraie position suivante
plt.scatter(target[0], target[1], c="green", marker="x", s=80, label="Vraie position suivante")
# position prédite
plt.scatter(pred[0], pred[1], c="red", marker="*", s=120, label="Position prédite")

plt.title("Prédiction LSTM d'une position future")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()