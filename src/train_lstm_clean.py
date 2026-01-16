# train_lstm_clean.py
import os
import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fmt_lr(lr: float) -> str:
    # ex: 0.01 -> "1e-02" ; 0.0001 -> "1e-04"
    return f"{lr:.0e}"


# =========================
# Dataset
# =========================
class TrajectoryDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================
# Model
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, output_dim=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        last_out = out[:, -1, :]       # (B, H)
        pred = self.fc(last_out)       # (B, 2)
        return pred


# =========================
# Metrics
# =========================
@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, acc_thresholds=(2.0, 5.0, 10.0)):
    """
    pred, target: (B, 2)
    - MSE : mean squared error (sur x,y)
    - ADE : average displacement error = mean(||pred-target||2)
    - Acc@k : % d'exemples dont la distance <= k
    """
    # MSE sur les deux coordonnées
    mse = torch.mean((pred - target) ** 2).item()

    # distance euclidienne
    d = torch.norm(pred - target, dim=1)  # (B,)
    ade = torch.mean(d).item()

    accs = {}
    for thr in acc_thresholds:
        accs[thr] = (torch.mean((d <= thr).float()).item())  # fraction in [0,1]

    return mse, ade, accs


def epoch_pass(model, loader, device, criterion, optimizer=None, acc_thresholds=(2.0, 5.0, 10.0)):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_mse_loss = 0.0
    total_mse_metric = 0.0
    total_ade = 0.0
    total_acc = {thr: 0.0 for thr in acc_thresholds}
    n_batches = 0

    for batch_X, batch_Y in loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(batch_X)
        loss = criterion(pred, batch_Y)

        if is_train:
            loss.backward()
            # petit + : évite les explosions de gradient si ça arrive
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        mse_metric, ade, accs = compute_metrics(pred, batch_Y, acc_thresholds=acc_thresholds)

        total_mse_loss += loss.item()
        total_mse_metric += mse_metric
        total_ade += ade
        for thr in acc_thresholds:
            total_acc[thr] += accs[thr]

        n_batches += 1

    # moyennes par batch
    avg_loss = total_mse_loss / max(1, n_batches)
    avg_mse = total_mse_metric / max(1, n_batches)
    avg_ade = total_ade / max(1, n_batches)
    avg_acc = {thr: total_acc[thr] / max(1, n_batches) for thr in acc_thresholds}

    return avg_loss, avg_mse, avg_ade, avg_acc


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path", type=str, default="../data/X.npy")
    parser.add_argument("--y_path", type=str, default="../data/Y.npy")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--hidden_dim", type=int, default=61)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # scheduler
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "step"])
    parser.add_argument("--plateau_patience", type=int, default=8)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--step_size", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.5)

    parser.add_argument("--acc_thresholds", type=float, nargs="+", default=[2.0, 5.0, 10.0])

    parser.add_argument("--models_dir", type=str, default="../models")
    parser.add_argument("--runs_dir", type=str, default="../runs")
    parser.add_argument("--run_name", type=str, default="lstm_clean")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    # -------------------------
    # Load data
    # -------------------------
    X = np.load(args.x_path)  # (N, seq_len, 2)
    Y = np.load(args.y_path)  # (N, 2)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    N = X.shape[0]
    print("Données :", X.shape, Y.shape)

    # split 80/20
    indices = np.arange(N)
    np.random.shuffle(indices)
    split = int(0.8 * N)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    print(f"Train : {X_train.shape[0]}  |  Val : {X_val.shape[0]}")

    train_loader = DataLoader(TrajectoryDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TrajectoryDataset(X_val, Y_val), batch_size=args.batch_size, shuffle=False)

    # -------------------------
    # Model / optim
    # -------------------------
    model = LSTMModel(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=2,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    else:
        scheduler = None

    # -------------------------
    # Logging / saving
    # -------------------------
    ensure_dir(args.models_dir)
    ensure_dir(args.runs_dir)

    lr_tag = fmt_lr(args.lr)
    run_id = f"{args.run_name}_hd{args.hidden_dim}_ly{args.num_layers}_lr{lr_tag}_ep{args.epochs}"
    run_dir = os.path.join(args.runs_dir, run_id)
    ensure_dir(run_dir)

    csv_path = os.path.join(run_dir, "metrics.csv")
    fig_path = os.path.join(run_dir, "curves.png")

    best_model_path = os.path.join(args.models_dir, f"{run_id}_BEST.pth")
    final_model_path = os.path.join(args.models_dir, f"{run_id}_FINAL.pth")

    # CSV header
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["epoch", "lr", "train_mse", "train_ade", "val_mse", "val_ade"]
        for thr in args.acc_thresholds:
            cols.append(f"acc@{int(thr)}")
        f.write(",".join(cols) + "\n")

    train_mse_hist, val_mse_hist = [], []
    train_ade_hist, val_ade_hist = [], []

    best_val_mse = math.inf
    best_epoch = -1

    print("\nDébut entraînement LSTM (clean)...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mse, train_ade, _ = epoch_pass(
            model, train_loader, device, criterion, optimizer=optimizer, acc_thresholds=args.acc_thresholds
        )
        val_loss, val_mse, val_ade, val_acc = epoch_pass(
            model, val_loader, device, criterion, optimizer=None, acc_thresholds=args.acc_thresholds
        )

        # step scheduler
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_mse)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        train_mse_hist.append(train_mse)
        val_mse_hist.append(val_mse)
        train_ade_hist.append(train_ade)
        val_ade_hist.append(val_ade)

        acc_str = " | ".join([f"Acc@{int(thr)}={val_acc[thr]*100:.1f}%" for thr in args.acc_thresholds])

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- Train MSE={train_mse:.2f} ADE={train_ade:.2f} | "
            f"Val MSE={val_mse:.2f} ADE={val_ade:.2f} | "
            f"{acc_str} | lr={current_lr:.2e}"
        )

        # CSV row
        with open(csv_path, "a", encoding="utf-8") as f:
            row = [str(epoch), f"{current_lr:.8f}", f"{train_mse:.6f}", f"{train_ade:.6f}", f"{val_mse:.6f}", f"{val_ade:.6f}"]
            for thr in args.acc_thresholds:
                row.append(f"{val_acc[thr]*100:.6f}")  # en %
            f.write(",".join(row) + "\n")

        # Save best model (based on val MSE)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    # Save final model
    torch.save(model.state_dict(), final_model_path)

    print("\nEntraînement terminé ✔️")
    print(f"✅ Best Val MSE = {best_val_mse:.4f} (epoch {best_epoch})")
    print(f"💾 Best modèle sauvegardé : {best_model_path}")
    print(f"💾 Modèle final sauvegardé : {final_model_path}")
    print(f"🧾 CSV métriques : {csv_path}")

    # -------------------------
    # Plot curves
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(train_mse_hist, label="Train MSE")
    plt.plot(val_mse_hist, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Courbes MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "mse_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(train_ade_hist, label="Train ADE")
    plt.plot(val_ade_hist, label="Val ADE")
    plt.xlabel("Epoch")
    plt.ylabel("ADE (distance)")
    plt.title("Courbes ADE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "ade_curve.png"), dpi=150)
    plt.close()

    # courbe combinée
    plt.figure(figsize=(7, 5))
    plt.plot(val_mse_hist, label="Val MSE")
    plt.plot(val_ade_hist, label="Val ADE")
    plt.xlabel("Epoch")
    plt.title("Validation : MSE & ADE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"📈 Courbes sauvegardées dans : {run_dir}")


if __name__ == "__main__":
    main()
