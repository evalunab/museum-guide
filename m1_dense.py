"""Étape 1 : réseau dense (feed-forward) sur les pixels, avec une boucle d'entraînement écrite à la main."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on enregistre directement l'image
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import SEED, load_images, normalize, set_seed, split_indices

torch.set_num_threads(4)  # ma machine a 8 coeurs logiques : on en utilise la moitié
set_seed()

IMG_SIZE = 32      # un réseau dense voit une liste de pixels : on réduit l'image pour garder peu de poids
HIDDEN = 128       # nombre de neurones de la couche cachée
LR = 1e-3
EPOCHS = 30
BATCH = 64


class MLP(nn.Module):
    """Image aplatie -> couche cachée (ReLU) -> 5 scores (un par style)."""

    def __init__(self, n_inputs, hidden, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)       # (B, 3, 32, 32) -> (B, 3072)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def evaluate(model, loader, loss_fn):
    """Renvoie (perte moyenne, exactitude) sur un jeu de données, sans calculer de gradients."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


if __name__ == "__main__":
    # --- Données : split fixe, normalisation calculée sur le train seulement
    X, y, classes = load_images(IMG_SIZE)
    train_idx, val_idx, test_idx = split_indices(y)  # test_idx n'est PAS utilisé ici
    X, mean, std = normalize(X, train_idx)
    print(f"Images : {len(y)} | train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)} (réservé)")
    print(f"Classes : {classes}")
    print(f"Moyenne train (R,V,B) : {[round(m, 3) for m in mean]} | écart-type : {[round(s, 3) for s in std]}")

    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(TensorDataset(X[val_idx], y[val_idx]), batch_size=256, num_workers=0)

    # --- Référence : deviner toujours la classe la plus fréquente du train
    majority = torch.bincount(y[train_idx]).max().item() / len(train_idx)
    print(f"Référence 'hasard' (classe majoritaire) : {majority:.1%}")

    model = MLP(3 * IMG_SIZE * IMG_SIZE, HIDDEN, len(classes))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres du modèle : {n_params}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Boucle d'entraînement
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            out = model(xb)              # 1. forward : prédire
            loss = loss_fn(out, yb)      # 2. loss : mesurer l'erreur
            optimizer.zero_grad()        # 3. remettre les gradients à zéro
            loss.backward()              # 4. backward : calculer les gradients
            optimizer.step()             # 5. step : corriger les poids
        # fin d'époque : mesures sur le train (mode eval) et la validation
        tr_loss, tr_acc = evaluate(model, DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=256), loss_fn)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        if epoch == 1:
            print(f"Durée de la 1re époque : {time.time() - t0:.1f} s -> total estimé {(time.time() - t0) * EPOCHS:.0f} s")
        print(f"époque {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.1%} | val loss {va_loss:.3f} acc {va_acc:.1%}")

    print(f"Entraînement terminé en {time.time() - t0:.0f} s")
    print(f"Meilleure exactitude validation : {max(history['val_acc']):.1%} (époque {history['val_acc'].index(max(history['val_acc'])) + 1})")

    # --- Matrice de confusion sur la VALIDATION (modèle de la dernière époque) : lignes = vrai style, colonnes = prédit
    model.eval()
    with torch.no_grad():
        val_pred = model(X[val_idx]).argmax(dim=1)
    cm = confusion_matrix(y[val_idx].numpy(), val_pred.numpy())
    print("Matrice de confusion (validation), lignes = vrai style, colonnes = prédit :")
    print(cm)
    Path("results").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("MLP : matrice de confusion (validation)")
    fig.tight_layout()
    fig.savefig("results/m1_confusion_val.png", dpi=150)

    json.dump({"config": {"img_size": IMG_SIZE, "hidden": HIDDEN, "lr": LR, "epochs": EPOCHS, "batch": BATCH},
               "majority_baseline": majority, "n_params": n_params, "history": history},
              open("results/m1_history.json", "w"), indent=2)
