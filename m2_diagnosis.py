"""Étape 2 : suivi de l'entraînement (courbes), vérifications de base et diagnostic du surapprentissage."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import SEED, load_images, normalize, set_seed, split_indices
from m1_dense import BATCH, IMG_SIZE, MLP, evaluate

EPOCHS = 30
Path("results").mkdir(exist_ok=True)

# --- Mêmes données et même split qu'à l'étape 1 (le test reste réservé)
X, y, classes = load_images(IMG_SIZE)
train_idx, val_idx, test_idx = split_indices(y)
X, _, _ = normalize(X, train_idx)
Xtr, ytr, Xva, yva = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256, num_workers=0)
train_eval_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, num_workers=0)
loss_fn = nn.CrossEntropyLoss()


def train(hidden=128, dropout=0.0, weight_decay=0.0, flip=False, lr=1e-3, epochs=EPOCHS):
    """Entraîne un MLP avec la boucle écrite à la main et renvoie les courbes (une valeur par époque)."""
    set_seed()  # même départ pour toutes les configurations : la comparaison est équitable
    model = MLP(3 * IMG_SIZE * IMG_SIZE, hidden, len(classes), dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            if flip:  # augmentation : on retourne (miroir gauche-droite) la moitié des images au hasard
                mask = torch.rand(len(xb)) < 0.5
                xb = torch.where(mask[:, None, None, None], xb.flip(3), xb)
            out = model(xb)
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tr_loss, tr_acc = evaluate(model, train_eval_loader, loss_fn)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc)):
            hist[k].append(v)
    return model, hist


# =====================================================================================
# 1. Vérifications de base : est-ce que le code lui-même est correct ?
# =====================================================================================
print("=== 1. Vérifications de base ===")
set_seed()
model = MLP(3 * IMG_SIZE * IMG_SIZE, 128, len(classes))
init_loss, init_acc = evaluate(model, val_loader, loss_fn)
print(f"Perte avant entraînement : {init_loss:.3f} (attendu pour 5 classes au hasard : ln(5) = {math.log(5):.3f})")

# Test classique : le réseau doit pouvoir apprendre PAR CŒUR un seul petit lot. Sinon, il y a un bug.
set_seed()
model = MLP(3 * IMG_SIZE * IMG_SIZE, 128, len(classes))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
xb, yb = Xtr[:32], ytr[:32]
for step in range(200):
    loss = loss_fn(model(xb), yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Perte sur un seul lot de 32 images après 200 pas : {loss.item():.4f} (doit tendre vers 0)")

# =====================================================================================
# 2. Courbes de la baseline (celle de l'étape 1) et diagnostic
# =====================================================================================
print("\n=== 2. Diagnostic de la baseline ===")
_, base = train()
epochs = range(1, EPOCHS + 1)
best_epoch = min(range(EPOCHS), key=lambda i: base["val_loss"][i]) + 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(epochs, base["train_loss"], label="train")
a1.plot(epochs, base["val_loss"], label="validation")
a1.axvline(best_epoch, color="gray", linestyle="--", label=f"min val loss (époque {best_epoch})")
a1.set(title="Perte (erreur) par époque", xlabel="époque", ylabel="loss")
a1.legend()
a2.plot(epochs, base["train_acc"], label="train")
a2.plot(epochs, base["val_acc"], label="validation")
a2.axhline(0.2, color="red", linestyle=":", label="hasard (20 %)")
a2.set(title="Exactitude par époque", xlabel="époque", ylabel="accuracy")
a2.legend()
fig.suptitle("MLP de base : le surapprentissage")
fig.tight_layout()
fig.savefig("results/m2_curves_baseline.png", dpi=150)
gap = base["train_acc"][-1] - base["val_acc"][-1]
print(f"Fin d'entraînement : train acc {base['train_acc'][-1]:.1%} | val acc {base['val_acc'][-1]:.1%} | écart {gap:.1%}")
print(f"Perte de validation minimale à l'époque {best_epoch} ({base['val_loss'][best_epoch - 1]:.3f}), "
      f"puis elle remonte jusqu'à {base['val_loss'][-1]:.3f}")

# =====================================================================================
# 3. Remèdes : on les compare sur la VALIDATION
# =====================================================================================
print("\n=== 3. Remèdes (choix sur la validation) ===")
configs = {
    "base": dict(),
    "dropout 0.5": dict(dropout=0.5),
    "weight decay 0.1": dict(weight_decay=0.1),
    "petit (16 neurones)": dict(hidden=16),
    "retournement d'images": dict(flip=True),
    "dropout + wd + retournement": dict(dropout=0.5, weight_decay=0.1, flip=True),
}
runs, summary = {}, {}
for name, cfg in configs.items():
    h = base if name == "base" else train(**cfg)[1]  # la baseline est déjà entraînée plus haut
    runs[name] = h
    b = min(range(EPOCHS), key=lambda i: h["val_loss"][i])  # "arrêt anticipé" : on garde l'époque de plus petite perte val
    summary[name] = {"config": cfg, "best_epoch": b + 1, "min_val_loss": h["val_loss"][b], "val_acc_at_best": h["val_acc"][b],
                     "final_train_acc": h["train_acc"][-1], "final_val_acc": h["val_acc"][-1]}
    print(f"{name:30s} | fin: train {h['train_acc'][-1]:.1%} val {h['val_acc'][-1]:.1%} | "
          f"meilleure époque {b + 1:2d}: val loss {h['val_loss'][b]:.3f} val acc {h['val_acc'][b]:.1%}")

best_name = min(summary, key=lambda n: summary[n]["min_val_loss"])
print(f"\nMeilleure configuration (plus petite perte de validation) : {best_name}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
for name, h in runs.items():
    a1.plot(epochs, h["val_loss"], label=name)
    a2.plot(epochs, h["val_acc"], label=name)
a1.set(title="Perte de validation", xlabel="époque", ylabel="loss")
a2.set(title="Exactitude de validation", xlabel="époque", ylabel="accuracy")
a2.axhline(0.2, color="red", linestyle=":")
a1.legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/m2_curves_compare.png", dpi=150)

json.dump({"initial_loss": init_loss, "ln5": math.log(5), "overfit_one_batch_loss": loss.item(),
           "baseline_best_epoch": best_epoch, "runs_summary": summary, "best_config": best_name},
          open("results/m2_summary.json", "w"), indent=2)
