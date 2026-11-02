"""Étape 3 : ResNet-18 pré-entraîné sur ImageNet, adapté à nos 5 styles.
A) backbone gelé : on n'entraîne que la dernière couche (nn.Linear)
B) fine-tuning : on entraîne tout le réseau avec un petit pas d'apprentissage."""
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from data_utils import SEED, load_images, normalize, set_seed, split_indices

torch.set_num_threads(4)
set_seed()

IMG_SIZE = 128   # taille des images stockées : on ne les réduit pas, le CNN profite des détails
BATCH = 64
EPOCHS_A = 30    # la tête seule est très rapide à entraîner
EPOCHS_B = 8
LR_A = 1e-3
LR_B = 1e-4      # petit pas : on ne veut pas détruire ce que ResNet a déjà appris


def make_resnet(n_classes, freeze_backbone):
    """ResNet-18 avec poids ImageNet, dont la dernière couche est remplacée par une couche à 5 sorties."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False   # ces poids ne seront pas modifiés
    model.fc = nn.Linear(model.fc.in_features, n_classes)  # nouvelle couche : toujours entraînable
    return model


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train(model, params, lr, epochs, flip, label):
    """Boucle d'entraînement écrite à la main. Garde les poids de l'époque à la plus petite perte de validation."""
    optimizer = torch.optim.Adam(params, lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state, best_epoch = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_correct = 0.0, 0
        for xb, yb in loader:
            if flip:
                mask = torch.rand(len(xb)) < 0.5
                xb = torch.where(mask[:, None, None, None], xb.flip(3), xb)
            out = model(xb)
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(yb)
            run_correct += (out.argmax(dim=1) == yb).sum().item()
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        for k, v in zip(hist, (run_loss / len(ytr), va_loss, run_correct / len(ytr), va_acc)):
            hist[k].append(v)
        if va_loss < best_loss:
            best_loss, best_epoch, best_state = va_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1:
            print(f"[{label}] 1re époque : {time.time() - t0:.1f} s")
        print(f"[{label}] époque {epoch:2d} | train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:.1%}"
              f" | val loss {va_loss:.3f} acc {va_acc:.1%}")
    print(f"[{label}] terminé en {time.time() - t0:.0f} s | meilleure val loss {best_loss:.3f} à l'époque {best_epoch}"
          f" (val acc {hist['val_acc'][best_epoch - 1]:.1%})")
    model.load_state_dict(best_state)
    return hist, best_epoch


if __name__ == "__main__":
    X, y, classes = load_images(IMG_SIZE)
    train_idx, val_idx, test_idx = split_indices(y)  # le test reste réservé
    X, _, _ = normalize(X, train_idx)                # statistiques du train seulement
    Xtr, ytr, Xva, yva = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=128, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # --- A) backbone gelé
    set_seed()
    model_a = make_resnet(len(classes), freeze_backbone=True)
    n_train_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    print(f"A) backbone gelé : {n_train_a} paramètres entraînés (sur {sum(p.numel() for p in model_a.parameters())})")
    hist_a, best_a = train(model_a, model_a.fc.parameters(), LR_A, EPOCHS_A, flip=False, label="A gelé")

    # --- B) fine-tuning complet
    set_seed()
    model_b = make_resnet(len(classes), freeze_backbone=False)
    print(f"\nB) fine-tuning : {sum(p.numel() for p in model_b.parameters())} paramètres entraînés")
    hist_b, best_b = train(model_b, model_b.parameters(), LR_B, EPOCHS_B, flip=True, label="B fine-tuning")

    # --- Choix sur la validation
    acc_a, acc_b = hist_a["val_acc"][best_a - 1], hist_b["val_acc"][best_b - 1]
    loss_a, loss_b = hist_a["val_loss"][best_a - 1], hist_b["val_loss"][best_b - 1]
    chosen = "B" if loss_b < loss_a else "A"
    model = model_b if chosen == "B" else model_a
    print(f"\nValidation : A gelé {acc_a:.1%} (loss {loss_a:.3f}) | B fine-tuning {acc_b:.1%} (loss {loss_b:.3f})")
    print(f"Variante retenue (plus petite perte de validation) : {chosen}")
    torch.save(model.state_dict(), "models/resnet18_best.pt")  # ignoré par git (*.pt)

    # --- Coût en production : taille et temps de prédiction sur CPU (une image à la fois, comme sur une borne)
    n_params = sum(p.numel() for p in model.parameters())
    model.eval()
    one = Xva[:1]
    with torch.no_grad():
        for _ in range(5):
            model(one)  # échauffement
        t = time.time()
        for _ in range(50):
            model(one)
        latency_ms = (time.time() - t) / 50 * 1000
    print(f"ResNet-18 : {n_params} paramètres, {n_params * 4 / 1e6:.1f} Mo, {latency_ms:.1f} ms par image (CPU, 4 threads)")

    # --- Courbes
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("loss", "acc"), ("Perte", "Exactitude")):
        for hist, name, ls in ((hist_a, "A gelé", "--"), (hist_b, "B fine-tuning", "-")):
            ep = range(1, len(hist[f"val_{key}"]) + 1)
            ax.plot(ep, hist[f"train_{key}"], ls, color="tab:blue", label=f"{name} train")
            ax.plot(ep, hist[f"val_{key}"], ls, color="tab:orange", label=f"{name} validation")
        ax.set(title=f"ResNet-18 : {title.lower()} par époque", xlabel="époque", ylabel=key)
        ax.legend(fontsize=7)
    axes[1].axhline(0.2, color="red", linestyle=":")
    fig.tight_layout()
    fig.savefig("results/m3_curves.png", dpi=150)

    json.dump({"A_frozen": {"best_epoch": best_a, "val_acc": acc_a, "val_loss": loss_a, "trained_params": n_train_a, "history": hist_a},
               "B_finetune": {"best_epoch": best_b, "val_acc": acc_b, "val_loss": loss_b, "history": hist_b},
               "chosen": chosen, "n_params": n_params, "latency_ms_per_image": latency_ms},
              open("results/m3_resnet.json", "w"), indent=2)
