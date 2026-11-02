"""Étape 4 : mesure finale sur le TEST (une seule fois), coût en production, tableau comparatif.

    python m4_final.py        -> répétition à blanc sur la VALIDATION (n'écrit aucun fichier)
    python m4_final.py test   -> mesure finale sur le TEST (refuse de tourner si elle a déjà eu lieu)
"""
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import SEED, load_images, normalize, set_seed, split_indices
from m1_dense import BATCH, MLP, evaluate
from m3_resnet import make_resnet

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
assert SPLIT in ("val", "test")
FINAL = Path("results/m4_final.json")
if SPLIT == "test" and FINAL.exists():
    sys.exit("Le test a déjà été mesuré (results/m4_final.json existe) : on ne le mesure qu'une fois.")

# --- Données : même split pour tout le monde. 32x32 pour logreg et MLP, 128x128 pour ResNet.
X32, y, classes = load_images(32)
train_idx, val_idx, test_idx = split_indices(y)
X32, _, _ = normalize(X32, train_idx)
X128, _, _ = normalize(load_images(128)[0], train_idx)
eval_idx = test_idx if SPLIT == "test" else val_idx
print(f"Mesure sur : {SPLIT.upper()} ({len(eval_idx)} images)\n")
loss_fn = nn.CrossEntropyLoss()


def latency_ms(fn, n=100):
    """Temps moyen d'une prédiction sur UNE image (après 5 prédictions d'échauffement)."""
    for _ in range(5):
        fn()
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n * 1000


def torch_latencies(model, x):
    """Temps par image avec 1 thread (borne modeste) et 4 threads (mon ordinateur)."""
    model.eval()
    out = {}
    with torch.no_grad():
        for threads in (1, 4):
            torch.set_num_threads(threads)
            out[threads] = latency_ms(lambda: model(x))
    torch.set_num_threads(4)
    return out


results = {}

# =====================================================================================
# 1. Régression logistique (C choisi sur la validation à l'étape 1)
# =====================================================================================
best_C = json.load(open("results/m1_logreg.json"))["best_C"]
Xf = X32.flatten(start_dim=1).numpy()
set_seed()
clf = LogisticRegression(C=best_C, max_iter=2000, random_state=0)
t = time.time()
clf.fit(Xf[train_idx], y.numpy()[train_idx])
train_s = time.time() - t
pred = clf.predict(Xf[eval_idx])
results["Régression logistique"] = {
    "pred": pred, "val_acc": clf.score(Xf[val_idx], y.numpy()[val_idx]), "train_s": train_s,
    "params": clf.coef_.size + clf.intercept_.size,
    "lat1": latency_ms(lambda: clf.predict(Xf[:1])), "lat4": None}

# =====================================================================================
# 2. MLP : meilleure configuration de l'étape 2 (plus petite perte de validation), arrêt à la meilleure époque
# =====================================================================================
m2 = json.load(open("results/m2_summary.json"))
cfg = m2["runs_summary"][m2["best_config"]]["config"]
assert set(cfg) <= {"hidden"}, "ce script ne gère que le changement de taille cachée"
set_seed()
mlp = MLP(3 * 32 * 32, cfg.get("hidden", 128), len(classes))
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.0)
loader = DataLoader(TensorDataset(X32[train_idx], y[train_idx]), batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), num_workers=0)
val_loader = DataLoader(TensorDataset(X32[val_idx], y[val_idx]), batch_size=256, num_workers=0)
best = (float("inf"), None, 0)
t = time.time()
for epoch in range(1, 31):
    mlp.train()
    for xb, yb in loader:
        loss = loss_fn(mlp(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    vl, va = evaluate(mlp, val_loader, loss_fn)
    if vl < best[0]:
        best = (vl, {k: v.clone() for k, v in mlp.state_dict().items()}, epoch)
train_s = time.time() - t
mlp.load_state_dict(best[1])
mlp.eval()
with torch.no_grad():
    pred = mlp(X32[eval_idx]).argmax(dim=1).numpy()
    val_acc = (mlp(X32[val_idx]).argmax(dim=1) == y[val_idx]).float().mean().item()
print(f"MLP ({m2['best_config']}) : meilleure époque {best[2]}, val acc {val_acc:.1%} "
      f"(étape 2 : époque {m2['runs_summary'][m2['best_config']]['best_epoch']}, "
      f"{m2['runs_summary'][m2['best_config']]['val_acc_at_best']:.1%})")
lat = torch_latencies(mlp, X32[:1])
results["MLP (16 neurones)"] = {"pred": pred, "val_acc": val_acc, "train_s": train_s,
                                "params": sum(p.numel() for p in mlp.parameters()), "lat1": lat[1], "lat4": lat[4]}

# =====================================================================================
# 3. Meilleur ResNet-18 (variante retenue à l'étape 3 sur la validation)
# =====================================================================================
m3c = json.load(open("results/m3c_layer4.json"))
assert m3c["chosen_resnet"] == "C", "ce script suppose que la variante C a été retenue"
resnet = make_resnet(len(classes), freeze_backbone=False)
resnet.load_state_dict(torch.load("models/resnet18_best.pt"))
resnet.eval()
with torch.no_grad():
    pred = resnet(X128[eval_idx]).argmax(dim=1).numpy()
    val_acc = (resnet(X128[val_idx]).argmax(dim=1) == y[val_idx]).float().mean().item()
lat = torch_latencies(resnet, X128[:1])
results["ResNet-18 (layer4)"] = {"pred": pred, "val_acc": val_acc, "train_s": m3c["C_layer4"]["train_seconds"],
                                 "params": sum(p.numel() for p in resnet.parameters()), "lat1": lat[1], "lat4": lat[4]}

# =====================================================================================
# Tableau final
# =====================================================================================
y_eval = y.numpy()[eval_idx]
rows = []
for name, r in results.items():
    acc = float((r["pred"] == y_eval).mean())
    ci = 1.96 * math.sqrt(acc * (1 - acc) / len(y_eval))  # intervalle de confiance à 95 % (loi binomiale)
    rows.append({"model": name, f"{SPLIT}_acc": acc, "ci95": ci, "val_acc": r["val_acc"], "params": int(r["params"]),
                 "size_mb": r["params"] * 4 / 1e6, "ms_per_image_1thread": r["lat1"], "ms_per_image_4threads": r["lat4"],
                 "train_seconds": r["train_s"]})

print(f"\n{'Modèle':24s} {SPLIT + ' acc':>10s} {'±95%':>6s} {'val acc':>8s} {'params':>10s} {'Mo':>7s} {'ms (1 thr)':>11s} {'ms (4 thr)':>11s} {'train (s)':>10s}")
for r in rows:
    l4 = f"{r['ms_per_image_4threads']:.2f}" if r["ms_per_image_4threads"] is not None else "-"
    print(f"{r['model']:24s} {r[SPLIT + '_acc']:10.1%} {r['ci95']:6.1%} {r['val_acc']:8.1%} {r['params']:10d} {r['size_mb']:7.2f} "
          f"{r['ms_per_image_1thread']:11.2f} {l4:>11s} {r['train_seconds']:10.1f}")
print("Hasard : 20.0 % (classe majoritaire du train)")

cm = confusion_matrix(y_eval, results["ResNet-18 (layer4)"]["pred"])
print(f"\nRésultat ResNet-18, exactitude par style ({SPLIT}) :",
      {c: f"{cm[i, i] / cm[i].sum():.0%}" for i, c in enumerate(classes)})
print(cm)

if SPLIT == "test":
    Path("results").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("ResNet-18 (layer4) : matrice de confusion (test)")
    fig.tight_layout()
    fig.savefig("results/m4_confusion_test_resnet.png", dpi=150)
    pd_rows = ["model,test_acc,ci95,val_acc,params,size_mb,ms_per_image_1thread,ms_per_image_4threads,train_seconds"]
    for r in rows:
        l4 = "" if r["ms_per_image_4threads"] is None else f"{r['ms_per_image_4threads']:.2f}"
        pd_rows.append(f"{r['model']},{r['test_acc']:.4f},{r['ci95']:.4f},{r['val_acc']:.4f},{r['params']},"
                       f"{r['size_mb']:.2f},{r['ms_per_image_1thread']:.2f},{l4},{r['train_seconds']:.1f}")
    Path("results/m4_final_table.csv").write_text("\n".join(pd_rows) + "\n", encoding="utf-8")
    FINAL.write_text(json.dumps({"rows": rows, "confusion_resnet_test": cm.tolist(), "classes": classes}, indent=2), encoding="utf-8")
    print("\nFichiers écrits : results/m4_final.json, results/m4_final_table.csv, results/m4_confusion_test_resnet.png")
