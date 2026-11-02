"""Étape 3 (suite) : variante C = on dégèle seulement layer4 + la dernière couche, petit learning rate.
On compare C à A (backbone gelé) et B (tout dégelé) sur la VALIDATION et on garde la meilleure.
On mesure aussi A "strict" : un vrai classifieur linéaire sur les features figées du ResNet (BatchNorm inclus)."""
import copy
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import SEED, load_images, normalize, set_seed, split_indices
from m3_resnet import BATCH, IMG_SIZE, evaluate, make_resnet

torch.set_num_threads(4)
EPOCHS_C = 6
LR_C = 1e-4

X, y, classes = load_images(IMG_SIZE)
train_idx, val_idx, test_idx = split_indices(y)  # le test reste réservé
X, _, _ = normalize(X, train_idx)
Xtr, ytr, Xva, yva = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=128, num_workers=0)
loss_fn = nn.CrossEntropyLoss()
prev = json.load(open("results/m3_resnet.json"))  # résultats de A et B (m3_resnet.py)

# ---------------------------------------------------------------------------------
# C : seuls layer4 et la dernière couche sont entraînés
# ---------------------------------------------------------------------------------
set_seed()
model = make_resnet(len(classes), freeze_backbone=True)  # tout gelé, nouvelle couche fc entraînable
for p in model.layer4.parameters():
    p.requires_grad = True                                # on dégèle le dernier bloc
params = [p for p in model.parameters() if p.requires_grad]
print(f"C : {sum(p.numel() for p in params)} paramètres entraînés (sur {sum(p.numel() for p in model.parameters())})")
optimizer = torch.optim.Adam(params, lr=LR_C)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), num_workers=0)
hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_loss, best_state, best_epoch = float("inf"), None, 0
t0 = time.time()
for epoch in range(1, EPOCHS_C + 1):
    model.eval()          # les couches gelées gardent leurs statistiques BatchNorm
    model.layer4.train()  # seul le bloc dégelé est en mode entraînement
    run_loss, run_correct = 0.0, 0
    for xb, yb in loader:
        mask = torch.rand(len(xb)) < 0.5  # retournement d'images, comme pour B
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
        print(f"1re époque : {time.time() - t0:.1f} s")
    print(f"[C layer4] époque {epoch} | train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:.1%}"
          f" | val loss {va_loss:.3f} acc {va_acc:.1%}")
train_seconds_c = time.time() - t0
model.load_state_dict(best_state)
acc_c = hist["val_acc"][best_epoch - 1]
print(f"C terminé en {train_seconds_c:.0f} s | meilleure val loss {best_loss:.3f} à l'époque {best_epoch} (val acc {acc_c:.1%})")

# ---------------------------------------------------------------------------------
# A strict : features figées (calculées une fois, mode eval) + une seule couche nn.Linear
# ---------------------------------------------------------------------------------
set_seed()
backbone = make_resnet(len(classes), freeze_backbone=True)
backbone.fc = nn.Identity()  # on récupère les 512 nombres juste avant la dernière couche
backbone.eval()
with torch.no_grad():
    Ftr, Fva = backbone(Xtr), backbone(Xva)
head = nn.Linear(512, len(classes))
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
fl = DataLoader(TensorDataset(Ftr, ytr), batch_size=BATCH, shuffle=True,
                generator=torch.Generator().manual_seed(SEED), num_workers=0)
best_strict = (float("inf"), 0.0, 0)
for epoch in range(1, 31):
    head.train()
    for fb, yb in fl:
        loss = loss_fn(head(fb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        out = head(Fva)
        vl, va = loss_fn(out, yva).item(), (out.argmax(dim=1) == yva).float().mean().item()
    if vl < best_strict[0]:
        best_strict = (vl, va, epoch)
print(f"\nA strict (classifieur linéaire sur features figées) : val acc {best_strict[1]:.1%} "
      f"(loss {best_strict[0]:.3f}, époque {best_strict[2]})")

# ---------------------------------------------------------------------------------
# Comparaison sur la validation et choix
# ---------------------------------------------------------------------------------
cands = {"A": (prev["A_frozen"]["val_loss"], prev["A_frozen"]["val_acc"]),
         "B": (prev["B_finetune"]["val_loss"], prev["B_finetune"]["val_acc"]),
         "C": (best_loss, acc_c)}
print("\n--- Validation (meilleure époque de chaque variante) ---")
for k, (vl, va) in cands.items():
    print(f"{k} : val loss {vl:.3f} | val acc {va:.1%}")
chosen = min(cands, key=lambda k: cands[k][0])
print(f"Variante retenue (plus petite perte de validation) : {chosen}")
if chosen == "C":
    torch.save(model.state_dict(), "models/resnet18_best.pt")  # remplace B (ignoré par git)
else:
    print("Le fichier models/resnet18_best.pt n'est pas modifié.")

# Les durées de A et B ont été affichées par m3_resnet.py (401 s et 307 s) ; celle de C est mesurée ici.
json.dump({"C_layer4": {"best_epoch": best_epoch, "val_acc": acc_c, "val_loss": best_loss, "train_seconds": train_seconds_c,
                        "trained_params": sum(p.numel() for p in params), "history": hist},
           "A_strict_linear_probe": {"val_acc": best_strict[1], "val_loss": best_strict[0], "best_epoch": best_strict[2]},
           "chosen_resnet": chosen},
          open("results/m3c_layer4.json", "w"), indent=2)
