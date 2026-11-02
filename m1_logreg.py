"""Étape 1 : baseline classique (régression logistique scikit-learn) sur les mêmes pixels 32x32 normalisés."""
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from data_utils import load_images, normalize, set_seed, split_indices

set_seed()
IMG_SIZE = 32  # identique au réseau MLP

X, y, classes = load_images(IMG_SIZE)
train_idx, val_idx, test_idx = split_indices(y)  # même split que le MLP (test non utilisé ici)
X, _, _ = normalize(X, train_idx)
X = X.flatten(start_dim=1).numpy()  # une image = une liste de 3072 nombres
y = y.numpy()

# On choisit la force de régularisation C sur la validation (petit C = modèle plus simple)
results = {}
for C in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    clf = LogisticRegression(C=C, max_iter=2000, random_state=0)
    clf.fit(X[train_idx], y[train_idx])
    results[C] = {"train_acc": clf.score(X[train_idx], y[train_idx]), "val_acc": clf.score(X[val_idx], y[val_idx])}
    print(f"C={C:<7} | train acc {results[C]['train_acc']:.1%} | val acc {results[C]['val_acc']:.1%}")

best_C = max(results, key=lambda c: results[c]["val_acc"])
print(f"Meilleur C (choisi sur la validation) : {best_C}")

# Comparaison avec le MLP (historique enregistré par m1_dense.py)
mlp = json.load(open("results/m1_history.json"))["history"]
print("\n--- Exactitude sur la VALIDATION ---")
print(f"Hasard (classe majoritaire)   : {json.load(open('results/m1_history.json'))['majority_baseline']:.1%}")
print(f"Régression logistique         : {results[best_C]['val_acc']:.1%}")
print(f"MLP (dernière époque)         : {mlp['val_acc'][-1]:.1%}")
print(f"MLP (meilleure époque)        : {max(mlp['val_acc']):.1%}")

Path("results").mkdir(exist_ok=True)
json.dump({"best_C": best_C, "all": {str(c): r for c, r in results.items()}},
          open("results/m1_logreg.json", "w"), indent=2)
