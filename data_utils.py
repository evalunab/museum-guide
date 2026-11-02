"""Chargement des données WikiArt, split train/val/test et normalisation (partagé par toutes les étapes)."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = Path("data/wikiart_style5")


def set_seed(seed=SEED):
    """Fixe les 3 générateurs aléatoires pour que les résultats soient reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_images(size):
    """Lit toutes les images, les met à `size` x `size` pixels.
    Renvoie X (N, 3, size, size) avec des valeurs entre 0 et 1, y (N,) les numéros de classe, et les noms de classes."""
    classes = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    images, labels = [], []
    for label, name in enumerate(classes):
        for path in sorted((DATA_DIR / name).glob("*.jpg")):
            img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            labels.append(label)
    X = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()  # (N, H, W, 3) -> (N, 3, H, W)
    return X, torch.tensor(labels), classes


def split_indices(y, seed=SEED):
    """Split 70 % train / 15 % validation / 15 % test, avec la même proportion de chaque style dans chaque partie."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.30, stratify=y.numpy(), random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=y.numpy()[rest], random_state=seed)
    return train, val, test


def normalize(X, train_idx):
    """Centre et réduit chaque canal (R, V, B). Moyenne et écart-type calculés sur le TRAIN seulement."""
    mean = X[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    std = X[train_idx].std(dim=(0, 2, 3), keepdim=True)
    return (X - mean) / std, mean.flatten().tolist(), std.flatten().tolist()
