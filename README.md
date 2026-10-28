# museum-guide

Projet "Neural Networks & Deep Learning Foundations" (Albert School) :
un guide de musée qui reconnaît le style d'une peinture à partir d'une photo.

Question posée par le directeur : un réseau de neurones vaut-il le coup sur des bornes
à CPU modeste, ou un modèle classique plus simple suffit-il ?

## Données (non versionnées : licence WikiArt inconnue)

Les images ne sont pas dans ce dépôt. Pour les recréer :

```
pip install "datasets>=3.0" pillow
python make_wikiart_subset.py --label style --classes Impressionism Baroque Cubism Expressionism Romanticism --per-class 300 --size 128 --out data/wikiart_style5
```

5 styles x 300 images, 128x128 pixels. Révision du dataset épinglée dans le script.
