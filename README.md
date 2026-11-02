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

## Résultats de l'étape 1 : baselines (validation, 225 images)

| Modèle | Exactitude validation |
|---|---|
| Hasard (classe majoritaire) | 20.0 % |
| Régression logistique (C=0.0001) | 34.2 % |
| MLP, dernière époque | 29.3 % |
| MLP, meilleure époque | 34.2 % |

Scripts : `m1_dense.py` (MLP), `m1_logreg.py` (régression logistique). Split 70/15/15 avec seed 42,
normalisation calculée sur le train seulement. Le test (225 images) n'est pas encore utilisé.
Avec 225 images, l'incertitude sur une exactitude est d'environ ±3 points.

![Matrice de confusion du MLP (validation)](results/m1_confusion_val.png)

Styles les plus confondus : Baroque <-> Romanticism (16 + 9 erreurs), Cubism <-> Expressionism (13 + 9).

## Failure diagnosis

Script : `m2_diagnosis.py`.

**Symptôme.** Le MLP atteint 99.0 % d'exactitude sur le train mais seulement 29.3 % sur la validation
(écart de 69.6 points). La perte de validation est minimale à l'époque 3 (1.636) puis remonte jusqu'à 3.114,
alors que la perte du train tombe à 0.06.

![Courbes du MLP de base](results/m2_curves_baseline.png)

**Vérification du code.** La perte initiale est 1.625 (attendu pour 5 classes au hasard : ln 5 = 1.609).
Sur un seul lot de 32 images, la perte tombe à 0.0001 après 200 pas. Le code apprend donc correctement : ce n'est pas un bug.

**Cause.** Surapprentissage : 393 989 poids pour 1 050 images d'entraînement. Le réseau mémorise les images
au lieu d'apprendre des motifs de style. Un MLP sur pixels ne voit pas les textures ni les formes locales.

**Remèdes testés** (30 époques, mêmes données, choix sur la validation) :

| Remède | Train final | Val finale | Plus petite perte val (époque) |
|---|---|---|---|
| Aucun | 99.0 % | 29.3 % | 1.636 (3) |
| Dropout 0.5 | 92.6 % | 34.2 % | 1.600 (3) |
| Weight decay 0.1 | 99.0 % | 30.7 % | 1.636 (3) |
| Petit réseau (16 neurones) | 92.1 % | 34.7 % | 1.557 (4) |
| Retournement d'images | 91.7 % | 29.3 % | 1.647 (3) |
| Dropout + weight decay + retournement | 75.9 % | 33.8 % | 1.560 (15) |

**Preuve chiffrée / conclusion.** Les remèdes réduisent l'écart train/validation mais la validation reste
autour de 30-35 %, dans la marge d'incertitude (±3 points). Le problème vient du type de modèle, ce qui motive
un CNN (étape 3).

![Comparaison des remèdes](results/m2_curves_compare.png)

## Étape 3 : ResNet-18 pré-entraîné (validation, 225 images)

Script : `m3_resnet.py`. ResNet-18 avec poids ImageNet, images 128x128, dernière couche remplacée par un `nn.Linear` à 5 sorties.
Boucle d'entraînement écrite à la main. Le test n'est toujours pas utilisé.

| Variante | Paramètres entraînés | Val (meilleure époque) | Val finale | Train final | Durée |
|---|---|---|---|---|---|
| A : backbone gelé (30 époques) | 2 565 | 51.1 % (époque 9) | 50.7 % | 79.8 % | 401 s |
| B : fine-tuning complet (8 époques, retournement d'images) | 11 179 077 | 57.8 % (époque 2) | 59.1 % | 99.7 % | 307 s |

Dans cette première version, la variante B menait (1.066 contre 1.163) ; la variante C ci-dessous l'a ensuite dépassée.
Coût de la variante B : 11 179 077 paramètres (44.7 Mo), 24.6 ms par image sur CPU (4 threads, une image à la fois).
Pour comparaison, le MLP et la régression logistique plafonnent autour de 34 % sur la validation.

![Courbes ResNet-18](results/m3_curves.png)

### Variante C : on dégèle seulement layer4 (`m3c_layer4.py`)

layer4 + dernière couche entraînés (8 396 293 paramètres), learning rate 1e-4, 6 époques, couches gelées en mode eval, retournement d'images.

| Variante | Perte val | Exactitude val (meilleure époque) | Durée d'entraînement |
|---|---|---|---|
| A : backbone gelé (BatchNorm encore mis à jour) | 1.163 | 51.1 % | 401 s |
| B : tout dégelé | 1.066 | 57.8 % | 307 s |
| **C : layer4 dégelé** | **1.018** | **58.7 %** (époque 4) | **78 s** |
| A strict : classifieur linéaire sur features figées | 1.041 | 56.9 % | quelques secondes |

C est retenue sur la plus petite perte de validation. Le « A strict » (features calculées une fois, backbone en mode eval,
une seule couche `nn.Linear`) fait nettement mieux que mon A initial : dans A, `model.train()` laissait les statistiques
de BatchNorm du backbone changer, ce qui le dégradait. Un classifieur linéaire sur features pré-entraînées est donc
presque aussi bon que le fine-tuning (écart de 1.8 point, dans la marge de ±6 points).

## Étape 4 : mesure finale sur le test (une seule fois) et recommandation

Script : `m4_final.py` (répétition à blanc sur la validation, puis `python m4_final.py test` une seule fois ; le script refuse de se relancer).
Résultats bruts : `results/m4_final.json`, `results/m4_final_table.csv`.

| Modèle | Exactitude **test** (±95 %) | Val | Paramètres | Taille | ms/image (1 thread) | ms/image (4 threads) | Entraînement |
|---|---|---|---|---|---|---|---|
| Hasard | 20.0 % | | | | | | |
| Régression logistique (pixels 32x32) | 28.9 % (±5.9) | 34.2 % | 15 365 | 0.06 Mo | 0.12 | - | 0.5 s |
| MLP 16 neurones (pixels 32x32) | 30.2 % (±6.0) | 34.2 % | 49 253 | 0.20 Mo | 0.05 | 0.04 | 0.8 s |
| **ResNet-18, layer4 dégelé (128x128)** | **62.2 % (±6.3)** | 58.7 % | 11 179 077 | 44.72 Mo | 18.94 | 11.45 | 78 s |

Test : 225 images jamais utilisées avant. ±95 % = intervalle de confiance binomial. Temps mesurés sur mon ordinateur portable
(CPU, une image à la fois, modèle seul : lecture et redimensionnement de la photo exclus). Entraînement du ResNet : 78 s, durée mesurée dans `m3c_layer4.py`.

![Matrice de confusion du ResNet-18 sur le test](results/m4_confusion_test_resnet.png)

Exactitude par style sur le test : Cubism 84 %, Baroque 69 %, Impressionism 62 %, Expressionism 53 %, Romanticism 42 %.
Confusions principales : Romanticism pris pour Baroque (14) ou Impressionism (7), Impressionism pris pour Romanticism (10).

### Recommandation au directeur

**Oui, un réseau de neurones vaut le coup, mais un réseau pré-entraîné, pas un réseau entraîné de zéro.**

- **Précision.** Le ResNet-18 reconnaît le bon style dans 62 % des cas contre 29-30 % pour les modèles simples sur pixels
  (hasard : 20 %). L'écart de plus de 30 points est bien supérieur à l'incertitude (environ ±6 points).
- **Bornes à CPU modeste.** Le ResNet-18 prend 19 ms par image avec un seul cœur : imperceptible pour un visiteur.
  Il pèse 45 Mo, ce qui passe sans problème sur un téléphone ou une borne. Le critère « CPU modeste » n'est donc pas bloquant.
- **Petite équipe, budget limité.** Pas de GPU : l'entraînement a pris 78 s sur un ordinateur portable. On réutilise un modèle
  public (ResNet-18) et on n'ajoute qu'une petite partie à entraîner. Pas de service à maintenir : le modèle tourne sur l'appareil.
- **Option la plus simple à maintenir.** La variante gelée « A strict » est un classifieur linéaire sur des features pré-entraînées :
  on utilise le ResNet tel quel pour décrire l'image, puis une simple couche linéaire donne le style. Sur la validation elle
  atteint 56.9 % contre 58.7 % pour la variante retenue (écart non significatif ; elle n'a pas été mesurée sur le test).
  C'est la variante à essayer en premier si l'équipe veut un entraînement presque instantané et peu de risques de surapprentissage.
- **Avant la mise en service.** Environ 4 réponses sur 10 restent fausses : proposer les deux styles les plus probables ou afficher un
  niveau de confiance, et tester sur de vraies photos de visiteurs.

Un modèle classique n'est pas assez précis ici, mais seulement celui testé (pixels bruts). Des descripteurs classiques conçus à la main
(histogrammes de couleurs, textures) n'ont pas été essayés.

### Limites

- **Peu de données** : 1 500 images (300 par style), 225 images de test. L'incertitude est d'environ ±6 points.
  Les écarts de quelques points entre A, B et C ne sont pas significatifs.
- **5 styles seulement** parmi les 27 du jeu de données : un vrai musée aurait plus de styles, plus proches, et la précision baisserait.
- **Une seule seed** (42) et un seul split : la variabilité d'une exécution à l'autre n'est pas mesurée.
- **Biais d'artiste possible** : un même peintre apparaît dans plusieurs images (limité à 50 par artiste), et 60 % des images viennent
  d'« artiste inconnu ». Le modèle peut reconnaître un peintre ou une collection plutôt qu'un style. Aucun split séparé par artiste n'a été testé.
- **Images WikiArt, pas de vraies photos** : reflets, angles et éclairage d'un téléphone ne sont pas représentés. Images réduites à 128x128.
- **Temps mesurés sur mon ordinateur**, pas sur la vraie borne ; ils varient d'une mesure à l'autre
  (la même mesure du ResNet donnait 24.6 ms à l'étape 3, 11.5 ms ici avec 4 threads).
- **Baseline classique limitée** à la régression logistique sur pixels.
