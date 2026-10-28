# Classification de thème d'articles de presse

Application de classification de texte : à partir d'un article de presse **en anglais**, le modèle prédit son thème parmi 12 catégories et renvoie les alternatives les plus probables avec leur score.

## Fonctionnement

1. **Prétraitement** ([preprocess.py](preprocess.py)) : suppression des URL et balises HTML, puis lemmatisation avec spaCy (`en_core_web_sm`), en retirant les mots vides et les tokens non alphabétiques ou de moins de 3 lettres.
2. **Vectorisation** : TF-IDF ([models/tfidf.joblib](models/tfidf.joblib)), 20 000 termes, unigrammes et bigrammes.
3. **Classification** : régression logistique ([models/logreg.joblib](models/logreg.joblib)).
4. **Prédiction** ([predict.py](predict.py)) : renvoie les `top_k` catégories (3 par défaut) avec leur probabilité. Si la meilleure probabilité est inférieure au seuil (`threshold=0.4` par défaut), l'article est classé `INCERTAIN (hors des catégories connues)`.

### Catégories

`ARTS & CULTURE`, `BUSINESS`, `ENTERTAINMENT`, `FOOD & DRINK`, `PARENTING`, `POLITICS`, `SCIENCE & TECH`, `SPORTS`, `STYLE & BEAUTY`, `TRAVEL`, `WELLNESS`, `WORLD NEWS`

## Données et entraînement

L'entraînement est détaillé dans le notebook [notebook/Classification d'articles de presse notebook.ipynb](notebook/Classification%20d'articles%20de%20presse%20notebook.ipynb) (exécuté sur Kaggle).

- **Jeu de données** : [News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset), 209 527 articles répartis en 42 catégories.
- **Préparation** : fusion des catégories proches (`ARTS` + `CULTURE & ARTS` → `ARTS & CULTURE`, `TECH` + `SCIENCE` → `SCIENCE & TECH`, `HEALTHY LIVING` → `WELLNESS`, `MONEY` → `BUSINESS`, `PARENTS` → `PARENTING`, etc.), puis sélection de 12 catégories (151 352 articles). Le texte d'entrée est le titre suivi de la description courte. Après suppression des textes trop courts et des doublons, puis du prétraitement : **150 822 articles**. Le vocabulaire passe de 213 545 à 68 175 mots.
- **Découpage** : 80 % entraînement (120 657 articles) / 20 % test (30 165 articles), stratifié, réalisé avant l'ajustement du TF-IDF pour éviter toute fuite de données.
- **Modèle retenu** : régression logistique (`max_iter=1000`, `class_weight="balanced"`).

## Performances

### Choix de la représentation

Comparaison sur un échantillon de 20 000 articles (16 000 entraînement / 4 000 test), avec une régression logistique :

| Représentation | Accuracy | F1 macro |
|----------------|----------|----------|
| **TF-IDF**     | 0,7333   | 0,6847   |
| Doc2Vec        | 0,4848   | 0,4402   |

### Choix du modèle

Évaluation sur le jeu de test complet (30 165 articles), avec TF-IDF :

| Modèle                       | Accuracy | F1 macro | mAP    |
|------------------------------|----------|----------|--------|
| **Régression logistique**    | 0,7764   | 0,7389   | 0,7898 |
| Random Forest (100 arbres)   | 0,7113   | 0,6592   | 0,7000 |

### Résultats par catégorie (régression logistique)

| Catégorie        | Précision | Rappel | F1   | AP (mAP) | Support |
|------------------|-----------|--------|------|----------|---------|
| ARTS & CULTURE   | 0,49      | 0,67   | 0,57 | 0,585    | 782     |
| BUSINESS         | 0,57      | 0,69   | 0,62 | 0,641    | 1 546   |
| ENTERTAINMENT    | 0,81      | 0,76   | 0,78 | 0,848    | 3 468   |
| FOOD & DRINK     | 0,76      | 0,85   | 0,80 | 0,843    | 1 685   |
| PARENTING        | 0,77      | 0,78   | 0,77 | 0,813    | 2 532   |
| POLITICS         | 0,92      | 0,79   | 0,85 | 0,932    | 7 107   |
| SCIENCE & TECH   | 0,47      | 0,66   | 0,55 | 0,613    | 852     |
| SPORTS           | 0,69      | 0,84   | 0,76 | 0,795    | 1 014   |
| STYLE & BEAUTY   | 0,85      | 0,83   | 0,84 | 0,909    | 2 385   |
| TRAVEL           | 0,78      | 0,79   | 0,78 | 0,848    | 1 976   |
| WELLNESS         | 0,83      | 0,75   | 0,79 | 0,864    | 4 913   |
| WORLD NEWS       | 0,69      | 0,80   | 0,74 | 0,786    | 1 905   |
| **Moyenne macro**| 0,72      | 0,77   | 0,74 | 0,7898   | 30 165  |

Les catégories les mieux reconnues sont `POLITICS`, `STYLE & BEAUTY` et `FOOD & DRINK`. Les plus difficiles sont `ARTS & CULTURE`, `SCIENCE & TECH` et `BUSINESS`.

## Structure du projet

```
projet3_app/
├── models/
│   ├── tfidf.joblib      # vectoriseur TF-IDF entraîné
│   └── logreg.joblib     # modèle de régression logistique entraîné
├── notebook/
│   └── Classification d'articles de presse notebook.ipynb   # entraînement et évaluation
├── preprocess.py         # nettoyage et lemmatisation du texte
├── predict.py            # fonction predict(text, threshold, top_k)
├── predict_cli.py        # interface en ligne de commande
└── requirements.txt      # dépendances Python
```

## Installation

Prérequis : Python 3.13 (version utilisée pour le développement).

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Les fichiers `.joblib` ont été produits avec `scikit-learn==1.6.1`. Garder cette version pour éviter les incompatibilités au chargement.

## Utilisation

### En ligne de commande

Depuis la racine du projet (les modèles sont chargés par chemin relatif) :

```powershell
# Texte passé directement
python predict_cli.py --text "The team won the championship after a thrilling final match."

# Texte lu depuis un fichier .txt (UTF-8)
python predict_cli.py --file article.txt
```

Exemple de sortie :

```
Catégorie prédite : SPORTS

Catégories alternatives :
  SPORTS             87.3%
  ENTERTAINMENT      4.1%
  WORLD NEWS         2.5%
```

(Valeurs données à titre d'illustration.)

### Depuis Python

```python
from predict import predict

best, results = predict("Stocks rallied as the central bank cut interest rates.", threshold=0.4, top_k=3)
print(best)      # catégorie prédite, ou "INCERTAIN (...)" si le score < threshold
print(results)   # [(catégorie, probabilité), ...]
```

## Limites

- Le modèle est entraîné sur des textes **en anglais** ; les articles dans d'autres langues donneront des résultats peu fiables.
- Les articles hors des 12 catégories connues sont signalés comme incertains seulement si la probabilité maximale passe sous le seuil.
