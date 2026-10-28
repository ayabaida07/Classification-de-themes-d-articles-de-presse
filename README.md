# Classification de thème d'articles de presse

Application de classification de texte : à partir d'un article de presse **en anglais**, le modèle prédit son thème parmi 12 catégories et renvoie les alternatives les plus probables avec leur score.

## Données

Jeu de données [News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (Kaggle, articles HuffPost) : 209 527 articles répartis en 42 catégories.

Préparation (voir le [notebook](notebook/Classification%20d'articles%20de%20presse%20notebook.ipynb)) :

- **Fusion de catégories proches** : `ARTS` + `CULTURE & ARTS` → `ARTS & CULTURE`, `TECH` + `SCIENCE` → `SCIENCE & TECH`, `HEALTHY LIVING` → `WELLNESS`, `MONEY` → `BUSINESS`, `PARENTS` → `PARENTING`, etc.
- **Sélection de 12 catégories** (151 352 articles).
- **Texte d'entrée** = titre + description courte, puis suppression des textes trop courts (≤ 20 caractères) et des doublons → **150 822 articles** après prétraitement.
- **Découpage 80 / 20 stratifié** (120 657 articles d'entraînement, 30 165 de test), réalisé avant l'ajustement du TF-IDF pour éviter toute fuite de données.

### Catégories

`ARTS & CULTURE`, `BUSINESS`, `ENTERTAINMENT`, `FOOD & DRINK`, `PARENTING`, `POLITICS`, `SCIENCE & TECH`, `SPORTS`, `STYLE & BEAUTY`, `TRAVEL`, `WELLNESS`, `WORLD NEWS`

## Fonctionnement

1. **Prétraitement** ([preprocess.py](preprocess.py)) : suppression des URL et balises HTML, puis lemmatisation avec spaCy (`en_core_web_sm`), en retirant les mots vides et les tokens non alphabétiques ou de moins de 3 lettres. Le vocabulaire passe de 213 545 à 68 175 mots.
2. **Vectorisation** : TF-IDF (20 000 termes, unigrammes et bigrammes) — [models/tfidf.joblib](models/tfidf.joblib).
3. **Classification** : régression logistique (`class_weight="balanced"`) — [models/logreg.joblib](models/logreg.joblib).
4. **Prédiction** ([predict.py](predict.py)) : renvoie les `top_k` catégories (3 par défaut) avec leur probabilité. Si la meilleure probabilité est inférieure au seuil (`threshold=0.4` par défaut), l'article est classé `INCERTAIN (hors des catégories connues)`.

## Entraînement et évaluation

Le notebook [notebook/Classification d'articles de presse notebook.ipynb](notebook/Classification%20d'articles%20de%20presse%20notebook.ipynb) contient toute la démarche (exécuté sur Kaggle) : exploration, nettoyage, prétraitement, comparaison des représentations, modélisation, évaluation et export des modèles.

**Représentation** (échantillon de 20 000 articles, régression logistique) :

| Représentation | Accuracy | F1 macro |
|----------------|----------|----------|
| **TF-IDF**     | 0,733    | 0,685    |
| Doc2Vec        | 0,485    | 0,440    |

**Modèles** (corpus complet, TF-IDF, jeu de test de 30 165 articles) :

| Modèle                  | Accuracy | F1 macro | mAP    |
|-------------------------|----------|----------|--------|
| **Régression logistique** | 0,776  | 0,739    | 0,790  |
| Random Forest (100 arbres) | 0,71  | 0,66     | 0,70   |

La régression logistique est le modèle retenu. Les catégories les mieux reconnues sont `POLITICS`, `STYLE & BEAUTY` et `FOOD & DRINK` ; les plus difficiles sont `ARTS & CULTURE`, `SCIENCE & TECH` et `BUSINESS` (F1 entre 0,55 et 0,62), en partie à cause de leur faible effectif et de leur proximité avec d'autres thèmes.

## Structure du projet

```
.
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

Prérequis : Python 3.12 ou 3.13.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Les fichiers `.joblib` ont été produits avec `scikit-learn==1.6.1` et `spacy==3.8.16`. Garder ces versions pour éviter les incompatibilités au chargement.

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

- Le modèle est entraîné sur des textes **en anglais** (titre + description courte) ; les articles dans d'autres langues ou très longs donneront des résultats moins fiables.
- Les articles hors des 12 catégories connues sont signalés comme incertains seulement si la probabilité maximale passe sous le seuil.
- Les classes sont déséquilibrées (de 3 922 articles pour `ARTS & CULTURE` à 35 602 pour `POLITICS`) ; `class_weight="balanced"` compense en partie, mais les petites catégories restent moins bien prédites.
