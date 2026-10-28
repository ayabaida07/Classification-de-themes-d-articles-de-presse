# Classification de thème d'articles de presse

Application de classification de texte : à partir d'un article de presse **en anglais**, le modèle prédit son thème parmi 12 catégories et renvoie les alternatives les plus probables avec leur score.

## Fonctionnement

1. **Prétraitement** ([preprocess.py](preprocess.py)) : suppression des URL et balises HTML, puis lemmatisation avec spaCy (`en_core_web_sm`), en retirant les mots vides et les tokens non alphabétiques ou de moins de 3 lettres.
2. **Vectorisation** : TF-IDF ([models/tfidf.joblib](models/tfidf.joblib)).
3. **Classification** : régression logistique ([models/logreg.joblib](models/logreg.joblib)).
4. **Prédiction** ([predict.py](predict.py)) : renvoie les `top_k` catégories (3 par défaut) avec leur probabilité. Si la meilleure probabilité est inférieure au seuil (`threshold=0.4` par défaut), l'article est classé `INCERTAIN (hors des catégories connues)`.

### Catégories

`ARTS & CULTURE`, `BUSINESS`, `ENTERTAINMENT`, `FOOD & DRINK`, `PARENTING`, `POLITICS`, `SCIENCE & TECH`, `SPORTS`, `STYLE & BEAUTY`, `TRAVEL`, `WELLNESS`, `WORLD NEWS`

## Structure du projet

```
projet3_app/
├── models/
│   ├── tfidf.joblib      # vectoriseur TF-IDF entraîné
│   └── logreg.joblib     # modèle de régression logistique entraîné
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
