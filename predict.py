import joblib
from preprocess import preprocess

tfidf = joblib.load("models/tfidf.joblib")
model = joblib.load("models/logreg.joblib")

def predict(text, threshold=0.4, top_k=3):
    vec = tfidf.transform([preprocess(text)])
    probs = model.predict_proba(vec)[0]
    order = probs.argsort()[::-1][:top_k]
    results = [(model.classes_[i], float(probs[i])) for i in order]
    best, score = results[0]
    if score < threshold:
        best = "INCERTAIN (hors des catégories connues)"
    return best, results