import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")

def basic_clean(text):
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text):
    doc = nlp(basic_clean(text))
    return " ".join(t.lemma_.lower() for t in doc
                    if t.is_alpha and not t.is_stop and len(t) > 2)