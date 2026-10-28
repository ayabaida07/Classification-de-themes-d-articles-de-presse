import argparse
from predict import predict

parser = argparse.ArgumentParser(description="Classification de thème d'article de presse")
parser.add_argument("--text", help="Texte de l'article")
parser.add_argument("--file", help="Fichier .txt contenant l'article")
args = parser.parse_args()

if args.file:
    text = open(args.file, encoding="utf-8").read()
elif args.text:
    text = args.text
else:
    parser.error("Donne --text ou --file")

best, results = predict(text)
print(f"\nCatégorie prédite : {best}\n")
print("Catégories alternatives :")
for label, score in results:
    print(f"  {label:<18} {score:.1%}")