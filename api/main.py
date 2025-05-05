"""
FastAPI application pour l'analyse de sentiments Air Paradis.
Charge le modèle LSTM final, le tokenizer, et la config.
Expose un endpoint /predict.
"""

import os
import pickle
import json
import re
import time # Pour mesurer le temps de prédiction
from pathlib import Path # Pour gérer les chemins de manière plus robuste

# --- Framework API ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- ML & Preprocessing ---
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import numpy as np

# ==================================================
# Configuration et Chargement Initial
# ==================================================
print("--- Démarrage de l'application API Sentiment ---")

# --- Définir les chemins relatifs au script main.py ---
# On suppose que les fichiers sont dans le même dossier que main.py
BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = BASE_DIR / "Copie de tokenizer_full.pkl"
CONFIG_PATH = BASE_DIR / "Copie de preprocessing_config.json"
MODEL_PATH = BASE_DIR / "Copie de final_lstm_full_data.keras" # Le checkpoint Keras

# --- Variables globales pour les objets chargés ---
model = None
tokenizer = None
config = None
nlp_spacy = None # Pour spaCy

# --- Charger spaCy ---
# Mettre dans un try-except pour gérer le cas où le modèle n'est pas installé
SPACY_MODEL = "en_core_web_sm"
try:
    print(f"Chargement du modèle spaCy '{SPACY_MODEL}'...")
    # Désactiver les composants non nécessaires peut accélérer légèrement
    nlp_spacy = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
    print("✅ Modèle spaCy chargé.")
except OSError:
    print(f"❌ ERREUR: Modèle spaCy '{SPACY_MODEL}' non trouvé.")
    print("L'API ne pourra pas lemmatiser. Installez-le via:")
    print(f"python -m spacy download {SPACY_MODEL}")
    # On pourrait décider de stopper l'API ici ou continuer sans lemmatisation
    # Pour l'instant, on continue, mais la fonction de prétraitement échouera si spaCy est utilisé.
    nlp_spacy = None
except Exception as e:
    print(f"❌ ERREUR inattendue lors du chargement de spaCy: {e}")
    nlp_spacy = None


# --- Charger le Tokenizer ---
print(f"Chargement du Tokenizer depuis : {TOKENIZER_PATH}")
if TOKENIZER_PATH.exists():
    try:
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✅ Tokenizer chargé.")
    except Exception as e:
        print(f"❌ ERREUR lors du chargement du tokenizer : {e}")
        tokenizer = None # Marquer comme non chargé
else:
    print(f"❌ ERREUR: Fichier Tokenizer non trouvé à {TOKENIZER_PATH}")
    tokenizer = None

# --- Charger la Configuration (maxlen) ---
print(f"Chargement de la configuration depuis : {CONFIG_PATH}")
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            # Vérifier si maxlen est bien dans le fichier
            if 'maxlen' not in config:
                print("❌ ERREUR: La clé 'maxlen' manque dans le fichier de configuration.")
                config = None
            else:
                 print(f"✅ Configuration chargée (maxlen={config.get('maxlen')}).")
    except Exception as e:
        print(f"❌ ERREUR lors du chargement de la configuration : {e}")
        config = None
else:
    print(f"❌ ERREUR: Fichier de configuration non trouvé à {CONFIG_PATH}")
    config = None

# --- Charger le Modèle Keras ---
print(f"Chargement du modèle Keras depuis : {MODEL_PATH}")
if MODEL_PATH.exists():
    try:
        # Charger le modèle complet (structure + poids)
        # On pourrait ajouter compile=False si on n'a pas besoin de l'optimiseur, etc.
        # Mais le garder peut être plus sûr si des métriques custom étaient utilisées.
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modèle Keras chargé.")
        # Optionnel : Afficher un résumé pour vérifier
        # model.summary()
    except Exception as e:
        print(f"❌ ERREUR lors du chargement du modèle Keras : {e}")
        model = None
else:
    print(f"❌ ERREUR: Fichier modèle Keras non trouvé à {MODEL_PATH}")
    model = None

# --- Vérification finale ---
if not all([model, tokenizer, config, nlp_spacy]):
    print("\n‼️ ATTENTION: Un ou plusieurs composants essentiels n'ont pas pu être chargés.")
    print("   L'endpoint /predict risque de ne pas fonctionner correctement.")
else:
    print("\n✅ Tous les composants semblent chargés. Prêt à définir l'API.")

# ==================================================
# Définition de l'Application FastAPI
# ==================================================
app = FastAPI(
    title="Air Paradis Sentiment Analysis API",
    description="API pour prédire le sentiment (positif/négatif) d'un tweet, basée sur un modèle LSTM et FastText.",
    version="1.0.0" # Version initiale après entraînement complet
)

# ==================================================
# Modèles Pydantic (pour validation des requêtes/réponses)
# ==================================================
class TweetInput(BaseModel):
    tweet_text: str

class SentimentOutput(BaseModel):
    sentiment_label: str # "positif" ou "negatif"
    sentiment_score: float # Score brut du modèle (probabilité d'être positif)

# ==================================================
# Fonctions de Prétraitement
# ==================================================

# --- Fonction de Nettoyage Minimal (identique au notebook) ---
def clean_minimal(text: str) -> str:
    if not isinstance(text, str): # Garde-fou
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'#', '', text) # Supprimer aussi les hashtags ? À discuter.
    text = re.sub(r'&[a-z]+;', '', text) # Entités HTML comme &
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Fonction de Lemmatisation (utilisant spaCy chargé) ---
def lemmatize_text(text: str) -> str:
    if nlp_spacy is None:
        print("⚠️ spaCy non chargé, lemmatisation impossible. Retour du texte nettoyé.")
        return text # Retourne le texte juste nettoyé si spaCy n'est pas là
    # Traiter le texte avec spaCy
    doc = nlp_spacy(text)
    # Extraire les lemmes (en minuscules, sans stop words ni ponctuation)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    return " ".join(lemmas)

# --- Fonction de Prétraitement Complète ---
def preprocess_for_model(text: str) -> np.ndarray:
    """
    Applique toute la chaîne de prétraitement nécessaire avant l'inférence.
    """
    start_time = time.time()

    # 1. Vérifier les dépendances chargées
    if not all([tokenizer, config]):
        print("❌ ERREUR Prétraitement: Tokenizer ou Config manquants.")
        # Lever une exception ici informe FastAPI de l'erreur interne
        raise ValueError("Les composants de prétraitement (tokenizer/config) ne sont pas chargés.")

    # 2. Nettoyage et Lemmatisation
    cleaned_text = clean_minimal(text)
    lemmatized_text = lemmatize_text(cleaned_text) # Utilise spaCy si disponible

    # 3. Tokenisation Keras
    # texts_to_sequences attend une liste, même pour un seul texte
    sequence = tokenizer.texts_to_sequences([lemmatized_text])
    # sequence est une liste de listes, ex: [[12, 5, 34]]

    # 4. Padding
    maxlen = config.get('maxlen') # Récupérer maxlen depuis la config chargée
    if maxlen is None:
        raise ValueError("La valeur 'maxlen' est manquante dans la configuration chargée.")

    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='pre', truncating='post')
    # padded_sequence est maintenant un array NumPy, ex: array([[ 0, 0, ..., 12, 5, 34]])

    end_time = time.time()
    print(f"Prétraitement effectué en {(end_time - start_time)*1000:.2f} ms")
    return padded_sequence # Retourne l'array NumPy prêt pour model.predict

# ==================================================
# Endpoints de l'API
# ==================================================

@app.get("/", tags=["Health Check"])
async def read_root():
    """Point de terminaison racine pour vérifier si l'API est en ligne."""
    print("Requête GET reçue sur /")
    return {"message": "API d'analyse de sentiments Air Paradis fonctionnelle!",
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "config_loaded": config is not None,
            "spacy_loaded": nlp_spacy is not None}

@app.post("/predict", response_model=SentimentOutput, tags=["Prediction"])
async def predict_sentiment(tweet_input: TweetInput):
    """
    Prédit le sentiment d'un tweet donné.
    - **tweet_text**: Le texte brut du tweet à analyser.
    """
    print(f"\nRequête POST reçue sur /predict avec texte: '{tweet_input.tweet_text[:50]}...'") # Log tronqué

    # --- Vérifier si le modèle est chargé ---
    if model is None:
        print("❌ ERREUR: Modèle non chargé, impossible de prédire.")
        raise HTTPException(status_code=503, # Service Unavailable
                            detail="Le modèle de prédiction n'est pas disponible.")

    # --- Prétraitement ---
    try:
        processed_input = preprocess_for_model(tweet_input.tweet_text)
        print(f"Shape après prétraitement: {processed_input.shape}")
    except ValueError as e: # Erreur attrapée depuis preprocess_for_model
        print(f"❌ ERREUR pendant le prétraitement: {e}")
        raise HTTPException(status_code=500, # Internal Server Error
                            detail=f"Erreur interne lors du prétraitement: {e}")
    except Exception as e: # Attraper d'autres erreurs potentielles
        print(f"❌ ERREUR inattendue pendant le prétraitement: {e}")
        # Logguer l'erreur complète côté serveur serait bien ici
        raise HTTPException(status_code=500, detail="Erreur interne inattendue lors du prétraitement.")

    # --- Prédiction ---
    try:
        start_pred_time = time.time()
        # model.predict renvoie un array NumPy, ex: array([[0.987]])
        prediction_score = model.predict(processed_input, verbose=0)[0][0] # Extraire le score scalaire
        end_pred_time = time.time()
        print(f"Prédiction effectuée en {(end_pred_time - start_pred_time)*1000:.2f} ms - Score: {prediction_score:.4f}")

        # --- Formatage de la Réponse ---
        sentiment_label = "positif" if prediction_score > 0.5 else "negatif"

        return SentimentOutput(
            sentiment_label=sentiment_label,
            sentiment_score=float(prediction_score) # Assurer que c'est un float standard pour JSON
        )

    except Exception as e:
        print(f"❌ ERREUR lors de la prédiction du modèle : {e}")
        # Logguer l'erreur complète
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'exécution du modèle.")

# ==================================================
# Lancement Uvicorn (pour info, ne pas exécuter directement dans Colab)
# ==================================================
# if __name__ == "__main__":
#     import uvicorn
#     print("Pour lancer l'API localement (hors Colab), utilisez:")
#     print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
#     # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)