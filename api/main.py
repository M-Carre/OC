# api/main.py (Version Baseline TF-IDF + LogReg)

import os
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager
import traceback

# --- Framework API ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- ML & Preprocessing (Baseline) ---
import joblib # Pour charger le pipeline sklearn
import numpy as np
# Note: On n'importe PLUS tensorflow, spacy, keras ici !

# ==================================================
# Configuration et Variables Globales
# ==================================================
print("--- Initialisation Globale API Baseline ---")

API_DIR = Path(__file__).resolve().parent
# --- MODIFIÉ ---
PIPELINE_PATH = API_DIR / "baseline_pipeline.joblib"
# -------------

# Dictionnaire pour les ressources
ml_resources = {
    "pipeline": None
}

# ==================================================
# Fonction de Chargement (appelée au démarrage)
# ==================================================
def load_ml_resources():
    print("\n--- Démarrage de la fonction load_ml_resources (Baseline) ---")
    global ml_resources

    # --- Charger le Pipeline Sklearn ---
    print(f"Chargement Pipeline Sklearn depuis : {PIPELINE_PATH}")
    if PIPELINE_PATH.exists():
        try:
            pipeline_local = joblib.load(PIPELINE_PATH)
            print("✅ Pipeline Sklearn chargé.")
            # Vérifier que c'est bien un pipeline avec les bonnes étapes
            if 'tfidf' in pipeline_local.named_steps and 'clf' in pipeline_local.named_steps:
                 print("   (Étapes 'tfidf' et 'clf' trouvées)")
                 ml_resources['pipeline'] = pipeline_local
            else:
                 print("❌ ERREUR: Le fichier chargé ne semble pas être le bon pipeline (étapes manquantes).")
                 ml_resources['pipeline'] = None

        except Exception as e:
            print(f"❌ ERREUR Pipeline joblib.load: {e}")
            traceback.print_exc()
            ml_resources['pipeline'] = None
    else:
        print(f"❌ Fichier pipeline non trouvé à {PIPELINE_PATH}")
        ml_resources['pipeline'] = None
    print(f"  >> Valeur ml_resources['pipeline'] après tentative: {type(ml_resources.get('pipeline'))}")

    if not ml_resources.get('pipeline'):
         print("\n⚠️ AVERTISSEMENT: Chargement du pipeline baseline ÉCHOUÉ.")
    else:
         print("\n✅ Pipeline baseline semble chargé.")

# ==================================================
# Événements Startup/Shutdown FastAPI
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Exécution de l'événement startup (lifespan) ---")
    load_ml_resources()
    yield
    print("--- Exécution de l'événement shutdown (lifespan) ---")
    ml_resources.clear()

# ==================================================
# Création de l'Application FastAPI AVEC Lifespan
# ==================================================
app = FastAPI(
    title="Air Paradis Sentiment Analysis API (Baseline)", # Nom mis à jour
    description="API pour prédire le sentiment (positif/négatif) d'un tweet, basée sur un modèle TF-IDF + LogReg.",
    version="1.2.0", # Incrémenter version (passage Baseline)
    lifespan=lifespan
)

# ==================================================
# Modèles Pydantic (inchangés)
# ==================================================
class TweetInput(BaseModel):
    tweet_text: str

class SentimentOutput(BaseModel):
    sentiment_label: str
    sentiment_score: float # Probabilité de la classe positive

# ==================================================
# Fonctions de Prétraitement (Simplifié pour Baseline)
# ==================================================
def clean_minimal(text: str) -> str:
    # Garder la même fonction de nettoyage de base
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# PAS BESOIN de lemmatize_text ni preprocess_for_model complexe ici

# ==================================================
# Endpoints de l'API (Adaptés pour Baseline)
# ==================================================
@app.get("/", tags=["Health Check"])
async def read_root():
    # Vérifier si le pipeline est chargé
    return {"message": "API d'analyse de sentiments Air Paradis (Baseline) fonctionnelle!",
            "pipeline_loaded": ml_resources.get('pipeline') is not None}

@app.post("/predict", response_model=SentimentOutput, tags=["Prediction"])
async def predict_sentiment(tweet_input: TweetInput):
    # Récupérer le pipeline chargé
    pipeline = ml_resources.get('pipeline')

    if pipeline is None:
        print("❌ ERREUR /predict: Pipeline non disponible.")
        raise HTTPException(status_code=503, detail="Modèle de prédiction (Baseline) non disponible.")

    # Prétraitement (nettoyage seulement + TF-IDF via pipeline)
    try:
        start_process_time = time.time()
        cleaned_text = clean_minimal(tweet_input.tweet_text)
        # Le pipeline s'attend à une liste ou itérable de documents
        # L'étape TF-IDF fera la vectorisation
        # L'étape 'clf' fera la prédiction
        # Note: On ne peut pas séparer facilement transform et predict ici
        #       si on veut aussi predict_proba. On prédit les deux.
        #       Si le pipeline a été entraîné sur text_lemma, il faut lemmatiser ici !
        #       MAIS on a dit qu'on simplifiait. On suppose que la baseline
        #       peut tourner sur du texte juste nettoyé ou qu'on ajoute la lemmatisation.
        #       => AJOUTONS la lemmatisation si spaCy est dispo, sinon juste nettoyé.

        # --- AJOUT potentiel Lemmatisation ---
        # if ml_resources.get('nlp_spacy'):
        #      # Recréer la fonction lemmatize_text ici ou l'importer
        #      # cleaned_text = lemmatize_text(cleaned_text)
        #      pass # Supposons pour l'instant qu'on utilise juste clean_minimal
        # -----------------------------------

        print(f"Input pour pipeline predict: ['{cleaned_text[:50]}...']") # Debug

    except Exception as e:
        print(f"❌ ERREUR Prétraitement simple: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur interne lors du prétraitement simple.")

    # Prédiction avec le pipeline Sklearn
    try:
        start_pred_time = time.time()

        # Prédire la classe (0 ou 1)
        prediction_label_array = pipeline.predict([cleaned_text])
        prediction_label = int(prediction_label_array[0]) # Obtenir 0 ou 1

        # Prédire les probabilités [prob_classe_0, prob_classe_1]
        prediction_proba_array = pipeline.predict_proba([cleaned_text])
        # Prendre la probabilité de la classe positive (index 1)
        prediction_score = float(prediction_proba_array[0][1])

        end_pred_time = time.time()
        # print(f"Prédiction Baseline effectuée en {(end_pred_time - start_pred_time)*1000:.2f} ms") # Debug

        # Formatage Réponse
        sentiment_label = "positif" if prediction_label == 1 else "negatif"
        return SentimentOutput(
            sentiment_label=sentiment_label,
            sentiment_score=prediction_score
        )
    except Exception as e:
        print(f"❌ ERREUR lors de pipeline.predict/predict_proba: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'exécution du modèle baseline.")

# --- Lancement Uvicorn (Commenté) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Lancer avec: uvicorn main:app --reload --host 0.0.0.0 --port 8000")