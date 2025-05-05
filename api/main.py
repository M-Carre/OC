# api/main.py (Version Baseline TF-IDF + LogReg AVEC Lemmatisation)

import os
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager
import traceback

# --- Framework API ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- ML & Preprocessing (Baseline + SpaCy) ---
import joblib # Pour charger le pipeline sklearn
import numpy as np
try:
    import spacy
    spacy_available = True
except ImportError:
    print("⚠️ WARNING: spaCy non trouvé. La lemmatisation (requise pour ce modèle) sera impossible.")
    spacy_available = False

# ==================================================
# Configuration et Variables Globales
# ==================================================
print("--- Initialisation Globale API Baseline (avec SpaCy) ---")

API_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = API_DIR / "baseline_pipeline.joblib"
SPACY_MODEL = "en_core_web_sm" # Modèle utilisé pour la lemmatisation

# Dictionnaire pour les ressources
ml_resources = {
    "pipeline": None,
    "nlp_spacy": None # On charge aussi SpaCy
}

# ==================================================
# Fonction de Chargement (appelée au démarrage)
# ==================================================
def load_ml_resources():
    print("\n--- Démarrage de la fonction load_ml_resources (Baseline + SpaCy) ---")
    global ml_resources

    # --- Charger le Pipeline Sklearn ---
    print(f"Chargement Pipeline Sklearn depuis : {PIPELINE_PATH}")
    if PIPELINE_PATH.exists():
        try:
            pipeline_local = joblib.load(PIPELINE_PATH)
            print("✅ Pipeline Sklearn chargé.")
            if 'tfidf' in pipeline_local.named_steps and 'clf' in pipeline_local.named_steps:
                 print("   (Étapes 'tfidf' et 'clf' trouvées)")
                 ml_resources['pipeline'] = pipeline_local
            else:
                 print("❌ ERREUR: Pipeline invalide.")
                 ml_resources['pipeline'] = None
        except Exception as e:
            print(f"❌ ERREUR Pipeline joblib.load: {e}")
            traceback.print_exc()
            ml_resources['pipeline'] = None
    else:
        print(f"❌ Fichier pipeline non trouvé à {PIPELINE_PATH}")
        ml_resources['pipeline'] = None
    print(f"  >> Valeur ml_resources['pipeline'] après tentative: {type(ml_resources.get('pipeline'))}")

    # --- Charger spaCy ---
    if spacy_available:
        try:
            print(f"Chargement spaCy '{SPACY_MODEL}'...")
            nlp_spacy_local = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
            print("✅ spaCy chargé.")
            ml_resources['nlp_spacy'] = nlp_spacy_local
        except Exception as e: # Peut échouer si modèle non téléchargé
            print(f"❌ ERREUR spaCy load_model: {e}")
            print(f"   Vérifiez que le modèle '{SPACY_MODEL}' est installé sur le serveur.")
            traceback.print_exc()
            ml_resources['nlp_spacy'] = None
    else:
        print("Skipping spaCy loading (library not found).")
        ml_resources['nlp_spacy'] = None
    print(f"  >> Valeur ml_resources['nlp_spacy'] après tentative: {type(ml_resources.get('nlp_spacy'))}")


    # Vérification finale
    if not ml_resources.get('pipeline'):
         print("\n⚠️ AVERTISSEMENT: Chargement du pipeline baseline ÉCHOUÉ.")
    elif not ml_resources.get('nlp_spacy'):
         print("\n⚠️ AVERTISSEMENT: Chargement de spaCy (requis pour lemmatisation) ÉCHOUÉ.")
    else:
         print("\n✅ Pipeline baseline et spaCy semblent chargés.")

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
    title="Air Paradis Sentiment Analysis API (Baseline+SpaCy)", # Nom mis à jour
    description="API pour prédire le sentiment (positif/négatif) d'un tweet, basée sur un modèle TF-IDF + LogReg et lemmatisation SpaCy.",
    version="1.3.0", # Incrémenter version (SpaCy ré-intégré)
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
# Fonctions de Prétraitement (AVEC Lemmatisation)
# ==================================================
def clean_minimal(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Fonction de Lemmatisation RÉINTRODUITE ---
def lemmatize_text(text: str) -> str:
    nlp_spacy_local = ml_resources.get('nlp_spacy')
    if nlp_spacy_local is None:
        print("❌ ERREUR: spaCy non chargé, impossible de lemmatiser (requis pour ce modèle).")
        # Lever une exception pour arrêter le traitement si spaCy est essentiel
        raise ValueError("Le composant de lemmatisation (SpaCy) n'est pas disponible.")
    try:
        # print(f"Lemmatisation de: '{text[:50]}...'") # Debug si besoin
        doc = nlp_spacy_local(text)
        lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
        result = " ".join(lemmas)
        # print(f"Résultat lemmatisation: '{result[:50]}...'") # Debug si besoin
        return result
    except Exception as e:
        print(f"❌ ERREUR pendant la lemmatisation: {e}")
        traceback.print_exc()
        # Lever une exception pour arrêter
        raise ValueError(f"Erreur interne lors de la lemmatisation: {e}")

# ==================================================
# Endpoints de l'API (Adaptés pour Baseline + Lemmatisation)
# ==================================================
@app.get("/", tags=["Health Check"])
async def read_root():
    # Vérifier si les DEUX composants sont chargés
    return {"message": "API d'analyse de sentiments Air Paradis (Baseline+SpaCy) fonctionnelle!",
            "pipeline_loaded": ml_resources.get('pipeline') is not None,
            "spacy_loaded": ml_resources.get('nlp_spacy') is not None} # Ajout vérification SpaCy

@app.post("/predict", response_model=SentimentOutput, tags=["Prediction"])
async def predict_sentiment(tweet_input: TweetInput):
    # Récupérer les ressources chargées
    pipeline = ml_resources.get('pipeline')
    # spaCy est vérifié implicitement par lemmatize_text

    if pipeline is None:
        print("❌ ERREUR /predict: Pipeline non disponible.")
        raise HTTPException(status_code=503, detail="Modèle de prédiction (Baseline) non disponible.")

    # Prétraitement (Nettoyage + Lemmatisation)
    try:
        # start_process_time = time.time() # Debug
        cleaned_text = clean_minimal(tweet_input.tweet_text)
        # Appliquer la lemmatisation (lèvera une erreur si spaCy non chargé)
        lemmatized_text = lemmatize_text(cleaned_text)
        # print(f"Input pour pipeline predict: ['{lemmatized_text[:50]}...']") # Debug

    except ValueError as e: # Erreur venant de lemmatize_text si spaCy manque
        print(f"❌ ERREUR Prétraitement (ValueError): {e}")
        raise HTTPException(status_code=503, detail=f"Erreur de prétraitement: {e}") # 503 car composant manquant
    except Exception as e: # Autres erreurs
        print(f"❌ ERREUR Inattendue Prétraitement: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur interne lors du prétraitement.")

    # Prédiction avec le pipeline Sklearn sur le texte lemmatisé
    try:
        # start_pred_time = time.time() # Debug
        # Prédire la classe (0 ou 1)
        prediction_label_array = pipeline.predict([lemmatized_text])
        prediction_label = int(prediction_label_array[0])

        # Prédire les probabilités
        prediction_proba_array = pipeline.predict_proba([lemmatized_text])
        prediction_score = float(prediction_proba_array[0][1])

        # end_pred_time = time.time() # Debug
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