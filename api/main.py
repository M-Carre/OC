# api/main.py

import os
import pickle
import json
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager # Pour gérer startup/shutdown
import traceback # Pour imprimer les traces d'erreur complètes

# --- Framework API ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- ML & Preprocessing ---
# Mettre les imports TF/SpaCy dans des try-except pour mieux gérer les erreurs d'import potentielles
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tf_available = True
except ImportError:
    print("⚠️ WARNING: TensorFlow non trouvé. L'API ne pourra pas charger/utiliser le modèle Keras.")
    tf_available = False
    # Définir une classe 'model' factice pour éviter les NameError plus tard si TF est absent
    class ModelMock:
        def predict(self, *args, **kwargs):
             raise RuntimeError("TensorFlow n'est pas installé, impossible de prédire.")
    model = ModelMock() # Sera écrasé si le chargement réussit

try:
    import spacy
    spacy_available = True
except ImportError:
    print("⚠️ WARNING: spaCy non trouvé. La lemmatisation sera désactivée.")
    spacy_available = False

import numpy as np

# ==================================================
# Configuration et Variables Globales (Initialisées à None)
# ==================================================
print("--- Initialisation Globale (avant startup) ---")

API_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = API_DIR / "tokenizer.pkl"
CONFIG_PATH = API_DIR / "config.json"
MODEL_PATH = API_DIR / "model.keras"
SPACY_MODEL = "en_core_web_sm"

# Dictionnaire pour contenir les ressources chargées
ml_models = {
    "tokenizer": None,
    "config": None,
    "nlp_spacy": None,
    "model": None
}

# ==================================================
# Fonction de Chargement (appelée au démarrage)
# ==================================================
def load_models():
    """Charge tous les modèles et dépendances nécessaires."""
    print("\n--- Démarrage de la fonction load_models ---")
    global ml_models # Indiquer qu'on modifie la variable globale

    # --- Charger spaCy ---
    if spacy_available:
        try:
            print(f"Chargement spaCy '{SPACY_MODEL}'...")
            # Utiliser une variable locale temporaire
            nlp_spacy_local = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
            print("✅ spaCy chargé.")
            ml_models['nlp_spacy'] = nlp_spacy_local # Mettre dans le dict seulement si succès
        except Exception as e:
            print(f"❌ ERREUR spaCy load_model: {e}")
            traceback.print_exc()
            ml_models['nlp_spacy'] = None # Assurer qu'il est None en cas d'échec
    else:
        print("Skipping spaCy loading (library not found).")
        ml_models['nlp_spacy'] = None
    print(f"  >> Valeur ml_models['nlp_spacy'] après tentative: {type(ml_models.get('nlp_spacy'))}")

    # --- Charger Tokenizer ---
    print(f"Chargement Tokenizer depuis : {TOKENIZER_PATH}")
    if TOKENIZER_PATH.exists():
        try:
            with open(TOKENIZER_PATH, 'rb') as handle:
                # Utiliser une variable locale temporaire
                tokenizer_local = pickle.load(handle)
            print("✅ Tokenizer chargé.")
            ml_models['tokenizer'] = tokenizer_local
        except Exception as e:
            print(f"❌ ERREUR Tokenizer pickle.load: {e}")
            traceback.print_exc()
            ml_models['tokenizer'] = None
    else:
        print(f"❌ Fichier Tokenizer non trouvé à {TOKENIZER_PATH}")
        ml_models['tokenizer'] = None
    print(f"  >> Valeur ml_models['tokenizer'] après tentative: {type(ml_models.get('tokenizer'))}")

    # --- Charger Config ---
    print(f"Chargement Config depuis : {CONFIG_PATH}")
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                 # Utiliser une variable locale temporaire
                config_local = json.load(f)
            if 'maxlen' not in config_local:
                print("❌ ERREUR: 'maxlen' manque dans config.")
                ml_models['config'] = None
            else:
                 print(f"✅ Config chargée (maxlen={config_local.get('maxlen')}).")
                 ml_models['config'] = config_local
        except Exception as e:
            print(f"❌ ERREUR Config json.load: {e}")
            traceback.print_exc()
            ml_models['config'] = None
    else:
        print(f"❌ Fichier config non trouvé à {CONFIG_PATH}")
        ml_models['config'] = None
    print(f"  >> Valeur ml_models['config'] après tentative: {type(ml_models.get('config'))}")

    # --- Charger Modèle Keras ---
    if tf_available:
        print(f"Chargement Modèle Keras depuis : {MODEL_PATH}")
        if MODEL_PATH.exists():
            try:
                print(f"  Tentative tf.keras.models.load_model('{MODEL_PATH}')...")
                 # Utiliser une variable locale temporaire
                model_local = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Modèle Keras chargé.")

                # Essayer une prédiction test ici peut révéler des erreurs internes
                config_for_test = ml_models.get('config') # Utiliser la config chargée
                if config_for_test and 'maxlen' in config_for_test:
                    try:
                        print("   Tentative de prédiction test...")
                        sample_input = np.zeros((1, config_for_test['maxlen']), dtype=np.int32)
                        _ = model_local.predict(sample_input, verbose=0)
                        print("   (Prédiction test après chargement réussie)")
                        ml_models['model'] = model_local # Mettre dans le dict seulement si tout OK
                    except Exception as pred_e:
                        print(f"   ❌ ERREUR lors de la prédiction test: {pred_e}")
                        traceback.print_exc()
                        ml_models['model'] = None # Échec si prédiction test échoue
                else:
                    print("   Skipping prediction test (config/maxlen non disponible).")
                    ml_models['model'] = model_local # Mettre dans le dict même sans test

            except Exception as e:
                print(f"❌ ERREUR Modèle Keras load_model: {e}")
                traceback.print_exc() # Imprimer la trace complète de l'erreur TF/Keras
                ml_models['model'] = None
        else:
            print(f"❌ Fichier modèle Keras non trouvé à {MODEL_PATH}")
            ml_models['model'] = None
    else:
         print("Skipping Keras model loading (TensorFlow not found).")
         ml_models['model'] = None # S'assurer qu'il est None
    print(f"  >> Valeur ml_models['model'] après tentative: {type(ml_models.get('model'))}")

    # Vérification finale
    if not all(ml_models.values()):
         # Ne pas afficher comme une erreur fatale, mais comme un avertissement
         print("\n⚠️ AVERTISSEMENT: Chargement incomplet d'un ou plusieurs composants ML lors du démarrage.")
    else:
         print("\n✅ Tous les composants ML semblent chargés via load_models.")

# ==================================================
# Événements Startup/Shutdown FastAPI
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    print("--- Exécution de l'événement startup (lifespan) ---")
    load_models() # Appelle notre fonction de chargement
    yield # L'application tourne ici
    # Code exécuté à l'arrêt (optionnel)
    print("--- Exécution de l'événement shutdown (lifespan) ---")
    ml_models.clear() # Libérer la mémoire

# ==================================================
# Création de l'Application FastAPI AVEC Lifespan
# ==================================================
app = FastAPI(
    title="Air Paradis Sentiment Analysis API",
    description="API pour prédire le sentiment (positif/négatif) d'un tweet, basée sur un modèle LSTM et FastText.",
    version="1.0.2", # Incrémenter version (debug logs)
    lifespan=lifespan # Associer la fonction lifespan
)

# ==================================================
# Modèles Pydantic (inchangés)
# ==================================================
class TweetInput(BaseModel):
    tweet_text: str

class SentimentOutput(BaseModel):
    sentiment_label: str
    sentiment_score: float

# ==================================================
# Fonctions de Prétraitement (modifiées pour utiliser ml_models)
# ==================================================
def clean_minimal(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text: str) -> str:
    # Récupérer depuis le dict global à chaque appel (au cas où il serait None)
    nlp_spacy_local = ml_models.get('nlp_spacy')
    if nlp_spacy_local is None:
        print("⚠️ spaCy non disponible pour lemmatisation.")
        return text # Retourne le texte non lemmatisé
    try:
        doc = nlp_spacy_local(text)
        lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
        return " ".join(lemmas)
    except Exception as e:
        print(f"⚠️ Erreur pendant lemmatisation: {e}")
        return text # Retourne texte original en cas d'erreur

def preprocess_for_model(text: str) -> np.ndarray:
    start_time = time.time()
    # Récupérer depuis le dict global à chaque appel
    tokenizer_local = ml_models.get('tokenizer')
    config_local = ml_models.get('config')

    if not tokenizer_local:
        # Log spécifique si tokenizer manque
        print("❌ ERREUR Prétraitement: Tokenizer non disponible dans ml_models.")
        raise ValueError("Composant Tokenizer non disponible.")
    if not config_local:
         print("❌ ERREUR Prétraitement: Config non disponible dans ml_models.")
         raise ValueError("Composant Config non disponible.")

    maxlen = config_local.get('maxlen')
    if maxlen is None: raise ValueError("'maxlen' manquant dans config.")

    cleaned_text = clean_minimal(text)
    lemmatized_text = lemmatize_text(cleaned_text) # Gère l'absence de spaCy

    # Tokenisation Keras
    try:
        sequence = tokenizer_local.texts_to_sequences([lemmatized_text])
    except Exception as e:
        print(f"❌ ERREUR lors de texts_to_sequences: {e}")
        raise ValueError("Erreur de tokenisation.")

    # Padding
    try:
        padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='pre', truncating='post')
    except Exception as e:
        print(f"❌ ERREUR lors de pad_sequences: {e}")
        raise ValueError("Erreur de padding.")

    end_time = time.time()
    # Éviter de logguer à chaque requête en production, mais utile pour debug
    # print(f"Prétraitement effectué en {(end_time - start_time)*1000:.2f} ms")
    return padded_sequence

# ==================================================
# Endpoints de l'API (modifiés pour utiliser ml_models)
# ==================================================
@app.get("/", tags=["Health Check"])
async def read_root():
    # print("Requête GET reçue sur /") # Peut devenir bruyant
    # Vérifier l'état via le dictionnaire global
    return {"message": "API d'analyse de sentiments Air Paradis fonctionnelle!",
            "model_loaded": ml_models.get('model') is not None,
            "tokenizer_loaded": ml_models.get('tokenizer') is not None,
            "config_loaded": ml_models.get('config') is not None,
            "spacy_loaded": ml_models.get('nlp_spacy') is not None}

@app.post("/predict", response_model=SentimentOutput, tags=["Prediction"])
async def predict_sentiment(tweet_input: TweetInput):
    # print(f"\nRequête POST reçue sur /predict avec texte: '{tweet_input.tweet_text[:50]}...'") # Bruyant
    # Récupérer le modèle chargé depuis le dict global
    model_local = ml_models.get('model')

    # Vérifier si le modèle est bien chargé au moment de la requête
    if model_local is None:
        print("❌ ERREUR /predict: Modèle non disponible dans ml_models au moment de la requête.")
        raise HTTPException(status_code=503, detail="Modèle de prédiction non disponible (échec chargement au démarrage?).")

    # Prétraitement
    try:
        processed_input = preprocess_for_model(tweet_input.tweet_text)
        # print(f"Shape après prétraitement: {processed_input.shape}") # Debug
    except ValueError as e: # Erreurs attendues de preprocess_for_model
        print(f"❌ ERREUR Prétraitement pour requête: {e}")
        raise HTTPException(status_code=422, # Unprocessable Entity pour erreur de prétraitement
                            detail=f"Erreur lors du prétraitement du texte: {e}")
    except Exception as e: # Autres erreurs
        print(f"❌ ERREUR Inattendue Prétraitement pour requête: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur interne inattendue lors du prétraitement.")

    # Prédiction
    try:
        start_pred_time = time.time()
        # Utiliser le modèle récupéré au début de la fonction
        prediction_score = model_local.predict(processed_input, verbose=0)[0][0]
        end_pred_time = time.time()
        # print(f"Prédiction effectuée en {(end_pred_time - start_pred_time)*1000:.2f} ms - Score: {prediction_score:.4f}") # Debug

        # Formatage Réponse
        sentiment_label = "positif" if prediction_score > 0.5 else "negatif"
        return SentimentOutput(
            sentiment_label=sentiment_label,
            sentiment_score=float(prediction_score)
        )
    except Exception as e:
        print(f"❌ ERREUR lors de la prédiction modèle pour requête: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'exécution du modèle.")

# --- Lancement Uvicorn (Commenté) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Lancer avec: uvicorn main:app --reload --host 0.0.0.0 --port 8000")