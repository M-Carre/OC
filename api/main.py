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
# Mettre les imports TF/SpaCy dans des try-except
try:
    import tensorflow as tf
    # Attention: Keras est nécessaire pour pad_sequences, même si on utilise TFLite pour l'inférence
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tf_available = True
except ImportError:
    print("⚠️ WARNING: TensorFlow (ou Keras) non trouvé. L'API ne pourra pas fonctionner.")
    tf_available = False
    # Créer des objets factices pour éviter les NameErrors
    class InterpreterMock:
        def allocate_tensors(self): pass
        def set_tensor(self, *args, **kwargs): pass
        def invoke(self): pass
        def get_tensor(self, *args, **kwargs): return [[0.5]] # Retourne score neutre
        def get_input_details(self): return [{'index': 0, 'dtype': 'int32'}] # Simule détails
        def get_output_details(self): return [{'index': 0}]
    interpreter = InterpreterMock()
    # pad_sequences factice
    def pad_sequences(*args, **kwargs): raise RuntimeError("Keras non disponible pour pad_sequences")

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
# --- MODIFIÉ ---
MODEL_PATH = API_DIR / "model.tflite" # Utiliser le fichier TFLite
# -------------
SPACY_MODEL = "en_core_web_sm"

# Dictionnaire pour contenir les ressources chargées
ml_resources = { # Renommé pour plus de clarté
    "tokenizer": None,
    "config": None,
    "nlp_spacy": None,
    "interpreter": None # Stocke l'interpréteur TFLite
}

# ==================================================
# Fonction de Chargement (appelée au démarrage)
# ==================================================
def load_ml_resources(): # Renommé pour plus de clarté
    """Charge tous les modèles et dépendances nécessaires."""
    print("\n--- Démarrage de la fonction load_ml_resources ---")
    global ml_resources # Indiquer qu'on modifie la variable globale

    # --- Charger spaCy ---
    if spacy_available:
        try:
            print(f"Chargement spaCy '{SPACY_MODEL}'...")
            nlp_spacy_local = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
            print("✅ spaCy chargé.")
            ml_resources['nlp_spacy'] = nlp_spacy_local
        except Exception as e:
            print(f"❌ ERREUR spaCy load_model: {e}")
            traceback.print_exc()
            ml_resources['nlp_spacy'] = None
    else:
        print("Skipping spaCy loading (library not found).")
        ml_resources['nlp_spacy'] = None
    print(f"  >> Valeur ml_resources['nlp_spacy'] après tentative: {type(ml_resources.get('nlp_spacy'))}")

    # --- Charger Tokenizer ---
    print(f"Chargement Tokenizer depuis : {TOKENIZER_PATH}")
    if TOKENIZER_PATH.exists():
        try:
            with open(TOKENIZER_PATH, 'rb') as handle:
                tokenizer_local = pickle.load(handle)
            print("✅ Tokenizer chargé.")
            ml_resources['tokenizer'] = tokenizer_local
        except Exception as e:
            print(f"❌ ERREUR Tokenizer pickle.load: {e}")
            traceback.print_exc()
            ml_resources['tokenizer'] = None
    else:
        print(f"❌ Fichier Tokenizer non trouvé à {TOKENIZER_PATH}")
        ml_resources['tokenizer'] = None
    print(f"  >> Valeur ml_resources['tokenizer'] après tentative: {type(ml_resources.get('tokenizer'))}")

    # --- Charger Config ---
    print(f"Chargement Config depuis : {CONFIG_PATH}")
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                config_local = json.load(f)
            if 'maxlen' not in config_local:
                print("❌ ERREUR: 'maxlen' manque dans config.")
                ml_resources['config'] = None
            else:
                 print(f"✅ Config chargée (maxlen={config_local.get('maxlen')}).")
                 ml_resources['config'] = config_local
        except Exception as e:
            print(f"❌ ERREUR Config json.load: {e}")
            traceback.print_exc()
            ml_resources['config'] = None
    else:
        print(f"❌ Fichier config non trouvé à {CONFIG_PATH}")
        ml_resources['config'] = None
    print(f"  >> Valeur ml_resources['config'] après tentative: {type(ml_resources.get('config'))}")

    # --- Charger Modèle TFLite --- # MODIFIÉ
    if tf_available:
        print(f"Chargement Modèle TFLite depuis : {MODEL_PATH}")
        if MODEL_PATH.exists():
            try:
                print(f"  Tentative tf.lite.Interpreter('{MODEL_PATH}')...")
                # Charger l'interpréteur TFLite
                interpreter_local = tf.lite.Interpreter(model_path=str(MODEL_PATH))
                print("  Allocation des tenseurs...")
                interpreter_local.allocate_tensors() # Étape nécessaire pour TFLite
                print("✅ Interpréteur TFLite chargé et tenseurs alloués.")
                ml_resources['interpreter'] = interpreter_local # Stocker l'interpréteur

                # Optionnel: Imprimer les détails des entrées/sorties pour info
                input_details = interpreter_local.get_input_details()
                output_details = interpreter_local.get_output_details()
                print(f"   Input details: {input_details}")
                print(f"   Output details: {output_details}")

                # Note: Pas de prédiction test ici, car on n'a pas encore de données prétraitées

            except Exception as e:
                print(f"❌ ERREUR Modèle TFLite: {e}")
                traceback.print_exc()
                ml_resources['interpreter'] = None
        else:
            print(f"❌ Fichier modèle TFLite non trouvé à {MODEL_PATH}")
            ml_resources['interpreter'] = None
    else:
         print("Skipping TFLite model loading (TensorFlow not found).")
         ml_resources['interpreter'] = None
    print(f"  >> Valeur ml_resources['interpreter'] après tentative: {type(ml_resources.get('interpreter'))}")

    # Vérification finale (vérifie 'interpreter' maintenant)
    if not all([ml_resources.get('tokenizer'), ml_resources.get('config'), ml_resources.get('nlp_spacy'), ml_resources.get('interpreter')]):
         print("\n⚠️ AVERTISSEMENT: Chargement incomplet d'un ou plusieurs composants ML lors du démarrage.")
    else:
         print("\n✅ Tous les composants ML (y compris Interpréteur TFLite) semblent chargés via load_ml_resources.")


# ==================================================
# Événements Startup/Shutdown FastAPI
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Exécution de l'événement startup (lifespan) ---")
    load_ml_resources() # Appelle notre fonction de chargement
    yield
    print("--- Exécution de l'événement shutdown (lifespan) ---")
    ml_resources.clear()

# ==================================================
# Création de l'Application FastAPI AVEC Lifespan
# ==================================================
app = FastAPI(
    title="Air Paradis Sentiment Analysis API (TFLite)", # Nom mis à jour
    description="API pour prédire le sentiment (positif/négatif) d'un tweet, basée sur un modèle LSTM optimisé (TFLite) et FastText.",
    version="1.1.0", # Incrémenter version (passage TFLite)
    lifespan=lifespan
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
# Fonctions de Prétraitement (modifiées pour utiliser ml_resources)
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
    nlp_spacy_local = ml_resources.get('nlp_spacy')
    if nlp_spacy_local is None:
        # print("⚠️ spaCy non disponible pour lemmatisation.") # Peut être trop bruyant
        return text
    try:
        doc = nlp_spacy_local(text)
        lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
        return " ".join(lemmas)
    except Exception as e:
        print(f"⚠️ Erreur pendant lemmatisation: {e}")
        return text

def preprocess_for_model(text: str) -> np.ndarray:
    # start_time = time.time() # Moins utile maintenant
    tokenizer_local = ml_resources.get('tokenizer')
    config_local = ml_resources.get('config')

    if not tokenizer_local: raise ValueError("Composant Tokenizer non disponible.")
    if not config_local: raise ValueError("Composant Config non disponible.")

    maxlen = config_local.get('maxlen')
    if maxlen is None: raise ValueError("'maxlen' manquant dans config.")

    cleaned_text = clean_minimal(text)
    lemmatized_text = lemmatize_text(cleaned_text)

    try:
        sequence = tokenizer_local.texts_to_sequences([lemmatized_text])
    except Exception as e:
        print(f"❌ ERREUR lors de texts_to_sequences: {e}")
        raise ValueError("Erreur de tokenisation.")

    # Utiliser pad_sequences importé de Keras (même si inférence TFLite)
    try:
        padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='pre', truncating='post')
    except Exception as e:
        print(f"❌ ERREUR lors de pad_sequences: {e}")
        raise ValueError("Erreur de padding.")

    # end_time = time.time()
    # print(f"Prétraitement effectué en {(end_time - start_time)*1000:.2f} ms") # Trop bruyant
    return padded_sequence

# ==================================================
# Endpoints de l'API (modifiés pour utiliser ml_resources['interpreter'])
# ==================================================
@app.get("/", tags=["Health Check"])
async def read_root():
    # Vérifier l'état via le dictionnaire global
    return {"message": "API d'analyse de sentiments Air Paradis (TFLite) fonctionnelle!",
            "interpreter_loaded": ml_resources.get('interpreter') is not None, # Vérifie l'interpréteur
            "tokenizer_loaded": ml_resources.get('tokenizer') is not None,
            "config_loaded": ml_resources.get('config') is not None,
            "spacy_loaded": ml_resources.get('nlp_spacy') is not None}

@app.post("/predict", response_model=SentimentOutput, tags=["Prediction"])
async def predict_sentiment(tweet_input: TweetInput):
    # Récupérer l'interpréteur chargé
    interpreter = ml_resources.get('interpreter')

    if interpreter is None:
        print("❌ ERREUR /predict: Interpréteur TFLite non disponible.")
        raise HTTPException(status_code=503, detail="Modèle de prédiction (TFLite) non disponible.")

    # Prétraitement
    try:
        processed_input = preprocess_for_model(tweet_input.tweet_text)

        # --- Adaptation pour TFLite ---
        input_details = interpreter.get_input_details()
        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']

        # Assurer le bon dtype pour l'entrée TFLite
        # Pour un embedding layer, c'est souvent int32. Le convertisseur TF Select Ops garde souvent le type original.
        if processed_input.dtype != input_dtype:
             print(f"⚠️ Attention: Conversion dtype de l'input de {processed_input.dtype} vers {input_dtype}")
             processed_input = processed_input.astype(input_dtype)
        # -----------------------------

        # print(f"Shape input TFLite: {processed_input.shape}, Dtype: {processed_input.dtype}") # Debug

    except ValueError as e:
        print(f"❌ ERREUR Prétraitement pour requête: {e}")
        raise HTTPException(status_code=422, detail=f"Erreur lors du prétraitement du texte: {e}")
    except Exception as e:
        print(f"❌ ERREUR Inattendue Prétraitement: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur interne inattendue lors du prétraitement.")

    # Prédiction TFLite
    try:
        start_pred_time = time.time()

        # Obtenir les détails des tenseurs d'entrée/sortie (peut être fait une seule fois au démarrage si constant)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Définir le tenseur d'entrée
        interpreter.set_tensor(input_details[0]['index'], processed_input)

        # Exécuter l'inférence
        interpreter.invoke()

        # Obtenir le résultat depuis le tenseur de sortie
        prediction_score_array = interpreter.get_tensor(output_details[0]['index'])
        prediction_score = prediction_score_array[0][0] # Extraire le score scalaire

        end_pred_time = time.time()
        # print(f"Prédiction TFLite effectuée en {(end_pred_time - start_pred_time)*1000:.2f} ms - Score: {prediction_score:.4f}") # Debug

        # Formatage Réponse
        sentiment_label = "positif" if prediction_score > 0.5 else "negatif"
        return SentimentOutput(
            sentiment_label=sentiment_label,
            sentiment_score=float(prediction_score)
        )
    except Exception as e:
        print(f"❌ ERREUR lors de l'inférence TFLite: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'exécution du modèle TFLite.")

# --- Lancement Uvicorn (Commenté) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Lancer avec: uvicorn main:app --reload --host 0.0.0.0 --port 8000")