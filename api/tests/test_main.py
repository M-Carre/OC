# api/tests/test_main.py (Version Baseline)

import pytest
from fastapi.testclient import TestClient
import os
import sys
from pathlib import Path
import numpy as np

# --- PYTHONPATH ---
api_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(api_dir.parent))
# -----------------

# Importer l'app (déclenche lifespan et chargement pipeline)
try:
    from api.main import app, clean_minimal, ml_resources
    print("Import api.main réussi.")
except Exception as startup_e:
    print(f"ERREUR pendant import/startup api.main: {startup_e}")
    import traceback
    traceback.print_exc()
    raise SystemExit("Arrêt tests - Erreur critique API.")

client = TestClient(app)

# ==========================================
# Tests Fonctions Prétraitement (Seulement clean_minimal)
# ==========================================
def test_clean_minimal():
    assert clean_minimal("Hello @user check http://example.com #awesome & stuff") == "Hello check awesome & stuff"
    assert clean_minimal("  Multiple   spaces  ") == "Multiple spaces"
    assert clean_minimal("") == ""
    assert clean_minimal(None) == ""

# Plus besoin de test_lemmatize_text
# Plus besoin de test_preprocess_for_model_shape

# ==========================================
# Tests Endpoint /predict
# ==========================================
PIPELINE_LOADED = ml_resources.get('pipeline') is not None

@pytest.mark.skipif(not PIPELINE_LOADED, reason="Pipeline baseline non chargé, skip tests /predict.")
def test_predict_endpoint_success():
    response = client.post("/predict", json={"tweet_text": "This is a wonderful day, I feel great!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment_label" in data
    assert "sentiment_score" in data
    assert data["sentiment_label"] in ["positif", "negatif"]
    assert isinstance(data["sentiment_score"], float)
    # assert data["sentiment_label"] == "positif" # Dépend du modèle baseline

@pytest.mark.skipif(not PIPELINE_LOADED, reason="Pipeline baseline non chargé, skip tests /predict.")
def test_predict_endpoint_negative():
    response = client.post("/predict", json={"tweet_text": "This is horrible, I had a very bad experience."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_label"] in ["positif", "negatif"]
    # assert data["sentiment_label"] == "negatif" # Dépend du modèle baseline

@pytest.mark.skipif(not PIPELINE_LOADED, reason="Pipeline baseline non chargé, skip tests /predict.")
def test_predict_endpoint_empty_text():
    response = client.post("/predict", json={"tweet_text": ""})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment_label" in data
    assert "sentiment_score" in data

def test_predict_endpoint_invalid_input_format():
    response = client.post("/predict", json={"wrong_field": "some text"})
    assert response.status_code == 422

def test_predict_endpoint_no_json():
    response = client.post("/predict")
    assert response.status_code == 422

# ==========================================
# Test Endpoint Racine / (Adapté)
# ==========================================
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "pipeline_loaded" in data # Vérifie la clé pour le pipeline
    assert isinstance(data["pipeline_loaded"], bool)