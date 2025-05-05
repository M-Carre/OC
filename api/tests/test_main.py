# api/tests/test_main.py

import pytest
from fastapi.testclient import TestClient
import os
import sys
from pathlib import Path
import numpy as np # Importé pour vérifier le type dans un test potentiel

# --- AJOUT IMPORTANT : S'assurer que le dossier 'api' est dans le PYTHONPATH ---
api_dir = Path(__file__).resolve().parent.parent # Chemin vers le dossier api/
sys.path.insert(0, str(api_dir.parent)) # Ajouter le dossier P7/ (parent de api/) au path
# -------------------------------------------------------------------------------

# Importer l'application FastAPI depuis api.main
# Cela déclenchera l'événement lifespan et le chargement des modèles
try:
    from api.main import app, preprocess_for_model, clean_minimal, lemmatize_text, ml_resources
    print("Import de 'app' et fonctions depuis api.main réussi.")
except ImportError as e:
    print(f"ERREUR: Impossible d'importer depuis api.main. Vérifiez PYTHONPATH. Erreur: {e}")
    raise SystemExit("Arrêt des tests - Impossible d'importer l'application FastAPI.")
except Exception as startup_e:
    # Attraper d'éventuelles erreurs PENDANT l'exécution du code de démarrage (lifespan/load_models)
    print(f"ERREUR pendant l'initialisation de api.main (lifespan/load_models probable): {startup_e}")
    import traceback
    traceback.print_exc()
    # On peut décider de s'arrêter ou de continuer avec des tests qui échoueront probablement
    # Pour le CI/CD, il vaut mieux s'arrêter si l'app ne peut pas démarrer.
    raise SystemExit("Arrêt des tests - Erreur critique lors du démarrage de l'API.")


# Créer une instance du client de test
client = TestClient(app)

# ==========================================
# Tests pour les Fonctions de Prétraitement
# ==========================================

def test_clean_minimal():
    """Teste la fonction de nettoyage de base."""
    assert clean_minimal("Hello @user check http://example.com #awesome & stuff") == "Hello check awesome & stuff" # Corrigé
    assert clean_minimal("  Multiple   spaces  ") == "Multiple spaces"
    assert clean_minimal("Already clean text.") == "Already clean text."
    assert clean_minimal("") == ""
    assert clean_minimal(None) == ""

# Test pour la lemmatisation (si spaCy est chargé)
spacy_loaded_in_main = ml_resources.get('nlp_spacy') is not None

@pytest.mark.skipif(not spacy_loaded_in_main, reason="spaCy n'a pas été chargé via lifespan dans main.py")
def test_lemmatize_text():
    """Teste la fonction de lemmatisation (nécessite spaCy)."""
    assert lemmatize_text("running dogs played better") == "run dog play well"
    assert lemmatize_text("The cats are sleeping") == "cat sleep"
    assert lemmatize_text("No change here.") == "change"
    assert lemmatize_text("") == ""

# Test pour preprocess_for_model (Adapté)
# On ne peut plus importer 'config' directement. On teste via l'endpoint ou en vérifiant
# que la fonction ne lève PAS d'erreur si les ressources sont chargées.
def test_preprocess_for_model_no_error_on_loaded_resources():
    """
    Teste que preprocess_for_model s'exécute sans ValueError
    si les ressources (tokenizer, config) sont chargées.
    Ce test suppose que le lifespan a réussi à les charger.
    """
    if not ml_resources.get('tokenizer') or not ml_resources.get('config'):
        pytest.skip("Tokenizer ou Config non chargés lors du startup, skip test.")

    try:
        # Exécuter avec un texte simple
        processed = preprocess_for_model("This is a test tweet.")
        # Vérifier la shape basée sur la config chargée (si possible)
        expected_maxlen = ml_resources['config'].get('maxlen')
        assert isinstance(processed, np.ndarray)
        if expected_maxlen:
            assert processed.shape == (1, expected_maxlen) # Doit être (1, 13)

        # Exécuter avec un texte vide
        processed_empty = preprocess_for_model("")
        assert isinstance(processed_empty, np.ndarray)
        if expected_maxlen:
             assert processed_empty.shape == (1, expected_maxlen)

    except ValueError as e:
        pytest.fail(f"preprocess_for_model a levé une ValueError inattendue alors que les ressources semblaient chargées: {e}")


# ==========================================
# Tests pour l'Endpoint /predict (Adaptés pour TFLite)
# ==========================================

# Helper pour vérifier si les ressources sont chargées avant de lancer les tests predict
# Ceci évite des erreurs 503 répétitives si le chargement a échoué au démarrage.
RESOURCES_LOADED = all(ml_resources.values())

@pytest.mark.skipif(not RESOURCES_LOADED, reason="Ressources ML non chargées, skip tests /predict.")
def test_predict_endpoint_success():
    """Teste une prédiction réussie sur l'endpoint /predict."""
    response = client.post("/predict", json={"tweet_text": "This is a wonderful day, I feel great!"})
    assert response.status_code == 200 # Doit être 200 maintenant
    data = response.json()
    assert "sentiment_label" in data
    assert "sentiment_score" in data
    assert data["sentiment_label"] in ["positif", "negatif"]
    assert isinstance(data["sentiment_score"], float)
    # assert data["sentiment_label"] == "positif" # Dépend du modèle TFLite

@pytest.mark.skipif(not RESOURCES_LOADED, reason="Ressources ML non chargées, skip tests /predict.")
def test_predict_endpoint_negative():
    """Teste une prédiction pour un texte négatif."""
    response = client.post("/predict", json={"tweet_text": "This is horrible, I had a very bad experience."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_label"] in ["positif", "negatif"]
    # assert data["sentiment_label"] == "negatif" # Dépend du modèle TFLite

@pytest.mark.skipif(not RESOURCES_LOADED, reason="Ressources ML non chargées, skip tests /predict.")
def test_predict_endpoint_empty_text():
    """Teste l'endpoint avec une chaîne vide."""
    response = client.post("/predict", json={"tweet_text": ""})
    assert response.status_code == 200 # Devrait toujours fonctionner
    data = response.json()
    assert "sentiment_label" in data
    assert "sentiment_score" in data

def test_predict_endpoint_invalid_input_format():
    """Teste l'endpoint avec un format JSON incorrect (champ manquant)."""
    response = client.post("/predict", json={"wrong_field": "some text"})
    assert response.status_code == 422 # Inchangé

def test_predict_endpoint_no_json():
    """Teste l'endpoint sans envoyer de corps JSON."""
    response = client.post("/predict")
    assert response.status_code == 422 # Inchangé

# ==========================================
# Test pour l'Endpoint Racine / (Adapté)
# ==========================================

def test_read_root():
    """Teste l'endpoint racine /."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    # Vérifier la présence des clés, mais pas forcément leur valeur booléenne exacte
    # car le chargement peut avoir échoué lors de l'init du TestClient
    assert "interpreter_loaded" in data # Clé modifiée
    assert "tokenizer_loaded" in data
    assert "config_loaded" in data
    assert "spacy_loaded" in data
    # Optionnel: Vérifier le type si on veut être plus strict
    assert isinstance(data["interpreter_loaded"], bool)
    assert isinstance(data["tokenizer_loaded"], bool)
    assert isinstance(data["config_loaded"], bool)
    assert isinstance(data["spacy_loaded"], bool)