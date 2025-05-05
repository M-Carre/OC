# api/tests/test_main.py

import pytest
from fastapi.testclient import TestClient
import os
import sys
from pathlib import Path

# --- AJOUT IMPORTANT : S'assurer que le dossier 'api' est dans le PYTHONPATH ---
# Ceci permet aux tests de trouver le module 'main' même s'ils sont exécutés
# depuis le dossier parent (P7/) par pytest.
# On remonte d'un niveau ('tests' -> 'api') puis on ajoute ce chemin.
api_dir = Path(__file__).resolve().parent.parent # Chemin vers le dossier api/
sys.path.insert(0, str(api_dir.parent)) # Ajouter le dossier P7/ au path
# -------------------------------------------------------------------------------

# Importer l'application FastAPI depuis api.main
# Attention: cela exécutera le code de chargement au début de main.py !
try:
    from api.main import app, preprocess_for_model, clean_minimal, lemmatize_text
    print("Import de 'app' depuis api.main réussi.")
    # Importer aussi les fonctions de prétraitement si on veut les tester séparément
except ImportError as e:
    print(f"ERREUR: Impossible d'importer 'app' depuis api.main. Vérifiez le chemin et PYTHONPATH. Erreur: {e}")
    # On ne peut pas continuer sans l'app
    raise SystemExit("Arrêt des tests - Impossible d'importer l'application FastAPI.")


# Créer une instance du client de test pour notre application FastAPI
# Ceci ne lance PAS de serveur, mais permet d'envoyer des requêtes HTTP simulées.
client = TestClient(app)

# ==========================================
# Tests pour les Fonctions de Prétraitement
# ==========================================
# On teste chaque étape individuellement si possible

def test_clean_minimal():
    """Teste la fonction de nettoyage de base."""
    assert clean_minimal("Hello @user check http://example.com #awesome & stuff") == "Hello check awesome & stuff"
    assert clean_minimal("  Multiple   spaces  ") == "Multiple spaces"
    assert clean_minimal("Already clean text.") == "Already clean text."
    assert clean_minimal("") == ""
    assert clean_minimal(None) == "" # Gestion du cas None

# Test pour la lemmatisation (si spaCy est chargé)
# On utilise @pytest.mark.skipif pour sauter le test si spaCy n'a pas pu être chargé dans main.py
# Pour cela, on doit importer nlp_spacy depuis main
try:
    from api.main import nlp_spacy as spacy_global # Importer la variable globale
    spacy_loaded_in_main = spacy_global is not None
except ImportError:
    spacy_loaded_in_main = False

@pytest.mark.skipif(not spacy_loaded_in_main, reason="spaCy n'a pas été chargé dans main.py")
def test_lemmatize_text():
    """Teste la fonction de lemmatisation (nécessite spaCy)."""
    assert lemmatize_text("running dogs played better") == "run dog play well" # Note: better -> well
    assert lemmatize_text("The cats are sleeping") == "cat sleep"
    assert lemmatize_text("No change here.") == "change"
    assert lemmatize_text("") == ""

def test_preprocess_for_model_shape():
    """Teste si la fonction de prétraitement complète retourne la bonne shape."""
    # Note: Ceci suppose que le tokenizer et la config sont bien chargés dans main.py
    # Idéalement, on pourrait "mocker" ces dépendances, mais pour commencer, on teste l'intégration.
    try:
        from api.main import config as config_global
        expected_maxlen = config_global.get('maxlen') if config_global else None

        if expected_maxlen is None:
             pytest.skip("Config (maxlen) non chargée dans main.py, impossible de vérifier la shape.")

        # Tester avec un texte simple
        processed = preprocess_for_model("This is a test tweet.")
        assert processed.shape == (1, expected_maxlen) # Doit être (1, 13) dans ton cas
        # Tester avec un texte vide (devrait retourner des paddings ?)
        processed_empty = preprocess_for_model("")
        assert processed_empty.shape == (1, expected_maxlen)

    except ValueError as e:
        # Si une ValueError est levée (ex: tokenizer non chargé), le test échoue, ce qui est bien.
        pytest.fail(f"preprocess_for_model a levé une ValueError inattendue: {e}")
    except NameError as e:
        # Si config_global n'existe pas
         pytest.skip(f"Impossible d'importer la config depuis main.py: {e}")


# ==========================================
# Tests pour l'Endpoint /predict
# ==========================================

def test_predict_endpoint_success():
    """Teste une prédiction réussie sur l'endpoint /predict."""
    # Utilise le TestClient pour envoyer une requête POST simulée
    response = client.post(
        "/predict",
        json={"tweet_text": "This is a wonderful day, I feel great!"} # Exemple positif
    )
    # Vérifier le code de statut HTTP
    assert response.status_code == 200
    # Vérifier le contenu de la réponse JSON
    data = response.json()
    assert "sentiment_label" in data
    assert "sentiment_score" in data
    assert data["sentiment_label"] in ["positif", "negatif"]
    assert isinstance(data["sentiment_score"], float)
    # On pourrait ajouter une assertion plus précise sur le label attendu si on connaît le modèle
    # assert data["sentiment_label"] == "positif" # ATTENTION: dépend fortement du modèle

def test_predict_endpoint_negative():
    """Teste une prédiction pour un texte négatif."""
    response = client.post(
        "/predict",
        json={"tweet_text": "This is horrible, I had a very bad experience."} # Exemple négatif
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_label"] in ["positif", "negatif"]
    # assert data["sentiment_label"] == "negatif" # ATTENTION: dépend du modèle

def test_predict_endpoint_empty_text():
    """Teste l'endpoint avec une chaîne vide."""
    response = client.post(
        "/predict",
        json={"tweet_text": ""}
    )
    # Le prétraitement devrait gérer ça et le modèle prédire quelque chose
    assert response.status_code == 200
    data = response.json()
    assert "sentiment_label" in data
    assert "sentiment_score" in data

def test_predict_endpoint_invalid_input_format():
    """Teste l'endpoint avec un format JSON incorrect (champ manquant)."""
    response = client.post(
        "/predict",
        json={"wrong_field": "some text"} # Manque tweet_text
    )
    # FastAPI devrait retourner une erreur 422 Unprocessable Entity
    assert response.status_code == 422

def test_predict_endpoint_no_json():
    """Teste l'endpoint sans envoyer de corps JSON."""
    response = client.post("/predict")
    # Devrait aussi être une erreur 422 car le corps est attendu
    assert response.status_code == 422

# ==========================================
# Test pour l'Endpoint Racine /
# ==========================================

def test_read_root():
    """Teste l'endpoint racine /."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model_loaded" in data
    assert "tokenizer_loaded" in data
    assert "config_loaded" in data
    assert "spacy_loaded" in data
    # On peut vérifier les valeurs booléennes si on est sûr de l'état attendu
    assert data["model_loaded"] is True # Suppose que le chargement a réussi
    assert data["tokenizer_loaded"] is True
    assert data["config_loaded"] is True
    assert data["spacy_loaded"] is True