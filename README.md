# Projet Air Paradis - Analyse de Sentiments (P7 OpenClassrooms) ✈️💬

Ce dépôt contient le code et la documentation pour le Projet 7 du parcours Ingénieur IA d'OpenClassrooms. L'objectif est de créer un prototype d'outil d'analyse de sentiments pour la compagnie aérienne fictive "Air Paradis", afin de l'aider à anticiper les "bad buzz" sur les réseaux sociaux, tout en mettant en œuvre une démarche MLOps complète.

## Objectif du Projet

Développer un prototype fonctionnel d'un produit IA capable de prédire le sentiment (positif ou négatif) associé à un tweet. Ce prototype inclut :
1.  L'exploration et l'entraînement de différents modèles (Classique, Deep Learning Avancé, Transfer Learning).
2.  Le déploiement d'un modèle sélectionné via une API REST sur le Cloud.
3.  Une interface locale pour tester l'API et collecter du feedback utilisateur.
4.  La mise en œuvre d'une démarche MLOps (Versioning, Expérimentations MLflow, CI/CD, Monitoring).

## Structure du Dépôt

*   `Notebook_de_modélisation.ipynb` : Notebook Jupyter principal contenant l'analyse exploratoire des données (EDA), le prétraitement, l'entraînement et l'évaluation comparative des différents modèles (Baseline, LSTM, BiLSTM, BERT FE, USE), ainsi que le suivi des expérimentations avec MLflow.
*   `api/` : Contient le code de l'API FastAPI pour le déploiement. **Note : En raison des contraintes des plans Cloud gratuits, c'est le modèle Baseline (TF-IDF + LogReg) qui est actuellement déployé.**
    *   `main.py`: Le code source de l'application FastAPI.
    *   `requirements.txt`: Les dépendances Python spécifiques à l'API et à ses tests.
    *   `tests/`: Les tests unitaires pour l'API (`pytest`).
    *   `baseline_pipeline.joblib`: Le pipeline Sklearn (TF-IDF + LogReg) sérialisé utilisé par l'API.
*   `feedback_interface/` : Contient l'interface locale (notebook) pour tester l'API et fournir du feedback.
    *   `interface_feedback.ipynb`: Le notebook Jupyter permettant de saisir un tweet, d'appeler l'API, et de signaler les prédictions incorrectes à Azure Application Insights.
    *   `requirements.txt`: Les dépendances Python spécifiques à cette interface locale.
    *   `.env.example`: Un exemple de fichier `.env`.
*   `.github/workflows/` : Contient les workflows GitHub Actions pour l'intégration continue et le déploiement continu (CI/CD).
    *   `deploy.yml`: Workflow qui teste et déploie l'API sur Azure Web App.
*   `.gitignore`: Fichier spécifiant les fichiers et dossiers à ignorer par Git (données, secrets, environnements virtuels, etc.).
*   `README.md`: Ce fichier.
*   `requirements.txt` (Racine) : Dépendances générales pour la partie modélisation du projet (celles utilisées dans le notebook principal).

## Modèle Déployé

L'API actuellement déployée utilise le modèle **Baseline (TF-IDF + Régression Logistique)** entraîné sur du texte lemmatisé.
*   **URL de l'API :** [`https://airparadis-sentiment-api.azurewebsites.net/`](https://airparadis-sentiment-api.azurewebsites.net/)
*   **Documentation (Swagger UI) :** [`https://airparadis-sentiment-api.azurewebsites.net/docs`](https://airparadis-sentiment-api.azurewebsites.net/docs)

## Instructions d'Utilisation

### 1. Partie Modélisation (Notebook Principal)

*   Clonez le dépôt : `git clone https://github.com/M-Carre/OC.git`
*   Naviguez vers le dossier : `cd OC`
*   Créez un environnement virtuel : `python -m venv venv`
*   Activez-le : `source venv/bin/activate` (Linux/macOS) ou `.\venv\Scripts\activate` (Windows)
*   Installez les dépendances de modélisation : `pip install -r requirements.txt` (celui à la racine)
*   Lancez Jupyter Lab ou Notebook : `jupyter lab`
*   Ouvrez et exécutez `Notebook_de_modélisation.ipynb`. (Nécessite le téléchargement initial des données et potentiellement des modèles GloVe/FastText comme indiqué dans le notebook).

### 2. API Déployée

*   L'API est accessible directement via son URL ou sa documentation Swagger UI.
*   Pour envoyer une requête POST à l'endpoint `/predict` depuis du code Python (par exemple) :
    ```python
    import requests
    import json

    api_url = "https://airparadis-sentiment-api.azurewebsites.net/predict"
    tweet = {"tweet_text": "Your tweet here!"}
    headers = {"Content-Type": "application/json"}

    response = requests.post(api_url, data=json.dumps(tweet), headers=headers)

    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    ```

### 3. Interface de Feedback Locale

*   Naviguez vers le sous-dossier : `cd feedback_interface` (depuis la racine `OC/`)
*   Créez un environnement virtuel séparé (recommandé) ou utilisez celui de la racine. S'il est séparé :
    *   `python -m venv venv_feedback`
    *   `source venv_feedback/bin/activate` (ou équivalent Windows)
*   Installez les dépendances spécifiques : `pip install -r requirements.txt` (celui dans `feedback_interface/`)
*   **Important :** Créez un fichier `.env` dans ce dossier (`feedback_interface/.env`) et ajoutez votre chaîne de connexion Application Insights :
    ```plaintext
    APPLICATIONINSIGHTS_CONNECTION_STRING="VOTRE_CHAINE_DE_CONNEXION_ICI"
    ```
*   Lancez Jupyter Lab ou Notebook depuis le dossier `feedback_interface` : `jupyter lab`
*   Ouvrez et exécutez `interface_feedback.ipynb`.

## Démarche MLOps Implémentée

*   **Gestion de Version :** Git & GitHub.
*   **Suivi d'Expérimentations :** MLflow (tracking, artefacts, registry).
*   **CI/CD :** GitHub Actions pour les tests (`pytest`) et le déploiement sur Azure Web App.
*   **Monitoring & Feedback :** Azure Application Insights pour collecter les retours sur les prédictions incorrectes via l'interface locale et déclencher des alertes.

## Auteur

*   **Mathis Carré** - [M-Carre](https://github.com/M-Carre)

---
