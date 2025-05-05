# Projet Air Paradis - Analyse de Sentiments (P7 OpenClassrooms)

Ce dépôt contient le code pour le Projet 7 du parcours Ingénieur IA d'OpenClassrooms,
visant à créer un prototype d'analyse de sentiments pour Air Paradis.

## Structure du Dépôt

*   `Notebook_de_modélisation.ipynb` : Notebook principal contenant l'EDA, le prétraitement, l'entraînement et l'évaluation des modèles.
*   `api/` : Contient le code de l'API FastAPI pour le déploiement du modèle sélectionné.
    *   `main.py`: Le code source de l'application FastAPI.
    *   `requirements.txt`: Les dépendances Python pour l'API et les tests.
    *   `tests/`: Les tests unitaires pour l'API (utilisant pytest).
    *   `Copie de *.pkl`, `*.json`, `*.keras`: Les artefacts nécessaires. 
*   `.gitignore`: Fichier spécifiant les fichiers à ignorer par Git.
*   `README.md`: Ce fichier.

## Instructions (Basiques)

*   **Installation locale :**
    ```bash
    git clone https://github.com/M-Carre/OC.git
    cd OC/api
    python -m venv venv
    source venv/bin/activate  # ou .\venv\Scripts\activate sous Windows
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
*   **Lancer l'API localement :** Depuis le dossier `OC/`, exécutez `uvicorn api.main:app --reload --port 8000`
*   **Exécuter les tests :** Depuis le dossier `OC/`, exécutez `pytest api/tests/`

