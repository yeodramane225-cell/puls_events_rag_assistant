Structure du projet
Voici l’arborescence réelle de ton projet, basée sur ce que tu m’as montré :


puls_events_rag_assistant/
│
├── chat.py
├── config.yaml
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── evenements-publics-openagenda.csv
│   ├── processed/
│   └── vectors/
│       └── vectorstore/
│           ├── faiss_index.bin
│           └── metadata.pkl
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── ingestion/
│   │   └── load_parquet.py
│   ├── preprocessing/
│   │   └── clean_and_chunk.py
│   ├── rag/
│   │   ├── rag_chain.py
│   │   └── rag_query.py
│   └── vectorization/
│       └── build_faiss_index.py
│
└── tests/
    ├── test_dates.py
    ├── test_chunks.py
    ├── test_embeddings.py
    └── test_faiss.py
🛠️ Installation et environnement
1. Cloner le projet

git clone <URL_DU_REPO>
cd puls_events_rag_assistant
2. Créer un environnement virtuel

python -m venv venv_rag
3. Activer l’environnement
Windows :

venv_rag\Scripts\activate

4. Installer les dépendances
Code
pip install -r requirements.txt
5. Installer les modèles Ollama
Code
ollama pull mistral
ollama pull nomic-embed-text


## Reconstruction de la base vectorielle
Le pipeline complet est découpé en modules :

1. Nettoyage + Chunking

python src/preprocessing/clean_and_chunk.py
2. Génération des embeddings

python src/vectorization/build_faiss_index.py
Ce script :

charge les chunks

génère les embeddings via Ollama

construit l’index FAISS

sauvegarde faiss_index.bin + metadata.pkl

Lancer le chatbot

python chat.py
Exemple :

Code
Vous : Quels événements ont lieu le 17 mai ?
Assistant : ...
Le chatbot utilise :

FAISS pour retrouver les chunks pertinents

Mistral pour générer la réponse

Les données OpenAgenda uniquement

Tests unitaires
Les tests se trouvent dans tests/ :

test_dates.py → vérifie que les événements ont moins d’un an

test_chunks.py → vérifie la cohérence du chunking

test_embeddings.py → vérifie la taille des embeddings

test_faiss.py → vérifie que l’index FAISS répond correctement

Lancer les tests :

pytest


Technologies utilisées
Composant	Rôle
Python 3	Base du projet
Pandas	Manipulation des données
FAISS CPU	Indexation vectorielle
Ollama	Exécution locale des modèles
Mistral	Génération de texte
Nomic Embed Text	Embeddings
PyTest	Tests unitaires
