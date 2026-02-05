## Puls Events — Assistant RAG
Assistant conversationnel basé sur un pipeline RAG (Retrieval-Augmented Generation) permettant d’interroger les événements publics OpenAgenda en langage naturel.

## Objectifs du projet
Construire un pipeline complet de traitement de données (ingestion → nettoyage → chunking → embeddings → FAISS → génération).

Mettre en place un système RAG local basé sur Ollama (Mistral + Nomic Embed Text).

Créer un assistant capable de répondre à des questions sur les événements publics.

Implémenter des tests unitaires garantissant la fiabilité du pipeline.

Structurer un projet Python reproductible et documenté.

## Structure du projet

puls_events_rag_assistant/
│
├── chat.py                     # Interface console du chatbot
├── config.yaml                 # Configuration globale
├── README.md                   # Documentation du projet
├── requirements.txt            # Dépendances Python
│
├── data/
│   ├── raw/                    # Données brutes OpenAgenda
│   ├── processed/              # Données nettoyées + chunks
│   └── vectorstore/
│       ├── faiss_index.bin     # Index vectoriel FAISS
│       └── metadata.pkl        # Chunks + metadata
│
├── notebooks/
│   └── exploration.ipynb       # Analyse exploratoire
│
├── src/
│   ├── ingestion/
│   │   └── load_parquet.py     # Chargement des données
│   ├── preprocessing/
│   │   └── clean_and_chunk.py  # Nettoyage + chunking
│   ├── rag/
│   │   ├── rag_chain.py        # Prompting + LLM
│   │   ├── rag_query.py        # Pipeline RAG complet
│   │   └── dataset_info.py     # Extraction des années
│   └── vectorization/
│       └── build_faiss_index.py # Embeddings + FAISS
│
└── tests/
    ├── test_dates.py
    ├── test_chunks.py
    ├── test_embeddings.py
    └── test_faiss.py

##  Installation et environnement
1. Cloner le projet

git clone <URL_DU_REPO>
cd puls_events_rag_assistant
2. Créer un environnement virtuel

python -m venv venv_rag
3. Activer l’environnement (Windows)

venv_rag\Scripts\activate
4. Installer les dépendances

pip install -r requirements.txt
5. Installer les modèles Ollama

ollama pull mistral
ollama pull nomic-embed-text

## Pipeline de traitement des données
1. Nettoyage + Chunking

python src/preprocessing/clean_and_chunk.py
Ce script :

nettoie les descriptions d’événements

découpe en chunks cohérents

sauvegarde events_chunks.csv

2. Génération des embeddings + FAISS

python src/vectorization/build_faiss_index.py
Ce script :

charge les chunks

génère les embeddings via Ollama

construit l’index FAISS

sauvegarde faiss_index.bin + metadata.pkl

**Lancer le chatbot**

python chat.py
Exemple :

Vous : Quels événements ont lieu le 17 mai ?
Assistant : ...
Le chatbot utilise :

FAISS pour retrouver les chunks pertinents

Mistral pour générer la réponse

OpenAgenda comme source unique

**Tests unitaires**
Les tests se trouvent dans tests/ :

Fichier	Rôle
test_dates.py	Vérifie la cohérence des dates
test_chunks.py	Vérifie la qualité du chunking
test_embeddings.py	Vérifie la taille des embeddings
test_faiss.py	Vérifie que FAISS répond correctement
Lancer les tests :


**pytest**
**Choix techniques**
RAG : permet de répondre à des questions sur des données non structurées.

FAISS : index vectoriel performant pour la recherche sémantique.

Ollama : exécution locale, rapide, sans dépendance cloud.

Mistral : modèle léger et performant pour la génération.

Nomic Embed Text : embeddings adaptés aux données textuelles courtes.

**Limitations actuelles**
Pas de filtrage géographique avancé (ville, département).

Pas de gestion des doublons OpenAgenda.

Pas de mise à jour automatique des données.

Pas encore de test unitaire pour rag_query().

**Améliorations possibles**
Ajouter un filtrage par ville / date / catégorie.

Ajouter un test d’intégration complet.

Ajouter une interface web (Streamlit).

Ajouter un scheduler pour mettre à jour les données.