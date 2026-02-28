## Puls Events — Assistant RAG
Assistant conversationnel basé sur un pipeline RAG (Retrieval‑Augmented Generation) permettant d’interroger en langage naturel les événements publics issus d’OpenAgenda.

## Objectifs du projet
Construire un pipeline complet : ingestion → nettoyage → enrichissement → chunking → embeddings → FAISS → génération.

Mettre en place un système RAG local basé sur les modèles Mistral.

Créer un assistant capable de répondre à des questions sur les événements publics.

Structurer un projet Python reproductible, clair et documenté.

Fournir un chatbot interactif pour tester le moteur RAG.

## Structure du projet
text
puls_events_rag_assistant/
│
├── chat.py                     # Interface console du chatbot (boucle interactive)
├── README.md                   # Documentation complète du projet
├── requirements.txt            # Dépendances Python
│
├── data/
│   ├── raw/                    # Données brutes OpenAgenda (JSONL)
│   ├── processed/              # Données nettoyées + chunks
│   └── vectorstore/
│       ├── faiss_index.bin     # Index vectoriel FAISS
│       ├── metadata.pkl        # Métadonnées (pickle)
│       └── metadata.json       # Métadonnées (JSON)
│
├── notebooks/
│   └── exploration.ipynb       # Analyse exploratoire
│
├── src/
│   ├── ingestion/
│   │   └── load_jsonl.py       # Lecture en streaming du JSONL massif
│   │
│   ├── preprocessing/
│   │   └── clean_and_chunk.py  # Nettoyage, enrichissement et découpage en chunks
│   │
│   ├── vectorstore/
│   │   └── build_faiss_index.py # Génération des embeddings + construction FAISS
│   │
│   └── rag/
│       ├── rag_query.py        # Pipeline RAG complet (retriever + génération)
│       └── query_mistral.py    # Appel au modèle Mistral (embeddings + génération)
│
└── tests/                      # Tests fonctionnels (optionnels)
## Phase 1 — Préparation de l’environnement
Création de l’environnement virtuel (python -m venv venv_rag).

Activation de l’environnement.

Installation des dépendances (pip install -r requirements.txt).

Création du fichier .env contenant la clé API Mistral.

Résultat : environnement propre, isolé et compatible avec le pipeline RAG.

## Phase 2 — Ingestion des données OpenAgenda
Lecture du fichier JSONL en streaming pour éviter de charger plusieurs millions de lignes en mémoire.

Validation de la structure des événements.

Préparation des données pour le pré‑processing.

Script utilisé : src/ingestion/load_jsonl.py  
Résultat : données brutes prêtes à être nettoyées.

## Phase 3 — Pré‑processing, enrichissement et chunking
Nettoyage des champs (titres, descriptions, dates, lieux…).

Normalisation des textes.

Enrichissement des descriptions.

Découpage en chunks textuels cohérents.

Sauvegarde dans data/processed/.

Script utilisé : src/preprocessing/clean_and_chunk.py  
Résultat : données propres, enrichies et adaptées à la vectorisation.

## Phase 4 — Construction de la base vectorielle FAISS
Chargement des chunks pré‑traités.

Génération des embeddings via Mistral.

Construction de l’index FAISS (faiss_index.bin).

Sauvegarde des métadonnées (metadata.pkl, metadata.json).

Script utilisé : src/vectorstore/build_faiss_index.py  
Résultat : index FAISS performant et prêt pour la recherche vectorielle.

## Phase 5 — Intégration du moteur RAG avec LangChain
Chargement de l’index FAISS et des métadonnées.

Initialisation du modèle Mistral (embeddings + génération).

Création d’un retriever basé sur FAISS.

Construction du pipeline RAG complet :

embedding de la question,

recherche vectorielle,

récupération des chunks pertinents,

injection dans un prompt structuré,

génération de la réponse.

Scripts utilisés : rag_query.py, query_mistral.py  
Résultat : moteur RAG capable de répondre de manière contextualisée.

## Phase 6 — Vérifications intermédiaires du pipeline RAG
Vérification de la cohérence des métadonnées.

Vérification de la pertinence des résultats FAISS.

Vérification de la qualité des réponses générées.

Validation du pipeline complet avant intégration du chatbot.

Résultat : pipeline RAG validé techniquement.

## Phase 7 — Création du ChatBot interactif
Développement du script chat.py.

Mise en place d’une boucle interactive dans le terminal.

Gestion des commandes /help et exit.

Passage automatique de chaque question dans le pipeline RAG.

Formatage clair et lisible des réponses.

Exemple testé :  
« quels événement a lieu ce weekend à paris en 2025 ? »

Résultat : interface simple, fonctionnelle et intuitive.

## Phase 8 — Tests fonctionnels du RAG
Test réel avec une question complexe.

Vérification de la compréhension de la date, de la ville et du contexte.

Vérification de la recherche vectorielle FAISS.

Vérification de la cohérence des réponses générées.

Résultat : réponses cohérentes, contextualisées et fiables.

## Phase 9 — Finalisation et validation du projet
Vérification du bon fonctionnement du pipeline complet.

Validation de la cohérence FAISS → RAG → Chatbot.