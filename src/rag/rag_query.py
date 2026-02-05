"""
Pipeline RAG du projet Puls-Events.

Ce module :
- génère les embeddings des questions via Ollama,
- interroge l’index FAISS pour retrouver les chunks pertinents,
- applique un filtrage par année si la question en contient une,
- reconstruit les embeddings directement depuis FAISS,
- génère une réponse finale via le modèle Mistral.

Auteur : Yeo
Projet : Puls-Events RAG Assistant
"""

import faiss
import pandas as pd
import numpy as np
import subprocess
import json
import re

from src.rag.dataset_info import event_years


# ---------------------------------------------------------
# 1. Embedding via Ollama
# ---------------------------------------------------------
def embed_text(text):
    """Génère un embedding pour un texte donné via le modèle nomic-embed-text."""
    result = subprocess.run(
        ["ollama", "run", "nomic-embed-text"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE
    )

    lines = result.stdout.decode("utf-8").strip().split("\n")
    last_line = lines[-1]

    data = json.loads(last_line)
    return np.array(data, dtype="float32")


# ---------------------------------------------------------
# 2. Appel modèle génératif (Mistral local)
# ---------------------------------------------------------
def query_ollama(prompt):
    """Envoie un prompt au modèle Mistral via Ollama et retourne la réponse brute."""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")


# ---------------------------------------------------------
# 3. Charger FAISS + metadata
# ---------------------------------------------------------
index = faiss.read_index("data/vectorstore/faiss_index.bin")
df_chunks = pd.read_pickle("data/vectorstore/metadata.pkl")


# ---------------------------------------------------------
# 4. Pipeline RAG complet avec filtrage par année
# ---------------------------------------------------------
def rag_query(question, k=5):
    """
    Exécute une requête RAG complète.

    Étapes :
    - extraction éventuelle d’une année dans la question,
    - filtrage des événements correspondants,
    - reconstruction des embeddings via FAISS,
    - recherche sémantique (FAISS),
    - génération d’une réponse via Mistral.

    Args:
        question (str): question utilisateur en langage naturel.
        k (int): nombre de chunks à récupérer.

    Returns:
        str: réponse générée par le modèle.
    """

    # 1. Détecter une année dans la question
    match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    year_filter = match.group(0) if match else None

    # 2. Si une année est trouvée → filtrer les événements
    if year_filter:
        valid_event_ids = [
            eid for eid, years in event_years.items()
            if year_filter in years
        ]

        if not valid_event_ids:
            return f"Aucun événement trouvé pour l'année {year_filter}."

        filtered_df = df_chunks[df_chunks["event_id"].isin(valid_event_ids)]

        if len(filtered_df) > 0:
            q_emb = embed_text(question).reshape(1, -1)

            # Construire un index FAISS temporaire filtré
            sub_index = faiss.IndexFlatL2(index.d)

            # ⚠️ On reconstruit les embeddings directement depuis FAISS
            emb_list = []
            for original_idx in filtered_df.index:
                emb = index.reconstruct(original_idx)
                emb_list.append(emb)

            sub_index.add(np.vstack(emb_list))

            distances, indices = sub_index.search(q_emb, min(k, len(filtered_df)))

            retrieved = "\n".join(
                filtered_df.iloc[i]["text_chunk"] for i in indices[0]
            )

            prompt = f"""
Tu es un assistant spécialisé dans les événements publics en France.
Voici des informations pertinentes extraites de la base OpenAgenda :

{retrieved}

Question : {question}

Réponds de manière claire, concise et utile.
"""
            return query_ollama(prompt)

    # 3. Fallback : pipeline RAG normal
    q_emb = embed_text(question).reshape(1, -1)
    distances, indices = index.search(q_emb, k)

    retrieved = "\n".join(df_chunks.iloc[i]["text_chunk"] for i in indices[0])

    prompt = f"""
Tu es un assistant spécialisé dans les événements publics en France.
Voici des informations pertinentes extraites de la base OpenAgenda :

{retrieved}

Question : {question}

Réponds de manière claire, concise et utile.
"""

    return query_ollama(prompt)
