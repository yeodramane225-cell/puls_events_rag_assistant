"""
Pipeline RAG du projet Puls-Events.

Ce module :
- génère les embeddings des questions via Mistral,
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
import json
import re
import requests

from src.rag.dataset_info import event_years

MISTRAL_API_KEY = "koYSEltxtO2OhW0twIdOJTzbZDxYUEzt"
EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "mistral-large-latest"


# ---------------------------------------------------------
# 1. Embedding via Mistral
# ---------------------------------------------------------
def embed_text_mistral(text: str):
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return np.array(data["data"][0]["embedding"], dtype="float32")


# ---------------------------------------------------------
# 2. Appel modèle génératif Mistral
# ---------------------------------------------------------
def query_mistral(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------
# 3. Charger FAISS + metadata
# ---------------------------------------------------------
index = faiss.read_index("data/vectorstore/faiss_index.bin")

with open("data/vectorstore/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

df_chunks = pd.DataFrame(metadata)


# ---------------------------------------------------------
# 4. Pipeline RAG complet avec filtrage par année
# ---------------------------------------------------------
def rag_query(question, k=5):
    """
    Exécute une requête RAG complète.
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
            q_emb = embed_text_mistral(question).reshape(1, -1)

            # Construire un index FAISS temporaire filtré
            sub_index = faiss.IndexFlatL2(index.d)

            emb_list = []
            for original_idx in filtered_df.index:
                emb = index.reconstruct(original_idx)
                emb_list.append(emb)

            sub_index.add(np.vstack(emb_list))

            distances, indices = sub_index.search(q_emb, min(k, len(filtered_df)))

            retrieved = "\n".join(
                filtered_df.iloc[i]["chunk"] for i in indices[0]
            )

            prompt = f"""
Tu es un assistant spécialisé dans les événements publics en France.
Voici des informations pertinentes extraites de la base OpenAgenda :

{retrieved}

Question : {question}

Réponds de manière claire, concise et utile.
"""
            return query_mistral(prompt)

    # 3. Fallback : pipeline RAG normal
    q_emb = embed_text_mistral(question).reshape(1, -1)
    distances, indices = index.search(q_emb, k)

    retrieved = "\n".join(df_chunks.iloc[i]["chunk"] for i in indices[0])

    prompt = f"""
Tu es un assistant spécialisé dans les événements publics en France.
Voici des informations pertinentes extraites de la base OpenAgenda :

{retrieved}

Question : {question}

Réponds de manière claire, concise et utile.
"""

    return query_mistral(prompt)
