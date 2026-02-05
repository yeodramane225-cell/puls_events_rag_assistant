"""
Pipeline RAG minimal du projet Puls-Events.

Ce module :
- génère les embeddings des questions via Ollama (nomic-embed-text),
- interroge l’index FAISS pour retrouver les chunks pertinents,
- construit un prompt contextuel,
- génère une réponse via le modèle Mistral.

Auteur : Yeo
Projet : Puls-Events RAG Assistant
"""

import faiss
import pandas as pd
import numpy as np
import subprocess
import json

def embed_text(text):
    result = subprocess.run(
        ["ollama", "run", "nomic-embed-text"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    output = json.loads(result.stdout.decode("utf-8"))
    return np.array(output["embedding"], dtype="float32")

def query_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

# Charger FAISS + chunks
index = faiss.read_index("data/processed/faiss_index.bin")
df_chunks = pd.read_csv("data/processed/events_chunks.csv")

def rag_query(question):
    """
    Exécute une requête RAG simple :
    - embed la question,
    - recherche les chunks les plus proches via FAISS,
    - construit un prompt,
    - génère une réponse via Mistral.

    Args:
        question (str): question utilisateur.

    Returns:
        str: réponse générée par le modèle.
    """
    q_emb = embed_text(question).reshape(1, -1)
    distances, indices = index.search(q_emb, 5)

    retrieved = "\n".join(df_chunks.iloc[i]["chunk"] for i in indices[0])

    prompt = f"""
Tu es un assistant spécialisé dans les événements publics.
Voici des informations pertinentes :

{retrieved}

Question : {question}

Réponds de manière claire et concise.
"""

    return query_ollama(prompt)

# Exemple
if __name__ == "__main__":
    print(rag_query("Quels événements ont lieu à Paris ce week-end ?"))
