import faiss
import pandas as pd
import numpy as np
import subprocess
import json


# ---------------------------------------------------------
# 1. Embedding via Ollama
# ---------------------------------------------------------
def embed_text(text):
    """
    Génère un embedding avec nomic-embed-text.
    Ollama renvoie directement une LISTE de floats.
    """
    result = subprocess.run(
        ["ollama", "run", "nomic-embed-text"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE
    )

    # Ollama peut renvoyer plusieurs lignes → on prend la dernière
    lines = result.stdout.decode("utf-8").strip().split("\n")
    last_line = lines[-1]

    data = json.loads(last_line)  # data = liste de floats
    return np.array(data, dtype="float32")


# ---------------------------------------------------------
# 2. Appel modèle génératif (Mistral local)
# ---------------------------------------------------------
def query_ollama(prompt):
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
# 4. Pipeline RAG complet
# ---------------------------------------------------------
def rag_query(question, k=5):
    # Embedding de la question
    q_emb = embed_text(question).reshape(1, -1)

    # Recherche FAISS
    distances, indices = index.search(q_emb, k)

    # Récupération des chunks pertinents
    retrieved = "\n".join(df_chunks.iloc[i]["text_chunk"] for i in indices[0])

    # Construction du prompt
    prompt = f"""
Tu es un assistant spécialisé dans les événements publics en France.
Voici des informations pertinentes extraites de la base OpenAgenda :

{retrieved}

Question : {question}

Réponds de manière claire, concise et utile.
"""

    # Génération de la réponse
    return query_ollama(prompt)


# ---------------------------------------------------------
# 5. Exemple d'utilisation
# ---------------------------------------------------------
if __name__ == "__main__":
    print(rag_query("Quels événements ont lieu à Paris ce week-end ?"))
