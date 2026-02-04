import pandas as pd
import faiss
import numpy as np
import subprocess
import json

def embed_text(text):
    """Appelle Ollama pour générer un embedding avec nomic-embed-text."""
    result = subprocess.run(
        ["ollama", "run", "nomic-embed-text"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    output = json.loads(result.stdout.decode("utf-8"))
    return np.array(output["embedding"], dtype="float32")

# Charger les chunks
df = pd.read_csv("data/processed/events_chunks.csv")

# Générer les embeddings
embeddings = np.vstack(df["chunk"].apply(embed_text).values)

# Construire l’index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Sauvegarder l’index
faiss.write_index(index, "data/processed/faiss_index.bin")

print("Index FAISS généré avec succès.")
