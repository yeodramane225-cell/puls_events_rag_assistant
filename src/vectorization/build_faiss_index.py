import pandas as pd
import faiss
import numpy as np
import json
import os
import requests

os.makedirs("data/vectorstore", exist_ok=True)

MISTRAL_API_KEY = "koYSEltxtO2OhW0twIdOJTzbZDxYUEzt"
EMBEDDING_MODEL = "mistral-embed"

CSV_PATH = "data/processed/events_chunks.csv"
INDEX_PATH = "data/vectorstore/faiss_index.bin"
META_PATH = "data/vectorstore/metadata.json"


def embed_batch_mistral(texts):
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()["data"]
    vectors = [np.array(item["embedding"], dtype="float32") for item in data]
    return np.vstack(vectors)


def build_faiss_index():
    print("Chargement des chunks...")
    df = pd.read_csv(CSV_PATH)

    if "chunk" not in df.columns:
        raise ValueError("La colonne 'chunk' est manquante")

    texts = df["chunk"].tolist()
    print(f"Nombre de chunks à vectoriser : {len(texts)}")

    batch_size = 64
    all_vectors = []

    print("\nVectorisation en cours...\n")

    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_id = i // batch_size + 1
        print(f"Batch {batch_id} / {total_batches}")

        vectors = embed_batch_mistral(batch)
        all_vectors.append(vectors)

    embeddings = np.vstack(all_vectors)

    print("\nCréation de l’index FAISS...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Sauvegarde de l’index...")
    faiss.write_index(index, INDEX_PATH)

    print("Sauvegarde des métadonnées...")
    metadata = df.to_dict(orient="records")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n🎉 Index FAISS généré avec succès !")


if __name__ == "__main__":
    build_faiss_index()
