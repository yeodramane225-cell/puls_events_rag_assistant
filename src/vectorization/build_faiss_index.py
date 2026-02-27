import json
import faiss
import numpy as np
import os
import requests

os.makedirs("data/vectorstore", exist_ok=True)

MISTRAL_API_KEY = "koYSEltxtO2OhW0twIdOJTzbZDxYUEzt"
EMBEDDING_MODEL = "mistral-embed"

JSONL_PATH = "data/raw/evenements-publics-openagenda.json"
INDEX_PATH = "data/vectorstore/faiss_index.bin"
META_PATH = "data/vectorstore/metadata.json"


def load_events_streaming(jsonl_path):
    """
    Lit un fichier JSONL massif ligne par ligne sans charger tout en mémoire.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunk_text(text, max_tokens=300):
    """
    Découpe un texte long en chunks de taille raisonnable.
    """
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]


def embed_batch_mistral(texts):
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": EMBEDDING_MODEL, "input": texts}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()["data"]
    return np.vstack([np.array(item["embedding"], dtype="float32") for item in data])


def build_faiss_index():
    print("Lecture du JSONL...")
    events = list(load_events_streaming(JSONL_PATH))
    print(f"{len(events)} événements chargés")

    print("Chunking...")
    chunks = []
    metadata = []

    for event in events:
        description = event.get("description", "")
        event_chunks = chunk_text(description)

        for c in event_chunks:
            chunks.append(c)
            metadata.append({
                "description": description,
                "chunk": c
            })

    print(f"{len(chunks)} chunks générés")

    print("Vectorisation...")
    batch_size = 64
    vectors = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors.append(embed_batch_mistral(batch))

    embeddings = np.vstack(vectors)

    print("Construction FAISS...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Sauvegarde...")
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("🎉 Index FAISS généré avec succès !")


if __name__ == "__main__":
    build_faiss_index()
