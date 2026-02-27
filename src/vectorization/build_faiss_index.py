import json
import faiss
import numpy as np
import os
import requests

os.makedirs("data/vectorstore", exist_ok=True)

MISTRAL_API_KEY = "koYSEltxtO2OhW0twIdOJTzbZDxYUEzt"
EMBEDDING_MODEL = "mistral-embed"

JSON_PATH = "data/raw/evenements-publics-openagenda.json"
INDEX_PATH = "data/vectorstore/faiss_index.bin"
META_PATH = "data/vectorstore/metadata.json"


def load_events(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("results", [])


def chunk_text(text, max_tokens=300):
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
    print("Lecture du JSON...")
    events = load_events(JSON_PATH)
    print(f"{len(events)} événements chargés")

    print("Extraction + chunking...")
    chunks = []
    metadata = []

    for event in events:
        fields = event.get("fields", {})

        description = " ".join([
            fields.get("title", ""),
            fields.get("free_text", ""),
            fields.get("longdescription", ""),
            fields.get("shortdescription", ""),
            fields.get("conditions", ""),
            fields.get("location_name", ""),
            fields.get("address", ""),
            fields.get("city", ""),
            fields.get("department", ""),
            fields.get("region", "")
        ]).strip()

        if len(description) < 10:
            continue

        for c in chunk_text(description):
            chunks.append(c)
            metadata.append({"description": description, "chunk": c})

    print(f"{len(chunks)} chunks générés")

    # 🔥 Correction : FAISS ne plante plus jamais
    if not chunks:
        print("⚠️ Aucun chunk généré, création d'un index FAISS vide.")
        dim = 1024  # dimension du modèle mistral-embed
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("Index FAISS vide généré.")
        return

    print("Vectorisation...")
    vectors = []
    batch_size = 64

    for i in range(0, len(chunks), batch_size):
        vectors.append(embed_batch_mistral(chunks[i:i + batch_size]))

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
