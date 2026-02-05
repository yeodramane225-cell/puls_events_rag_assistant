import pandas as pd
import faiss
import numpy as np
import subprocess
import json
import pickle
import time
import os

# Dossier de sortie
os.makedirs("data/vectorstore", exist_ok=True)


# ---------------------------------------------------------
# 1. Fonction pour générer un embedding via Ollama
# ---------------------------------------------------------
def embed_text(text):
    """
    Appelle Ollama pour générer un embedding avec nomic-embed-text.
    Ollama renvoie parfois plusieurs lignes → on prend la dernière ligne JSON.
    Le modèle renvoie directement une LISTE d'embeddings (pas un dict).
    """
    result = subprocess.run(
        ["ollama", "run", "nomic-embed-text"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Décodage + séparation des lignes
    output_raw = result.stdout.decode("utf-8").strip().split("\n")

    # La dernière ligne contient le JSON valide
    last_line = output_raw[-1]

    try:
        data = json.loads(last_line)

        # IMPORTANT : nomic-embed-text renvoie directement la liste des embeddings
        return np.array(data, dtype="float32")

    except Exception as e:
        print("\n--- ERREUR JSON OLLAMA ---")
        print("Dernière ligne reçue :", last_line)
        print("--------------------------\n")
        raise e


# ---------------------------------------------------------
# 2. Vectorisation par batch
# ---------------------------------------------------------
def embed_batch(texts):
    vectors = []
    for t in texts:
        vectors.append(embed_text(t))
        time.sleep(0.05)  # évite de saturer Ollama
    return np.vstack(vectors)


# ---------------------------------------------------------
# 3. Script principal
# ---------------------------------------------------------
def build_faiss_index():
    print("Chargement des chunks...")
    df = pd.read_csv("data/processed/events_chunks.csv")

    assert "text_chunk" in df.columns, "La colonne 'text_chunk' est manquante !"

    texts = df["text_chunk"].tolist()
    print(f"Nombre de chunks à vectoriser : {len(texts)}")

    batch_size = 64
    all_vectors = []

    print("\nVectorisation en cours...\n")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Batch {i//batch_size + 1} / {len(texts)//batch_size + 1}")
        vectors = embed_batch(batch)
        all_vectors.append(vectors)

    embeddings = np.vstack(all_vectors)

    print("\nCréation de l’index FAISS...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Sauvegarde de l’index...")
    faiss.write_index(index, "data/vectorstore/faiss_index.bin")

    print("Sauvegarde des métadonnées...")
    df.to_pickle("data/vectorstore/metadata.pkl")

    print("\n🎉 Index FAISS généré avec succès !")


if __name__ == "__main__":
    build_faiss_index()
