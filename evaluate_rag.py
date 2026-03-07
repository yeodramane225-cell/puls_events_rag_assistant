import json
import time
import faiss
import numpy as np
import pandas as pd
from mistralai import Mistral
from dotenv import load_dotenv
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("ERREUR : La clé API Mistral n'est pas définie dans le fichier .env")

client = Mistral(api_key=MISTRAL_API_KEY)

# Dossiers réels confirmés par tes captures
INDEX_PATH = "data/vectorstore/faiss_index.bin"
METADATA_PATH = "data/vectorstore/metadata.json"
CHUNKS_PATH = "data/processed/events_chunks.csv"

# ---------------------------
# METRIQUES MANUELLES
# ---------------------------
def precision_score_manual(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    return tp / (tp + fp + 1e-9)

def recall_score_manual(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp / (tp + fn + 1e-9)

def f1_score_manual(y_true, y_pred):
    p = precision_score_manual(y_true, y_pred)
    r = recall_score_manual(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-9)

# ---------------------------
# CHARGEMENT DES DONNÉES
# ---------------------------
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunks():
    return pd.read_csv(CHUNKS_PATH)

# ---------------------------
# DETECTION AUTOMATIQUE DE LA CLE DU CHUNK
# ---------------------------
def get_chunk_id(meta):
    possible_keys = ["id", "chunk_id", "chunk_index", "chunk", "index"]
    for key in possible_keys:
        if key in meta:
            return str(meta[key])
    raise KeyError(f"Aucune clé identifiant un chunk trouvée dans : {meta}")

# ---------------------------
# EMBEDDING VIA MISTRAL
# ---------------------------
def embed_text(text):
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ---------------------------
# CHARGEMENT DE FAISS
# ---------------------------
def load_faiss_index():
    return faiss.read_index(INDEX_PATH)

# ---------------------------
# EVALUATION DU RETRIEVER
# ---------------------------
def evaluate_retriever(test_set, index, metadata, k=5):
    y_true = []
    y_pred = []

    for sample in test_set:
        question = sample["question"]
        expected_chunks = [str(c) for c in sample["relevant_chunks"]]

        q_emb = embed_text(question).reshape(1, -1)
        distances, indices = index.search(q_emb, k)

        retrieved_chunks = [str(get_chunk_id(metadata[i])) for i in indices[0]]

        all_chunks = sorted(list(set(expected_chunks + retrieved_chunks)))
        true_vec = [1 if c in expected_chunks else 0 for c in all_chunks]
        pred_vec = [1 if c in retrieved_chunks else 0 for c in all_chunks]

        y_true.extend(true_vec)
        y_pred.extend(pred_vec)

    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)
    f1 = f1_score_manual(y_true, y_pred)

    return precision, recall, f1

# ---------------------------
# MESURE DES TEMPS DE TRAITEMENT
# ---------------------------
def measure_times():
    times = {}

    t0 = time.time()
    metadata = load_metadata()
    chunks = load_chunks()
    times["chargement_donnees"] = time.time() - t0

    t0 = time.time()
    index = load_faiss_index()
    times["chargement_faiss"] = time.time() - t0

    t0 = time.time()
    _ = embed_text("Quels événements ont lieu en 2025 ?")
    times["embedding_question"] = time.time() - t0

    q_emb = embed_text("Quels événements ont lieu en 2025 ?").reshape(1, -1)
    t0 = time.time()
    _ = index.search(q_emb, 5)
    times["recherche_faiss"] = time.time() - t0

    return times

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("Chargement des données…")
    metadata = load_metadata()
    index = load_faiss_index()

    test_set = [
        {
            "question": "Quels événements ont lieu en 2025 à Paris ?",
            "relevant_chunks": ["42", "87"]
        },
        {
            "question": "Quels événements ont lieu en 2023 à Lyon ?",
            "relevant_chunks": ["12", "13"]
        }
    ]

    print("\nÉvaluation du retriever…")
    precision, recall, f1 = evaluate_retriever(test_set, index, metadata)
    print(f"Précision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1-score  : {f1:.3f}")

    print("\nMesure des temps de traitement…")
    times = measure_times()
    for step, t in times.items():
        print(f"{step} : {t:.4f} sec")
