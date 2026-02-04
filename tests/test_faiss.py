import faiss
import numpy as np
import os

def test_faiss_index_exists():
    assert os.path.exists("data/processed/faiss_index.bin")

def test_faiss_search():
    index = faiss.read_index("data/processed/faiss_index.bin")
    vec = np.random.rand(1, index.d).astype("float32")
    distances, indices = index.search(vec, 3)
    assert len(indices[0]) == 3

