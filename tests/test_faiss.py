"""
Tests unitaires pour vérifier l'existence de l'index FAISS
et la capacité du moteur à effectuer une recherche vectorielle.
"""

import pytest
import faiss
import numpy as np
import os
import json

# Chemins réels selon ta structure actuelle
INDEX_PATH = "data/vectorstore/faiss_index.bin"
METADATA_PATH = "data/vectorstore/metadata.json"


def test_faiss_index_exists():
    """Vérifie que les fichiers FAISS et metadata existent."""
    assert os.path.exists(INDEX_PATH), "Le fichier faiss_index.bin est introuvable."
    assert os.path.exists(METADATA_PATH), "Le fichier metadata.json est introuvable."


def test_faiss_search():
    """Vérifie que l'index FAISS peut effectuer une recherche."""
    # Chargement de l'index
    index = faiss.read_index(INDEX_PATH)

    # Génération d'un vecteur aléatoire de la bonne dimension
    vec = np.random.rand(1, index.d).astype("float32")

    # Recherche des 3 voisins les plus proches
    distances, indices = index.search(vec, 3)

    # Vérifications
    assert indices.shape == (1, 3), "La recherche FAISS ne renvoie pas 3 résultats."
    assert not np.isnan(distances).any(), "Les distances contiennent des valeurs NaN."
    assert (indices >= 0).any(), "Aucun index valide n'a été retourné."
