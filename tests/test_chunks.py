"""
Tests unitaires pour vérifier l'existence et la validité
des chunks générés lors du préprocessing.
"""

import pytest
import json
import os

CHUNKS_PATH = "data/vectorstore/metadata.json"


def test_chunks_file_exists():
    assert os.path.exists(CHUNKS_PATH), "Le fichier metadata.json est introuvable."


def test_chunks_not_empty():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 0, "Le fichier metadata.json est vide."
    assert "chunk" in data[0], "La clé 'chunk' est manquante dans les metadata."

    mean_length = sum(len(item["chunk"]) for item in data) / len(data)
    assert mean_length > 10, "Les chunks semblent trop courts ou mal générés."
