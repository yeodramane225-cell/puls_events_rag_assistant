"""
Tests unitaires pour vérifier l'existence et la validité
des chunks générés lors du préprocessing.
"""

import pytest

# Ce test dépend de fichiers locaux non présents dans GitLab CI.
# On l'ignore automatiquement dans le pipeline CI.
pytest.skip("Test ignoré en CI car dépend de fichiers locaux.", allow_module_level=True)

import pandas as pd
import os


def test_chunks_exist():
    """Vérifie que le fichier de chunks a bien été généré."""
    assert os.path.exists("data/processed/events_chunks.csv"), \
        "Le fichier events_chunks.csv est introuvable."


def test_chunks_not_empty():
    """Vérifie que le fichier contient des données valides."""
    df = pd.read_csv("data/processed/events_chunks.csv")

    # Le fichier ne doit pas être vide
    assert len(df) > 0, "Le fichier des chunks est vide."

    # La colonne 'text_chunk' doit exister
    assert "text_chunk" in df.columns, "La colonne 'text_chunk' est manquante."

    # Les chunks doivent contenir du texte significatif
    mean_length = df["text_chunk"].astype(str).str.len().mean()
    assert mean_length > 10, "Les chunks semblent trop courts ou mal générés."

