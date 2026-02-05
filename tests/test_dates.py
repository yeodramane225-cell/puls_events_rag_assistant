"""
Tests unitaires pour vérifier que le filtrage des dates
ne conserve que les événements de moins d'un an.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.ingestion.load_parquet import filter_events


def test_filter_events_date():
    # Dates de référence
    today = datetime.today()
    too_old = today - timedelta(days=400)   # doit être filtré
    recent = today - timedelta(days=10)     # doit être conservé

    # Jeu de données simulé
    df = pd.DataFrame({
        "date": [too_old, recent],
        "location": ["Paris", "Paris"]
    })

    # Application du filtre
    filtered = filter_events(df, "Paris")

    # Vérifications
    assert len(filtered) == 1, "Seul l'événement récent doit être conservé"
    assert filtered.iloc[0]["date"] == recent, "La date conservée doit être la plus récente"
