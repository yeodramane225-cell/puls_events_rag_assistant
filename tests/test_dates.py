"""
Tests unitaires pour vérifier que le filtrage des dates
ne conserve que les événements de moins d'un an.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.ingestion.load_api import filter_events


def test_filter_events_date():
    today = datetime.today()
    too_old = today - timedelta(days=400)
    recent = today - timedelta(days=10)

    df = pd.DataFrame({
        "date": [too_old, recent],
        "location": ["Paris", "Paris"]
    })

    filtered = filter_events(df, "Paris")

    assert len(filtered) == 1, "Seul l'événement récent doit être conservé"
    assert filtered.iloc[0]["date"] == recent, "La date conservée doit être la plus récente"
