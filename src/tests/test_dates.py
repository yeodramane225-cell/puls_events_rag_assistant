"""
Tests unitaires simples pour vérifier le filtrage des dates.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.ingestion.load_parquet import filter_events


def test_filter_events_date():
    today = datetime.today()
    old_date = today - timedelta(days=400)
    recent_date = today - timedelta(days=10)

    df = pd.DataFrame({
        "date": [old_date, recent_date],
        "location": ["Paris", "Paris"]
    })

    filtered = filter_events(df, "Paris")

    assert len(filtered) == 1
    assert filtered.iloc[0]["date"] == recent_date
