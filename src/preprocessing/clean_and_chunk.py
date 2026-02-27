import json
from datetime import datetime
from dateutil import parser
import os
import pandas as pd

# Fichier généré par load_api.py
RAW_PATH = "data/raw/events_Paris_2025-01-01_to_future.json"
OUTPUT_PATH = "data/processed/events_chunks.csv"


def load_events():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_paris(city):
    if not city:
        return False
    return "paris" in city.lower()


def parse_date(value):
    if not value:
        return None
    try:
        dt = parser.parse(value)
        return dt.replace(tzinfo=None)
    except:
        return None


def filter_events(events):
    filtered = []
    for ev in events:
        city = ev.get("location_city")
        if not is_paris(city):
            continue

        start = parse_date(ev.get("firstdate_begin"))
        if not start:
            continue

        # Garder tous les événements depuis le 1er janvier 2025
        if start >= datetime(2025, 1, 1):
            filtered.append(ev)

    return filtered


def chunk_events(events):
    chunks = []
    for idx, ev in enumerate(events):

        # ⚠️ Ajout obligatoire pour le RAG
        event_id = str(idx)

        desc = ev.get("longdescription_fr") or ev.get("description_fr") or ""
        title = ev.get("title_fr") or "Sans titre"

        if not desc:
            continue

        # Découpage en chunks de 500 caractères
        for i in range(0, len(desc), 500):
            chunks.append({
                "event_id": event_id,              # ← indispensable pour le RAG
                "title": title,
                "city": ev.get("location_city"),
                "date_start": ev.get("firstdate_begin"),
                "date_end": ev.get("lastdate_end"),
                "chunk": desc[i:i+500]
            })

    return chunks


if __name__ == "__main__":
    events = load_events()
    print("Événements bruts :", len(events))

    filtered = filter_events(events)
    print("Événements filtrés (Paris + >= 2025-01-01) :", len(filtered))

    chunks = chunk_events(filtered)
    print("Chunks générés :", len(chunks))

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(chunks).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print("✔ Fichier écrit :", OUTPUT_PATH)
