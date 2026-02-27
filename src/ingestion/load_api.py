import requests
import json
import os
from datetime import datetime, timedelta
from dateutil import parser

BASE_URL = "https://public.opendatasoft.com/api/records/1.0/search/"


def fetch_events(city: str, limit=5000, year=None):

    today = datetime.today().date()
    one_year_ago = today - timedelta(days=365)

    # Si l'utilisateur veut filtrer par année précise
    if year:
        start_date = f"{year}-01-01"
        end_date = f"{year+1}-01-01"
        where_clause = (
            f'firstdate_begin >= "{start_date}" AND firstdate_begin < "{end_date}"'
        )
    else:
        # Mode par défaut : 1 an d’historique
        start_date = one_year_ago.strftime("%Y-%m-%d")
        where_clause = f'firstdate_begin >= "{start_date}"'

    params = {
        "dataset": "evenements-publics-openagenda",
        "rows": limit,
        "where": where_clause,
        "refine.location_city": city
    }

    print("Paramètres envoyés :", params)

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    results = data.get("records", [])

    # Filtrage Python (sécurité)
    filtered = []
    for r in results:
        fields = r.get("fields", {})
        start_raw = fields.get("firstdate_begin")
        if not start_raw:
            continue

        try:
            dt = parser.parse(start_raw).replace(tzinfo=None)
        except:
            continue

        # Si filtrage par année
        if year:
            if dt >= datetime(year, 1, 1) and dt < datetime(year+1, 1, 1):
                filtered.append(fields)
        else:
            # Mode historique 1 an
            if dt >= datetime(2025, 1, 1):
                filtered.append(fields)

    os.makedirs("data/raw", exist_ok=True)
    output_path = f"data/raw/events_{city}_{start_date}_to_future.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"✔ {len(filtered)} événements sauvegardés dans {output_path}")
    return output_path


def filter_events(df, city: str):
    today = datetime.today()
    one_year_ago = today - timedelta(days=365)

    df = df[df["location"] == city]
    df = df[df["date"] >= one_year_ago]

    return df


if __name__ == "__main__":
    fetch_events(city="Paris", year=2025)
