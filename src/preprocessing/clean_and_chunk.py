"""
Nettoyage, filtrage et chunking des événements OpenAgenda
Projet : Puls-Events RAG Assistant
Auteur : Yeo

Ce script :
- nettoie les champs textuels,
- normalise les dates,
- filtre les événements par région et période (< 1 an),
- génère des chunks pour la vectorisation,
- garantit qu'aucune ligne n'est supprimée AVANT filtrage.
"""

import pandas as pd
import re
from datetime import timedelta
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

# Désactiver le warning BeautifulSoup sur les URLs
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


# -----------------------------
# 1. Nettoyage texte
# -----------------------------
def clean_text(text: str) -> str:
    """Nettoie un texte : suppression HTML, espaces multiples, normalisation."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# 2. Préprocessing
# -----------------------------
def preprocess_events(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les colonnes importantes du DataFrame sans supprimer de lignes."""
    df = df.copy()

    # Renommer les colonnes importantes du CSV OpenAgenda
    mapping = {
        "Titre": "title",
        "Description": "description",
        "Description longue": "long_description",
        "Adresse": "address",
        "Nom du lieu": "location_name",
        "Ville": "city",
        "Département": "department",
        "Région": "region",
        "Pays": "country",
        "Mots clés": "keywords",
        "Première date - Début": "start_date",
        "Première date - Fin": "end_date",
    }
    df = df.rename(columns=mapping)

    # Colonnes textuelles à nettoyer
    text_cols = [
        "title", "description", "long_description",
        "address", "location_name", "city",
        "department", "region", "country", "keywords"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Normalisation des dates en UTC
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", utc=True)

    return df


# -----------------------------
# 3. Filtrage géographique + temporel
# -----------------------------
def filter_events(df: pd.DataFrame, region_cible: str) -> pd.DataFrame:
    """Filtre les événements par région et par période (< 1 an)."""
    df = df.copy()

    # Filtrage géographique
    df = df[df["region"] == region_cible]

    # Filtrage temporel : événements < 1 an (UTC-aware)
    un_an = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)

    df = df[
        (df["start_date"] >= un_an) |
        (df["end_date"] >= un_an)
    ]

    return df


# -----------------------------
# 4. Chunking
# -----------------------------
def chunk_text(text: str, max_tokens: int = 300) -> list:
    """Découpe un texte en chunks de taille raisonnable."""
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]


def create_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Génère un DataFrame contenant un chunk par ligne."""
    rows = []

    for _, row in df.iterrows():
        full_text = " ".join([
            row.get("title", ""),
            row.get("description", ""),
            row.get("long_description", ""),
            row.get("address", ""),
            row.get("location_name", ""),
            row.get("city", ""),
            row.get("region", ""),
            row.get("country", ""),
            row.get("keywords", "")
        ])

        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            rows.append({
                "event_id": row.get("Identifiant"),
                "chunk_id": i,
                "text_chunk": chunk
            })

    return pd.DataFrame(rows)


# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    # Charger le CSV brut
    df = pd.read_csv("data/raw/evenements-publics-openagenda.csv", sep=";")

    # Nettoyage
    df_clean = preprocess_events(df)

    # Vérification : aucune ligne supprimée avant filtrage
    assert len(df_clean) == len(df), (
        f"ERREUR : le nettoyage a supprimé des lignes ! "
        f"{len(df_clean)} vs {len(df)}"
    )

    # Filtrage (exemple : Île-de-France)
    REGION = "Île-de-France"
    df_filtered = filter_events(df_clean, REGION)

    # Chunking
    df_chunks = create_chunks(df_filtered)

    # Sauvegarde
    df_chunks.to_csv("data/processed/events_chunks.csv", index=False)

    print(f"Lignes initiales : {len(df)}")
    print(f"Lignes après nettoyage : {len(df_clean)}")
    print(f"Lignes après filtrage : {len(df_filtered)}")
    print(f"Chunks générés : {len(df_chunks)}")

