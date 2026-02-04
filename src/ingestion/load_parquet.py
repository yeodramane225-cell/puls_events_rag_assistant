"""
Module d'ingestion des données OpenAgenda pour le projet Puls-Events RAG Assistant.

Ce script :
- charge le fichier Parquet contenant les événements publics,
- filtre les événements selon une ville donnée,
- filtre les événements de moins d'un an,
- sauvegarde les données filtrées dans data/processed/.

Auteur : Yeo
"""

import pandas as pd
from datetime import datetime, timedelta
import os


def load_events(parquet_path: str) -> pd.DataFrame:
    """
    Charge le fichier Parquet contenant les événements OpenAgenda.

    Args:
        parquet_path (str): Chemin vers le fichier Parquet.

    Returns:
        pd.DataFrame: DataFrame contenant les événements.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Fichier introuvable : {parquet_path}")

    df = pd.read_parquet(parquet_path)
    return df


def filter_events(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Filtre les événements selon :
    - une ville donnée,
    - une période de moins d'un an.

    Args:
        df (pd.DataFrame): DataFrame des événements.
        city (str): Ville à filtrer.

    Returns:
        pd.DataFrame: DataFrame filtrée.
    """
    today = datetime.today()
    one_year_ago = today - timedelta(days=365)

    # Conversion des dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filtre : date >= il y a 1 an
    df = df[df["date"] >= one_year_ago]

    # Filtre : ville
    df = df[df["location"].str.contains(city, case=False, na=False)]

    return df


def save_filtered_events(df: pd.DataFrame, output_path: str):
    """
    Sauvegarde les événements filtrés dans un fichier CSV.

    Args:
        df (pd.DataFrame): DataFrame filtrée.
        output_path (str): Chemin de sortie.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Événements filtrés sauvegardés dans : {output_path}")


if __name__ == "__main__":
    # Chemin vers ton fichier Parquet
    parquet_file = "data/raw/evenements-publics-openagenda.parquet"

    # Charger les données
    df = load_events(parquet_file)

    # Filtrer par ville (exemple : Paris)
    df_filtered = filter_events(df, city="Paris")

    # Sauvegarder
    save_filtered_events(df_filtered, "data/processed/events_filtered.csv")

    print(f"Nombre d'événements retenus : {len(df_filtered)}")
