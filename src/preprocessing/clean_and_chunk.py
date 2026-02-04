"""
Module de nettoyage et de découpage (chunking) des événements
pour le projet Puls-Events RAG Assistant.

Ce script :
- nettoie les champs textuels (HTML, espaces, caractères spéciaux),
- normalise les dates,
- génère des chunks de texte pour la vectorisation.

Auteur : Yeo
"""

import pandas as pd
import re
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut :
    - suppression HTML
    - suppression espaces multiples
    - normalisation

    Args:
        text (str): Texte brut.

    Returns:
        str: Texte nettoyé.
    """
    if not isinstance(text, str):
        return ""

    # Supprimer HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Supprimer espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les colonnes importantes du DataFrame.

    Args:
        df (pd.DataFrame): DataFrame des événements.

    Returns:
        pd.DataFrame: DataFrame nettoyée.
    """
    df = df.copy()

    # Colonnes textuelles à nettoyer
    text_cols = ["title", "description", "location"]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Normalisation des dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def chunk_text(text: str, max_tokens: int = 300) -> list:
    """
    Découpe un texte en chunks de taille raisonnable.

    Args:
        text (str): Texte à découper.
        max_tokens (int): Taille max d'un chunk.

    Returns:
        list: Liste de chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)

    return chunks


def create_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un DataFrame contenant un chunk par ligne.

    Args:
        df (pd.DataFrame): DataFrame nettoyée.

    Returns:
        pd.DataFrame: DataFrame avec colonnes :
            - event_id
            - chunk_id
            - text_chunk
    """
    rows = []

    for idx, row in df.iterrows():
        text = f"{row.get('title', '')}. {row.get('description', '')}"
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            rows.append({
                "event_id": idx,
                "chunk_id": i,
                "text_chunk": chunk
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Charger les événements filtrés
    df = pd.read_csv("data/processed/events_filtered.csv")

    # Nettoyage
    df_clean = preprocess_events(df)

    # Chunking
    df_chunks = create_chunks(df_clean)

    # Sauvegarde
    df_chunks.to_csv("data/processed/events_chunks.csv", index=False)

    print(f"Chunks générés : {len(df_chunks)}")
