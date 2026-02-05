"""
Extraction des années présentes dans les événements OpenAgenda.

Ce module :
- analyse les chunks textuels pour détecter toutes les années mentionnées,
- fournit une fonction listant toutes les années présentes dans le dataset,
- construit un mapping event_id -> années détectées dans ses chunks,
- expose une variable globale `event_years` prête à être importée par le pipeline RAG.

Auteur : Yeo
Projet : Puls-Events RAG Assistant
"""

import pandas as pd
import re


# ---------------------------------------------------------
# 1. Lister toutes les années présentes dans le dataset
# ---------------------------------------------------------
def list_available_years():
    """
    Retourne la liste triée de toutes les années détectées
    dans le dataset chunké.
    """
    df = pd.read_pickle("data/vectorstore/metadata.pkl")

    pattern = r"\b(19\d{2}|20\d{2})\b"
    years = set()

    for text in df["text_chunk"].dropna():
        found = re.findall(pattern, text)
        years.update(found)

    return sorted(years)


# ---------------------------------------------------------
# 2. Construire un mapping event_id -> années détectées
# ---------------------------------------------------------
def build_event_years():
    """
    Construit un dictionnaire :
        event_id -> liste des années trouvées dans ses chunks.
    """
    df = pd.read_pickle("data/vectorstore/metadata.pkl")

    pattern = r"\b(19\d{2}|20\d{2})\b"
    event_years = {}

    for _, row in df.iterrows():
        eid = row["event_id"]
        text = row["text_chunk"]

        years = re.findall(pattern, text)

        if eid not in event_years:
            event_years[eid] = set()

        event_years[eid].update(years)

    # Convertir en listes triées
    return {k: sorted(list(v)) for k, v in event_years.items()}


# ---------------------------------------------------------
# 3. Variable globale prête à importer
# ---------------------------------------------------------
event_years = build_event_years()
