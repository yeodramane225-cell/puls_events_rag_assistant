import pandas as pd
import os

def test_chunks_exist():
    assert os.path.exists("data/processed/events_chunks.csv")

def test_chunks_not_empty():
    df = pd.read_csv("data/processed/events_chunks.csv")
    assert len(df) > 0
    assert df["chunk"].str.len().mean() > 10
