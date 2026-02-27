import os
import pytest
from langchain_mistralai import MistralAIEmbeddings

def test_embedding_shape():
    api_key = os.getenv("MISTRAL_API_KEY")
    assert api_key, "La variable d'environnement MISTRAL_API_KEY est manquante."

    embedder = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )

    vector = embedder.embed_query("Bonjour le monde")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)