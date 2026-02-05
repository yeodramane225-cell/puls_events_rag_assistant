import pytest
from unittest.mock import patch
from src.rag.rag_query import rag_query

def test_rag_query_basic():
    """
    Vérifie que rag_query retourne bien une chaîne de caractères
    et que le pipeline ne plante pas.
    """

    # On mock l'appel à Ollama pour éviter un vrai appel modèle
    with patch("src.rag.rag_query.query_ollama") as mock_llm:
        mock_llm.return_value = "Réponse simulée"

        response = rag_query("Quels événements ont lieu en 2023 ?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "Réponse simulée" in response


def test_rag_query_no_year():
    """
    Vérifie que rag_query fonctionne même sans année dans la question.
    """

    with patch("src.rag.rag_query.query_ollama") as mock_llm:
        mock_llm.return_value = "Réponse générique"

        response = rag_query("Quels événements ont lieu à Paris ?")

        assert isinstance(response, str)
        assert "Réponse générique" in response
