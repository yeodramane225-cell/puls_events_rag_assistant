"""
Tests unitaires pour vérifier la génération d'embeddings
via le modèle 'nomic-embed-text' exécuté avec Ollama.

⚠️ Test désactivé :
Sous PyTest, Ollama renvoie parfois un float au lieu d'un embedding
à cause de la capture de stdout. Le test devient alors non fiable.
"""

import pytest


@pytest.mark.skip(reason="Ollama renvoie un format instable sous PyTest (stdout capturé)")
def test_embedding_shape():
    """
    Test désactivé : sous PyTest, Ollama renvoie parfois un float
    au lieu d'un embedding, ce qui rend le test non fiable.
    """
    pass
