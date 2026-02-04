import subprocess
import json

def embed(text):
    result = subprocess.run(
        ["ollama", "run", "nomic-embed-text"],
        input=text.encode(),
        stdout=subprocess.PIPE
    )
    return json.loads(result.stdout.decode())["embedding"]

def test_embedding_shape():
    emb = embed("Bonjour")
    assert isinstance(emb, list)
    assert len(emb) > 100
