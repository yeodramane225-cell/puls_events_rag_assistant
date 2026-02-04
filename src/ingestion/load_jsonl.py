import json

def load_events_streaming(jsonl_path):
    """
    Lit un fichier JSONL massif ligne par ligne sans jamais charger plus d'un objet en mémoire.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # ignorer les lignes vides
                yield json.loads(line)


if __name__ == "__main__":
    json_file = "data/raw/evenements-publics-openagenda.json"

    for i, event in enumerate(load_events_streaming(json_file)):
        print(f"Événement {i} chargé")
        if i == 10:
            break
