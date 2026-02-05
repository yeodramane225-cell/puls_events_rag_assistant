import pandas as pd

def load_events(csv_path):
    return pd.read_csv(csv_path, sep=";")

if __name__ == "__main__":
    df = load_events("data/raw/evenements-publics-openagenda.csv")
    print(df.head())
    print("CSV chargé avec succès")
