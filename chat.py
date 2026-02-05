from src.rag.rag_query import rag_query
from src.rag.dataset_info import list_available_years  # pour /years

# Variable globale définie AVANT toute utilisation
DEBUG = False


def main():
    global DEBUG  # autorisé car DEBUG est déjà défini

    print("Assistant RAG — Puls Events")
    print("Tape 'exit' pour quitter.")
    print("Tape '/help' pour afficher les commandes.\n")

    while True:
        question = input("Vous: ").strip()

        # Commandes spéciales
        if question.lower() in ["exit", "quit", "/exit"]:
            print("Assistant: À bientôt !")
            break

        if question.lower() == "/help":
            print("""
Commandes disponibles :
  /help        → Affiche cette aide
  /years       → Liste toutes les années détectées dans le dataset
  /debug on    → Active le mode debug (affiche les chunks FAISS) — (désactivé pour l'instant)
  /debug off   → Désactive le mode debug
  exit         → Quitter le programme
""")
            continue

        if question.lower() == "/years":
            try:
                years = list_available_years()
                print("\nAnnées disponibles dans le dataset :", ", ".join(years), "\n")
            except Exception as e:
                print("\nErreur lors de la récupération des années :", e, "\n")
            continue

        # Mode debug (préparé mais pas encore utilisé dans rag_query)
        if question.lower() == "/debug on":
            DEBUG = True
            print("Mode debug activé (mais rag_query ne l'utilise pas encore).\n")
            continue

        if question.lower() == "/debug off":
            DEBUG = False
            print("Mode debug désactivé.\n")
            continue

        # Traitement RAG normal
        try:
            # rag_query ne supporte PAS encore debug=DEBUG → on l'enlève
            answer = rag_query(question)
            print("\nAssistant:", answer, "\n")
        except Exception as e:
            print("\nErreur:", e, "\n")


if __name__ == "__main__":
    main()
