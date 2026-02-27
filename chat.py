from src.rag.langchain_rag import rag_langchain
from src.rag.dataset_info import list_available_years  # pour /years

DEBUG = False  # variable globale


def main():
    global DEBUG

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
  /debug on    → Active le mode debug (affiche les chunks FAISS)
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

        if question.lower() == "/debug on":
            DEBUG = True
            print("Mode debug activé.\n")
            continue

        if question.lower() == "/debug off":
            DEBUG = False
            print("Mode debug désactivé.\n")
            continue

        # Traitement RAG via LangChain
        try:
            answer = rag_langchain(question)
            print("\nAssistant:", answer, "\n")
        except Exception as e:
            print("\nErreur:", e, "\n")


if __name__ == "__main__":
    main()
