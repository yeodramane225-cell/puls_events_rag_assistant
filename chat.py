from src.rag.rag_query import rag_query

def main():
    print("Assistant RAG — Puls Events")
    print("Tape 'exit' pour quitter.\n")

    while True:
        question = input("Vous: ")

        if question.lower() in ["exit", "quit"]:
            print("Assistant: À bientôt !")
            break

        try:
            answer = rag_query(question)
            print("\nAssistant:", answer, "\n")
        except Exception as e:
            print("\nErreur:", e, "\n")

if __name__ == "__main__":
    main()
