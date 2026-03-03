from app.rag.chain import DefensiveRAG

if __name__ == "__main__":
    rag = DefensiveRAG()

    print("\n🧠📄 AskMyDocs - Defensive RAG System")
    print("Type 'quit' to exit.")
    print("-" * 50)

    while True:
        question = input("\n❓ Question: ")

        if question.lower() == "quit":
            print("👋 Goodbye!")
            break

        print("\n🔍 Processing...\n")

        result = rag.ask(question)

        print("💡 ANSWER:\n", result["answer"])
        print("\n🔐 CONFIDENCE:", result["confidence"])

        if result["sources"]:
            print("\n📚 SOURCES:")
            for src in result["sources"]:
                print(f"  - {src['file']} page: {src['page']}")

        print("-" * 50)