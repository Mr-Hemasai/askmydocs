#!/usr/bin/env python3
"""
AskMyDocs - Defensive Manual RAG System
"""

import sys
from app.rag.chain import DefensiveRAG


def main():
    print("🧠📄 AskMyDocs - Defensive RAG System")
    print("=" * 50)

    try:
        print("Loading Defensive RAG engine...")
        rag = DefensiveRAG()
        print("✅ System ready!")

        print("\nEnter your questions (type 'quit' to exit):")
        print("-" * 50)

        while True:
            try:
                question = input("\n❓ Question: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                print("\n🔍 Processing...\n")

                result = rag.ask(question)

                print("💡 ANSWER:\n", result["answer"])
                print("\n🔐 CONFIDENCE:", result["confidence"])

                if result["sources"]:
                    print("\n📚 SOURCES:")
                    for src in result["sources"]:
                        print(f"  - {src['file']} page: {src['page']}")

                print("-" * 50)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

            except Exception as e:
                print(f"❌ Error while processing question: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()