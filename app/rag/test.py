from app.rag.chain import get_rag_chain

if __name__ == "__main__":
    qa = get_rag_chain()

    question = "What is Artificial Intelligence?"
    result = qa.invoke({"query": question})

    print("\nANSWER:\n", result["result"])
    print("\nSOURCES:")
    for doc in result["source_documents"]:
        print("-", doc.metadata["source"], "page:", doc.metadata.get("page"))