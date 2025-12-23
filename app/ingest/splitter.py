from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.ingest.loader import load_documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    docs = load_documents("data/documents")
    chunks = split_documents(docs)

    print(f"Original pages: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(chunks[0])