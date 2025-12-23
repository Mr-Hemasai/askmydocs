from langchain_community.vectorstores import FAISS
from app.ingest.loader import load_documents
from app.ingest.splitter import split_documents
from app.vectorstore.embeddings import get_embedding_model
import os

VECTOR_DB_PATH = "vector_db"

def build_vector_store():
    docs = load_documents("data/documents")
    chunks = split_documents(docs)

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTOR_DB_PATH)
    print("Vector store created and saved.")

if __name__ == "__main__":
    build_vector_store()