from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents(folder_path: str):
    documents = []

    folder = Path(folder_path)
    for pdf_file in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)

    return documents


if __name__ == "__main__":
    docs = load_documents("data/documents")
    print(f"Loaded {len(docs)} pages")
    print(docs[0])