from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from app.vectorstore.embeddings import get_embedding_model
from app.rag.prompt import PROMPT

VECTOR_DB_PATH = "vector_db"

def get_rag_chain():
    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(
        model="mistral",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain