from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from app.vectorstore.embeddings import get_embedding_model

VECTOR_DB_PATH = "vector_db"
SIMILARITY_THRESHOLD = 0.65   # Adjust after testing
TOP_K = 4


class DefensiveRAG:

    def __init__(self):
        embeddings = get_embedding_model()

        self.vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.llm = ChatOllama(
            model="mistral",
            temperature=0
        )

    def validate_query(self, query: str):
        query = query.strip().lower()

        if len(query) < 3:
            return False, "Please ask a meaningful question related to the documents."

        greetings = ["hi", "hello", "hey", "hii"]
        if query in greetings:
            return False, "Hello! Please ask a question related to the uploaded documents."

        return True, None

    def retrieve(self, query: str):
     docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=TOP_K)

     if not docs_with_scores:
        return [], None

      # FAISS returns L2 distance (lower is better)
     best_distance = min(score for _, score in docs_with_scores)

     # Reject if distance too large
     if best_distance > 1.2:   # Adjust after testing
        return [], best_distance

     # Keep documents close enough
     filtered_docs = [
        doc for doc, score in docs_with_scores
        if score <= 1.2
     ]
     return filtered_docs, best_distance

    def build_context(self, documents):
        return "\n\n".join(doc.page_content for doc in documents)

    def generate_answer(self, query: str, context: str):
        prompt = f"""
You are AskMyDocs, a strict document-grounded assistant.

RULES:
- Use ONLY the context below.
- If context is insufficient, say:
  "I could not find relevant information in the provided documents."
- Do not use outside knowledge.
- Do not guess.

Context:
{context}

Question:
{query}

Answer:
"""

        return self.llm.invoke(prompt).content

    def ask(self, query: str):

        # 1️⃣ Validate query
        valid, message = self.validate_query(query)
        if not valid:
            return {
                "answer": message,
                "confidence": "None",
                "sources": []
            }

        # 2️⃣ Retrieve
        documents, max_score = self.retrieve(query)

        if not documents:
            return {
                "answer": "I could not find relevant information in the provided documents.",
                "confidence": "Low",
                "sources": []
            }

        # 3️⃣ Build context
        context = self.build_context(documents)

        # 4️⃣ Generate answer
        answer = self.generate_answer(query, context)

        # 5️⃣ Confidence scoring
        if max_score > 0.80:
            confidence = "High"
        elif max_score > 0.70:
            confidence = "Medium"
        else:
            confidence = "Low"

        sources = [
            {
                "file": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources
        }