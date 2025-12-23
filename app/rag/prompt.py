from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)