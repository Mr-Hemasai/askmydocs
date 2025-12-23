# AskMyDocs 🧠📄  
*A Local, Free, Production-Ready RAG System*

AskMyDocs is a **Retrieval-Augmented Generation (RAG)** application that allows users to ask natural-language questions over their **private PDF documents** and receive **accurate, source-grounded answers**.

The entire system runs **locally**, without any paid APIs, making it privacy-friendly, cost-free, and suitable for real-world deployment and learning.

---

## 🚀 Key Features

- 📄 Ask questions directly on your PDF documents
- 🔍 Semantic search using vector embeddings (not keyword matching)
- 🧠 Grounded answers generated using retrieved context
- 📚 Source citations with page numbers
- 💻 Fully local & free (no OpenAI / cloud dependency)
- 🧩 Clean, modular, production-style architecture

---

## 🧠 How It Works (RAG Pipeline)
PDF Documents
↓
Text Extraction & Chunking
↓
Sentence-Transformer Embeddings
↓
FAISS Vector Database
↓
Retriever (Semantic Search)
↓
Prompt + Context Injection
↓
Local LLM (Ollama – Mistral / LLaMA)
↓
Answer + Source Pages

This follows the **Retrieval-Augmented Generation (RAG)** pattern used in modern AI systems.

---

## 🛠️ Tech Stack

### Core
- **Python 3**
- **LangChain**
- **FAISS** (Vector database)
- **Sentence-Transformers** (`all-MiniLM-L6-v2`)
- **Ollama** (Local LLM runtime)

### Models
- **Mistral** / **LLaMA 3.2 (3B)** — local inference

### Utilities
- **PyPDF** — PDF parsing
- **FastAPI** *(planned API layer)*

---

## 📂 Project Structure
askmydocs/
│
├── app/
│   ├── ingest/        # PDF loading & text splitting
│   ├── vectorstore/   # Embeddings & FAISS storage
│   ├── rag/           # Prompt & RAG chain
│   ├── api/           # API layer (planned)
│
├── data/
│   └── documents/     # Input PDFs
│
├── vector_db/         # FAISS index (generated)
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Mr-Hemasai/askmydocs.git
cd askmydocs
### Create virtual environment
python3 -m venv venv
source venv/bin/activate
### Install dependencies
pip install -r requirements.txt
### Install & run Ollama
brew install ollama
ollama pull mistral
