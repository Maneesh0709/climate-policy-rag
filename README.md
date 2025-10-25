<h1 align="center">🌍 Climate Policy RAG — Retrieval-Augmented Generation Chatbot</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/LLM-OpenAI%20GPT-green" alt="LLM Badge"/>
  <img src="https://img.shields.io/badge/Vector%20DB-FAISS-orange" alt="FAISS Badge"/>
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status Badge"/>
</p>

---

## 📘 Overview
**Climate Policy RAG (Retrieval-Augmented Generation)** is an intelligent chatbot that enables **context-aware question answering** and **summarization of climate policy documents** (such as IPCC reports).  
It combines **Large Language Models (LLMs)** with **vector-based retrieval (FAISS)** to ensure factual accuracy and grounded responses.

---

## ⚙️ Tech Stack
| Component | Description |
|------------|-------------|
| 🧠 **LLM** | OpenAI GPT model for natural language understanding and generation |
| 🧩 **Retriever** | FAISS index for semantic document similarity search |
| 📚 **Framework** | LangChain for building RAG pipelines |
| 🐍 **Language** | Python 3.9+ |
| 💾 **Data** | IPCC & renewable energy policy documents (stored in `/data`) |

---

## 🚀 Features
✅ Retrieve relevant context from large climate datasets  
✅ Generate fact-grounded answers with citations  
✅ Summarize lengthy policy reports concisely  
✅ Modular structure for retraining or model swapping  
✅ Extensible architecture for custom RAG pipelines  

---

## 🏗️ Project Structure

---

## 🧩 How It Works
1. **Document Ingestion** → Text files (e.g., IPCC data) are preprocessed and chunked.  
2. **Embedding Generation** → Converts text into dense vectors using OpenAI embeddings.  
3. **Indexing with FAISS** → Stores embeddings for fast semantic search.  
4. **Retrieval-Augmented Generation (RAG)** → Combines top-matching chunks with LLM context for accurate, cited responses.  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Maneesh0709/climate-policy-rag.git
cd climate-policy-rag
python -m venv .venv
.venv\Scripts\activate   # on Windows
# or
source .venv/bin/activate  # on macOS/Linux
pip install -r requirements.txt
python generate_faiss_index.py
python src/rag_chatbot.py
User: "Summarize the key points from the latest IPCC report on renewable energy."
Bot: "The report emphasizes global emission reduction goals, accelerated solar and wind adoption, and regional carbon policies..."
