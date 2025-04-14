# 🧠 Cortex: A Lightweight RAG App for Document Search
Cortex is a lightweight Retrieval-Augmented Generation (RAG) app built with Python and Streamlit. It allows users to upload documents (PDFs, TXT files, YouTube videos, raw text) and query them using natural language via an LLM, backed by a semantic search engine and persistent local memory.

---

## Features
### Multi-format Document Ingestion
- Upload files (PDF and TXT)
- Fetch and process YouTube transcripts (local only since YouTube has cracked down on bots)
- Manually input raw text

### Storage Interactions
- View stored documents
- Delete specific documents
- Reset storage entirely 

### Context-aware LLM Responses via Semantic Search with FAISS
- Retrieves the most relevant document chunks via semantic search (cosine similarity-based retrieval)
- Passes them as context to the LLM for accurate answers

### Persistent Metadata with SQLite
- Tracks document metadata and chunk mappings
- Tied to FAISS vector IDs for consistent retrieval

### Session-level Isolation
- New FAISS index and SQLite DB generated each session

### Deployable on Google Cloud Run
- Streamlit frontend served via GCP

---

## Tech Stack
- Streamlit — UI + session management
- FAISS — vector similarity search
- SQLite — lightweight relational DB for metadata
- sentence-transformers — for embedding generation
- Mistral - OCR PDF processing
- OpenAI — LLM responses
- llama-index — for document parsing and node chunking
- yt-dlp — (optional) YouTube transcript extraction

---

## Getting Started (Local)

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/cortex.git
cd cortex
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a .env file in the root with the following API keys.
```python
OPENAI_API_KEY=<your-secret>
MISTRAL_API_KEY=<your-secret>
```

### 4. Run locally
```bash
streamlit run main.py
```
