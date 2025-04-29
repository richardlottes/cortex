# 🧠 Cortex: A Lightweight RAG App for Document Search
Cortex is a lightweight Retrieval-Augmented Generation (RAG) app built with Python and Streamlit. It allows users to upload documents (PDFs, TXT files, YouTube videos, raw text) and query them using natural language via an LLM, backed by a semantic search engine and persistent local memory.

---

[![Watch the demo](https://img.youtube.com/vi/HciglJgE7XQ/maxresdefault.jpg)](https://youtu.be/HciglJgE7XQ)

---

## Features
### Multi-format Document Ingestion
- Upload files (PDF and TXT)
- Manually input raw text
- Fetch and process YouTube transcripts (local only since YouTube has cracked down on bots)

### Chat
- Switch between gpt-4o-mini and claude-3-haiku
- Toggle to show retrieved context chunks and relevant metadata such as similarity score

### Storage + Semantic Indexing
- SQLite for document and chunk metadata persistent
    - Tracks document metadata and chunk mappings
    - Tied to FAISS vector IDs for consistent retrieval
- FAISS vector store for vector retrieval
    - Retrieves the most relevant document chunks via semantic search (cosine similarity-based retrieval) to be passed as context to LLM for accurate answers
- User Operations
    - View stored documents
    - Delete specific documents
    - Reset storage entirely 
- New FAISS index and SQLite DB generated each session

### Chunk Size Evaluation Dashboard (Frozen, Reproducible)
- 14-document benchmark corpus
- Over 100 frozen QA pairs (1/2 from each GPT-4 and Claude to mitigate bias)
- Frozen Gemini-judged relevance annotations with JSON-based ground truth
- Metrics include:
    - Precision@k
    - Recall@k
    - Average Relevant Similarity
    - Average Overall Similarity
    - DCG@k
    - nDCG@k
- Plotted with Plotly in Streamlit

### Deployable on Google Cloud Run
- Streamlit frontend served via GCP

---

## Tech Stack
- Streamlit & Plotly — UI + session management
- FAISS — vector similarity search
- SQLite — lightweight relational DB for metadata
- sentence-transformers — for embedding generation
- Mistral - OCR PDF processing
- OpenAI, Anthropic, Google — LLM responses & Automated Data Labeling for Evals
- llama-index — for document parsing and node chunking

---

## Getting Started (Local)

### 1. Clone the repo
```bash
git clone git@github.com:richardlottes/cortex.git
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
ANTHROPIC_API_KEY=<your-secret>
MISTRAL_API_KEY=<your-secret>
GOOGLE_API_KEY=<your-secret>
```

### 4. Run locally
```bash
streamlit run main.py
```
