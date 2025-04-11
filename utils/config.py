from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
import os

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

#SQLite DB 
chunks_schema = """
CREATE TABLE IF NOT EXISTS chunks (
    uuid TEXT PRIMARY KEY,
    doc_uuid TEXT,
    vector_id INTEGER UNIQUE,
    text TEXT,
    embedding BLOB
)"""
docs_schema = """
CREATE TABLE IF NOT EXISTS documents (
    uuid TEXT PRIMARY KEY,
    name TEXT,
    url TEXT,
    text TEXT,
    type TEXT
)"""

schemas = [chunks_schema, docs_schema]

#FAISS
dim = 384

#RAG TEMPLATE
SYS_TEMPLATE = """You are a helpful assistant with access to storage notes.
Use the provided context to inform answer. If the context is empty
or you are unsure, say you don't know. All text beyond this point is context:"""

#Initialize embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


openai_client = OpenAI()
chunker = SentenceSplitter(chunk_size=512)
mistral_client = Mistral(api_key=api_key)