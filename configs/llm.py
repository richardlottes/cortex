import os 
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter
from mistralai import Mistral
from openai import OpenAI
from anthropic import Anthropic


#Load API keys
if os.getenv("GOOGLE_CLOUD_PROJECT") is None:
    load_dotenv()

#Lazy load embedding model - offline implementation
def load_embed_model():
    # return SentenceTransformer("all-MiniLM-L6-v2")
    return SentenceTransformer(".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")

#Lazy load chunker
def load_splitter(chunk_size=512):
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=30)

#Lazy load openai client
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Lazy load anthropic client
def get_anthropic_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

#Lazy load mistral client
def get_mistral_client():
    return Mistral(api_key=os.getenv("MISTRAL_API_KEY"))