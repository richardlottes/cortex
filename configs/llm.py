import os 
from dotenv import load_dotenv

from mistralai import Mistral
from openai import OpenAI
from anthropic import Anthropic
from llama_index.core.node_parser import SentenceSplitter


#Load API keys
if os.getenv("GOOGLE_CLOUD_PROJECT") is None:
    load_dotenv()

#Initialize chunker
def load_chunker():
    return SentenceSplitter(chunk_size=512, chunk_overlap=30)

#Initialize foundation model clients
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_anthropic_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_mistral_client():
    return Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


#FAISS
dim = 384

#RAG TEMPLATE
SYS_TEMPLATE = """You are a helpful assistant with access to storage notes.
Use only the provided context to answer the user's questions. If you reference a specific piece of context, include a reference number like [1], [2], etc.
If the context is missing, irrelevant, or insufficient, say "I don’t know based on the provided information."
Do not guess or make assumptions. If the user is just making conversation, feel free to respond conversationally.

Below is the context:"""

#MODEL SELECTION
MODELS = {
    "gpt-4o-mini": "openai",
    "claude-3-haiku-20240307": "anthropic"
}