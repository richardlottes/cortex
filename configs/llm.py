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
def load_splitter(chunk_size=512):
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=30)

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
SYS_TEMPLATE = """
You are a helpful assistant with access to storage notes.

Use only the provided context to answer the user's questions. If you reference a specific piece of context, include a reference number like [1], [2], etc.

If the context is missing, irrelevant, or insufficient, say "I don’t know based on the provided information."

Do not guess or make assumptions. If the user is just making conversation, respond conversationally.

Below is the context:"""

#MODEL SELECTION
MODELS = {
    "gpt-4o-mini": "openai",
    "claude-3-haiku-20240307": "anthropic"
}


QA_GENERATION_PROMPT = """
You’re a data generation assistant that outputs only json array objects. Your task is to produce only a **valid JSON list** of exactly 3 question-answer pairs about the following passage.

**Each question must be:**
- Fact-based
- Non-trivial
- Diverse in content

**Your response must folow these strict rules**
- Only return a valid JSON list
- Do not include any text outside of the JSON list
- Do not wrap the JSON in Markdown or code blocks
- Do not repeat these instructions
- The keys in each object must be "question" and "answer"

###Required output format:###
[
  {{
    "question": "What is the primary goal of reinforcement learning?",
    "answer": "The primary goal of reinforcement learning is to learn a policy that maximizes cumulative reward through trial-and-error interactions with an environment."
  }},
  ...
]

###Passage:###
{context}"""

DEDUP_QA_GENERATION_PROMPT = """
You’re a data generation assistant that outputs only json array objects. Your task is to produce only a **valid JSON list** of exactly 3 question-answer pairs about the following passage.

**Each question must be:**
- Fact-based
- Non-trivial
- Diverse in content

**Your response must folow these strict rules**
- Only return a valid JSON list
- Do not include any text outside of the JSON list
- Do not wrap the JSON in Markdown or code blocks
- Do not repeat these instructions
- The keys in each object must be "question" and "answer"
- Do not repeat, rephrase, paraphrase any of the following existing QA pairs
- Each QA pair must be unique

Do **not** repeat, rephrase, or paraphrase any of these existing QA pairs:
{pairs}

###Required output format:###
[
  {{
    "question": "What is the primary goal of reinforcement learning?",
    "answer": "The primary goal of reinforcement learning is to learn a policy that maximizes cumulative reward through trial-and-error interactions with an environment."
  }},
  ...
]

###Passage:###
{context}"""


ANCHORS = [256, 512, 768, 1024, 1280]
SPAN_TOLERANCE = 128