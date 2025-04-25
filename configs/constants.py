
from enum import Enum


#LLM class 
class LLM(str, Enum):
    openai = "openai"
    anthropic = "anthropic"

#FAISS DIMENSIONALITY
dim = 384

#MODEL SELECTION DROPDOWN
MODELS = {
    "gpt-4o-mini": "openai",
    "claude-3-haiku-20240307": "anthropic"
}

#EVAlUATION
ANCHORS = [256, 512, 768, 1024, 1280]
SPAN_TOLERANCE = 128

#FROZEN GEMINI MODEL FOR RELEVANCE JUDGEMENT
GEMINI_MODEL_NAME = "gemini-2.0-flash"

#PATHS FOR EVALUATION
ASSETS_PATH = "evaluation/assets"
CORPUS_PATH = "evaluation/corpus.json"
QA_PAIRS_PATH = "evaluation/qa_pairs.json"
QA_RELEVANCE_PATH = "evaluation/qa_pairs_relevance.json"