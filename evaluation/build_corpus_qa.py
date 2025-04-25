import os
import json
import sys
from typing import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.llm import QA_GENERATION_PROMPT, DEDUP_QA_GENERATION_PROMPT
from utils.processing import ask_openai, ask_anthropic, safe_parse_json


def build_rag_eval_corpus():
    """
    Builds corpus of documents in JSON format for evaluation.
    """

    docs = list()
    for i, filename in enumerate(sorted(os.listdir("evaluation/assets"))):
        with open(os.path.join("evaluation/assets", filename), "r", encoding="utf-8") as f:
            docs.append(
                {
                    "id": i,
                    "name": filename,
                    "content": f.read().strip()
                }
                
            )
    with open("evaluation/corpus.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)


def build_qa_pairs(ask: Callable):
    """
    Build QA pairs using JSON corpus of documents with OpenAI
    """

    with open("evaluation/corpus.json", "r", encoding="utf-8") as f:
        model = "gpt-4.1-2025-04-14"
        corpus = json.load(f)
        cumulative_pairs = list()
        for doc in corpus:
            prompt = [{
                "role": "user",
                "content" : QA_GENERATION_PROMPT.format(context=doc["content"])
                }]
            qa_pairs = ask(prompt, model=model)
            parsed_pairs = safe_parse_json(qa_pairs)
            for pair in parsed_pairs:
                pair["model"] = model
                pair["doc_id"] = doc["id"]
            cumulative_pairs+=parsed_pairs
        return cumulative_pairs


def build_qa_pairs_dedup(ask: Callable, pairs: list):
    """
    Build QA pairs using JSON corpus of documents with Anthropic
    """

    with open("evaluation/corpus.json", "r", encoding="utf-8") as f:
        model = "claude-3-7-sonnet-20250219"
        corpus = json.load(f)
        cumulative_pairs = list()
        for doc in corpus:
            prompt = [{
                "role": "user",
                "content" : DEDUP_QA_GENERATION_PROMPT.format(pairs=[pair for pair in pairs if pair["doc_id"]==doc["id"]], context=doc["content"])
                }]
            qa_pairs = ask(prompt, model=model)
            parsed_pairs = safe_parse_json(qa_pairs)
            for pair in parsed_pairs:
                pair["model"] = model
                pair["doc_id"] = doc["id"]
            cumulative_pairs+=parsed_pairs
        return cumulative_pairs

# build_rag_eval_corpus()

openai_qa_pairs = build_qa_pairs(ask_openai)
anthropic_qa_pairs = build_qa_pairs_dedup(ask_anthropic, openai_qa_pairs)
qa_pairs = openai_qa_pairs + anthropic_qa_pairs
with open("evaluation/qa_pairs.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=2)



