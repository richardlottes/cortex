import os
import json
import sys
from typing import Callable, List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.constants import ASSETS_PATH, CORPUS_PATH
from configs.llm import QA_GENERATION_PROMPT, DEDUP_QA_GENERATION_PROMPT

from utils.processing import safe_parse_json


def build_corpus():
    """
    Builds corpus of documents in JSON format for evaluation.
    """

    #Iterate through .txt files in assets/
    docs = list()
    for i, filename in enumerate(sorted(os.listdir(ASSETS_PATH))):
        with open(os.path.join(ASSETS_PATH, filename), "r", encoding="utf-8") as f:
            #Add each doc's metadata to docs list
            docs.append(
                {
                    "id": i,
                    "name": filename,
                    "content": f.read().strip()
                }
            )
    #Write locally as JSON file
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)


def build_qa_pairs(ask: Callable) -> List[Dict]:
    """
    Build QA pairs using JSON corpus of documents with OpenAI.

    Parameters:
    ----------
    - ask (Callable): A function to hit either OpenAI/Anthropic endpoints.

    Returns:
    - A list of QA pairs.
    -------
    """

    #Load corpus of docs
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        model = "gpt-4.1-2025-04-14"
        corpus = json.load(f)
        cumulative_pairs = list()
        #Iterate through docs and get QA pairs for each
        for doc in corpus:
            prompt = [{
                "role": "user",
                "content" : QA_GENERATION_PROMPT.format(context=doc["content"])
                }]
            qa_pairs = ask(prompt, model=model)
            parsed_pairs = safe_parse_json(qa_pairs)
            #Add helpful metadata to each QA pair
            for pair in parsed_pairs:
                pair["model"] = model
                pair["doc_id"] = doc["id"]
            cumulative_pairs+=parsed_pairs
        return cumulative_pairs


def build_qa_pairs_dedup(ask: Callable, pairs: List[Dict]) -> List[Dict]:
    """
    Build QA pairs using JSON corpus of documents with Anthropic.

    Parameters:
    ----------
    - ask (Callable): A function to hit either OpenAI/Anthropic endpoints.
    - pairs (List[Dict]): Existing QA pairs.

    Returns:
    -------
    - A list of QA pairs.
    """
    
    #Load corpus of docs
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        model = "claude-3-7-sonnet-20250219"
        corpus = json.load(f)
        cumulative_pairs = list()
        #Iterate through docs and get QA pairs for each
        for doc in corpus:
            prompt = [{
                "role": "user",
                "content" : DEDUP_QA_GENERATION_PROMPT.format(pairs=[pair for pair in pairs if pair["doc_id"]==doc["id"]], context=doc["content"])
                }]
            qa_pairs = ask(prompt, model=model)
            parsed_pairs = safe_parse_json(qa_pairs)
            #Add helpful metadata to each QA pair
            for pair in parsed_pairs:
                pair["model"] = model
                pair["doc_id"] = doc["id"]
            cumulative_pairs+=parsed_pairs
        return cumulative_pairs