import json
import sys
import os

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.processing import  ask_openai, ask_anthropic

from evaluation.corpus_qa_pairs import build_corpus, build_qa_pairs, build_qa_pairs_dedup
from evaluation.relevance import generate_relevance_scores
from evaluation.metrics import evaluate_top_k

#Build and save corpus JSON
build_corpus()

#Build and save QA pairs
openai_qa_pairs = build_qa_pairs(ask_openai)
anthropic_qa_pairs = build_qa_pairs_dedup(ask_anthropic, openai_qa_pairs)
qa_pairs = openai_qa_pairs + anthropic_qa_pairs
with open("evaluation/qa_pairs.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=2)

#Generate relevance scores for QA pairs
generate_relevance_scores()

#Evalute retrieval vs relevant chunks
df_3 = evaluate_top_k(3)
df_5 = evaluate_top_k(5)
df_10 = evaluate_top_k(10)

#Concatenate evaluation DFs and save locally
df = pd.concat([df_3, df_5, df_10])
df.to_csv("evaluation/custom_eval.csv")








