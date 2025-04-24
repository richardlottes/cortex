import sys
import os
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
from dotenv import load_dotenv

from configs.llm import ANCHORS, SPAN_TOLERANCE, load_splitter
from utils.processing import chunk_text

RELEVANCE_DETERMINATION_PROMPT = """
You are a relevance determination assistant that outputs **only** JSON array objects.

Given the following question-answer pair and chunked document, your task is to create a list of indexes of the chunks required to answer the question.

**Strict rules for your response**
- Only return a valid JSON list
- Do not include any text outside of the JSON list
- Do not wrap the JSON in Markdown or code blocks
- Do not repeat these instructions
- The keys in each object must be "question", "answer", "relevant_chunks"

**Important note**
- Some text will have indexes of sources that look like [index], the chunk indexes have been annotated as [CHUNK-index].

###Required output format:###
{{
    "question": question,
    "answer": answer,
    "relevant_chunks" index_list

Question: {question}
Answer: {answer}

Chunks:
{chunked_doc}
"""
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_name = "gemini-2.0-flash"
model = genai.GenerativeModel(model_name)

anchor_relevance = dict()
for i, anchor in enumerate(ANCHORS):
    anchor_relevance[anchor] = list()
    splitter = load_splitter(chunk_size=anchor)
    with open("evaluation/corpus.json", "r", encoding="utf-8") as f1, \
         open("evaluation/qa_pairs.json", "r", encoding="utf-8") as f2:
            corpus = json.load(f1)
            qa_pairs = json.load(f2)
            for document in corpus:
                chunked_doc = chunk_text(document["content"], splitter)
                indexed_chunks = "\n".join([f"[CHUNK-{i}] {chunk}" for i, chunk in enumerate(chunked_doc)])
                for qa in qa_pairs:
                    if qa["doc_id"] == document["id"]:
                        prompt = RELEVANCE_DETERMINATION_PROMPT.format(
                            question=qa["question"], 
                            answer=qa["answer"], 
                            chunked_doc=indexed_chunks
                        )
                        response = model.generate_content(prompt, generation_config={"temperature": 0})
                        raw_response = response.candidates[0].content.parts[0].text
                        cleaned_response = re.sub(r"```(?:json)?\n?", "", raw_response).strip("` \n")
                        json_response = json.loads(cleaned_response)

                        if isinstance(json_response, list):
                            json_response = json_response[0]

                        json_response["doc_id"] = document["id"]
                        json_response["model"] = qa["model"]
                        json_response["judge"] = model_name
                        anchor_relevance[anchor].append(json_response)
                        print(json_response)
with open("evaluation/qa_pairs_relevance.json", "w", encoding="utf-8") as f:
    json.dump(anchor_relevance, f, indent=2)