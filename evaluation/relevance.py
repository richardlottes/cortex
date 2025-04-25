import sys
import os
import json
import re

from dotenv import load_dotenv
import google.generativeai as genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.llm import load_splitter
from configs.constants import (
    GEMINI_MODEL_NAME, 
    RELEVANCE_DETERMINATION_PROMPT, 
    CORPUS_PATH, 
    QA_PAIRS_PATH, 
    QA_RELEVANCE_PATH,
    ANCHORS
)

from utils.processing import chunk_text


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def parse_gemini_response(raw_response: str):
    """
    Parse the raw response from gemini.

    Parameters:
    ----------
    - raw_response (str): The raw text from a Gemini repsonse.

    Returns:
    -------
    - Cleaned JSON version of Gemini's response.
    """

    #Remove HTML and transform into JSON 
    cleaned_response = re.sub(r"```(?:json)?\n?", "", raw_response).strip("` \n")
    json_response = json.loads(cleaned_response)
    
    #Pull our of array if incorrectly formatted
    if isinstance(json_response, list):
        json_response = json_response[0]

    return json_response


def generate_relevance_scores():
    """
    Generate and save relevance scores for all loaded QA pairs and corresponding chunks.
    """

    model = genai.GenerativeModel(GEMINI_MODEL_NAME) #Instantiate model
    anchor_relevance = dict() #
    #Iterate over chunk sizes (anchors)
    for anchor in ANCHORS:
        anchor_relevance[anchor] = list()
        splitter = load_splitter(chunk_size=anchor) #Load new splitter for given chunk size
        #Open & load Corpus and QA Pairs
        with open(CORPUS_PATH, "r", encoding="utf-8") as f1, \
            open(QA_PAIRS_PATH, "r", encoding="utf-8") as f2:
                corpus = json.load(f1)
                qa_pairs = json.load(f2)
                #Chunk each document in the corpus and format them for Gemini
                for document in corpus:
                    chunked_doc = chunk_text(document["content"], splitter)
                    indexed_chunks = "\n".join([f"[CHUNK-{i}] {chunk}" for i, chunk in enumerate(chunked_doc)])
                    #Iterate over QA pairse and, for QA pairs created from the chunked document, determine which chunks are relevant
                    for qa in qa_pairs:
                        if qa["doc_id"] == document["id"]:
                            prompt = RELEVANCE_DETERMINATION_PROMPT.format(
                                question=qa["question"], 
                                answer=qa["answer"], 
                                chunked_doc=indexed_chunks
                            )
                            response = model.generate_content(prompt, generation_config={"temperature": 0}) #Determine chunk relevance
                            raw_response = response.candidates[0].content.parts[0].text
                            #Format json response and enrich with metadata
                            json_response = parse_gemini_response(raw_response)
                            json_response["doc_id"] = document["id"]
                            json_response["model"] = qa["model"]
                            json_response["judge"] = GEMINI_MODEL_NAME
                            anchor_relevance[anchor].append(json_response) 
    #Save relevance judgements for later use                      
    with open(QA_RELEVANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(anchor_relevance, f, indent=2)