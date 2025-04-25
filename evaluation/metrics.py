import sys
import os
import json 
import math
from typing import Tuple, List

import numpy as np 
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.functionals import FAISSFunctional
from core.processing import chunk_text

from configs.constants import CORPUS_PATH, QA_RELEVANCE_PATH, ANCHORS
from configs.llm import load_splitter, load_embed_model



embedding_model = load_embed_model()


def recall_k(returned_relevant: int, total_relevant: int) -> float:
    """
    Computes Recall@k

    Parameters:
    ----------
    - returned_relevant (int): The number of relevant chunks returned 
    - total_relevant (int): The total relevant chunks for the question

    Returns:
    -------
    - Recall@k
    """

    return round(returned_relevant/total_relevant, 2)


def precision_k(returned_relevant: int, k: int) -> float:
    """
    Computes Precision@k


    Parameters:
    ----------
    - returned_relevant (int): The number of relevant chunks returned
    - k (int): The number of items to be retrieved

    Returns:
    -------
    - Precision@k
    """

    return round(returned_relevant/k, 2)


def dcg_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    """
    Computes DCG@k

    Parameters:
    ----------
    - retrieved_ids (List[int]): A list of retrieved chunks
    - relevant_ids (List[int]): A list of relevant chunks

    Returns:
    -------
    - DCG@k
    """

    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            dcg += 1/math.log2(i+2)

    return dcg

def ndcg_k(dcg: float, relevant_ids: List[int], k: int) -> float:
    """
    Computes nDCG@k

    Parameters:
    ----------
    - dcg (float): The DCG value computed with dcg_k()
    - relevant_ids (List[int]): A list of relevant chunks
    - k (int): The number of items to be retrieved

    Returns:
    -------
    - nDCG@k
    """

    ideal_rels = [1] * min(len(relevant_ids), k)
    idcg = sum([rel/math.log2(i+2) for i, rel in enumerate(ideal_rels)])

    if idcg == 0:
        return 0
    return round(dcg/idcg, 2)


def extract_doc_id_chunk_idx(faiss_id: int) -> Tuple(int, int):
    """
    Extracts document ID and chunk index for tie back 

    Parameters:
    ----------
    - FAISS ID constructed from a doc_id and chunk_idx

    Returns:
    -------
    - Document ID and chunk index
    """

    doc_id = (faiss_id-1000) // 1000
    chunk_idx = (faiss_id-1000) % 1000
    return doc_id, chunk_idx


def build_indexes():
    """
    Builds and saves FAISS indexes with chunks to be used in evaluation
    """
    
    #Load Corpus and QA Pairs
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

        #Iterate over chunk sizes (anchors)
        for anchor in ANCHORS:
            #Build FAISSFunctional for each anchor (chunk size) & instantiate splitter
            faiss_functional = FAISSFunctional(embedding_dim=embedding_model.get_sentence_embedding_dimension())
            splitter = load_splitter(chunk_size=int(anchor))
            #Chunk each doc in the corpus
            for doc in corpus:
                chunked = chunk_text(doc["content"], splitter)
                #Embed each chunk and add to FAISSFunctional index
                for i, chunk in enumerate(chunked):
                    emb = embedding_model.encode(chunk)
                    faiss_id = (1+doc['id'])*1000 + i
                    faiss_functional.add_embs(emb.reshape(1, -1), custom=[faiss_id])
            #At end of anchor iteration, save FAISSFunctional
            faiss_functional.save(f"evaluation/eval_{str(anchor)}.faiss")
    
        
def evaluate_top_k(k: int=3) -> pd.DataFrame:
    """
    Evaluates top k for: 
        - Precision@k
        - Recall@k
        - Average Relevant Similarity
        - Average Overall Similarity
        - DCG@k
        - nDCG@k

    Parameters:
    ----------
    - k (int): The number of items to be retrieved

    Returns:
    -------
    - A DF of all computed metrics at each anchor (chunk size) for k
    """

    #Load saved QA Pairs with relevance determinations
    metrics = dict()
    with open(QA_RELEVANCE_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

        #Iterate over each anchor (chunk size) and associated pairs (QA pairs)
        for anchor, pairs in qa_pairs.items():
            metrics[anchor]= {
                "dcg@k": list(),
                "ndcg@k": list(),
                "precision@k": list(),
                "recall@k": list(),
                "relevant_similarity": list(),
                "overall_similarity": list()
            }
            #Load faiss indexes saved in build_indexes()
            faiss_functional = FAISSFunctional()
            faiss_functional.load(f"evaluation/eval_{str(anchor)}.faiss")
            #Iterate over all QA pairs for anchor, querying each question and evaluating the retrieved chunks' relevance against the relevant chunks determined by Gemini
            for pair in pairs:                    
                query = embedding_model.encode([pair["question"]], convert_to_numpy=True)
                D, I = faiss_functional.query(query, k)
                total_relevant = len(pair["relevant_chunks"])
                tp = 0
                #Iterate over retrieved chunks and compare to relevant chunks
                for sim, i in zip(D[0], I[0]):
                    doc_id, chunk_id = extract_doc_id_chunk_idx(i)
                    if doc_id == pair["doc_id"] and chunk_id in pair["relevant_chunks"]:
                        tp+=1
                        metrics[anchor]["relevant_similarity"].append(sim)
                #Append metrics to dict that will be used to generate the final DF
                metrics[anchor]["precision@k"].append(precision_k(tp, k))
                metrics[anchor]["recall@k"].append(recall_k(tp, total_relevant))
                metrics[anchor]["overall_similarity"].append(D[0])
                retrieved_chunks = [extract_doc_id_chunk_idx(i)[1] for i in I[0] if extract_doc_id_chunk_idx(i)[0] == pair["doc_id"]]
                dcg = dcg_k(retrieved_chunks, pair["relevant_chunks"], k)
                ndcg = ndcg_k(dcg, pair["relevant_chunks"], k)
                metrics[anchor]["dcg@k"].append(dcg)
                metrics[anchor]["ndcg@k"].append(ndcg)
            metrics[anchor]["num_chunks"] = len(faiss_functional)
        
        df_dict = {
            "chunk_size": list(),
            "k": list(),
            "num_chunks": list(),
            "dcg@k": list(),
            "ndcg@k": list(),
            "precision@k": list(),
            "recall@k": list(),
            "relevant_similarity": list(),
            "overall_similarity": list()
        }

        #Generate final DF of metrics for each chunk size at the given k
        for anchor, metrics in metrics.items():
            df_dict["chunk_size"].append(anchor)
            df_dict["k"].append(k)
            df_dict["num_chunks"].append(metrics["num_chunks"])
            df_dict["dcg@k"].append(np.mean(metrics["dcg@k"]))
            df_dict["ndcg@k"].append(np.mean(metrics["ndcg@k"]))
            df_dict["precision@k"].append(np.mean(metrics["precision@k"]))
            df_dict["recall@k"].append(np.mean(metrics["recall@k"]))
            df_dict["relevant_similarity"].append(np.mean(metrics["relevant_similarity"]))
            df_dict["overall_similarity"].append(np.mean(metrics["overall_similarity"]))

        return pd.DataFrame(df_dict)