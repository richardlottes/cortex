import sys
import os
import json 
import math
from typing import Dict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import faiss
from utils.functionals import FAISSFunctional
from utils.processing import chunk_text
from configs.llm import load_splitter, ANCHORS
from configs.embed import load_embed_model
import numpy as np
import pandas as pd

embedding_model = load_embed_model()

def recall_k(returned_relevant: int, total_relevant: int) -> float:
    return round(returned_relevant/total_relevant, 2)


def precision_k(returned_relevant: int, k: int) -> float:
    return round(returned_relevant/k, 2)


def dcg_k(retrieved_ids, relevant_ids):
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            dcg += 1/math.log2(i+2)

    return dcg

def ndcg_k(dcg, relevant_ids, k):
    ideal_rels = [1] * min(len(relevant_ids), k)
    idcg = sum([rel/math.log2(i+2) for i, rel in enumerate(ideal_rels)])

    if idcg == 0:
        return 0
    return round(dcg/idcg, 2)


def extract_doc_id_chunk_idx(faiss_id):
    """
    
    """
    doc_id = (faiss_id-1000) // 1000
    chunk_idx = (faiss_id-1000) % 1000
    return doc_id, chunk_idx

#READ CORPUS
def build_indexes():
    """
    
    """
    with open("evaluation/corpus.json", "r", encoding="utf-8") as f:
        
        #Load up corpus and qa pairs & instantiate a dict to hold anchor <> faiss index map
        corpus = json.load(f)
        anchor_faiss_map = dict()

        #Iterate over anchors
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
            #At end of anchor iteration, set FAISSFunctional object to corresponding anchor key in map
            faiss_functional.save(f"evaluation/eval_{str(anchor)}.faiss")
    
        
def evaluate_top_k(k: int=3):
    """
    
    """
    print(k)
    metrics = dict()
    with open("evaluation/qa_pairs_relevance.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

        for anchor, pairs in qa_pairs.items():
            metrics[anchor]= {
                "dcg@k": list(),
                "ndcg@k": list(),
                "precision@k": list(),
                "recall@k": list(),
                "relevant_similarity": list(),
                "overall_similarity": list()
            }
            faiss_functional = FAISSFunctional()
            faiss_functional.load(f"evaluation/eval_{str(anchor)}.faiss")
            
            for pair in pairs:    
                query = embedding_model.encode([pair["question"]], convert_to_numpy=True)
                D, I = faiss_functional.query(query, k)
                tp = 0
                total_relevant = len(pair["relevant_chunks"])
                for sim, i in zip(D[0], I[0]):
                    doc_id, chunk_id = extract_doc_id_chunk_idx(i)
                    if doc_id == pair["doc_id"] and chunk_id in pair["relevant_chunks"]:
                        tp+=1
                        metrics[anchor]["relevant_similarity"].append(sim)
                
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

df_3 = evaluate_top_k(3)
print(df_3)
df_5 = evaluate_top_k(5)
print(df_5)
df_10 = evaluate_top_k(10)
print(df_10)

df = pd.concat([df_3, df_5, df_10])
print(df)
df.to_csv("evaluation/custom_eval.csv")







