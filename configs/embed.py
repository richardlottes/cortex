from sentence_transformers import SentenceTransformer

#Initialize embedding model
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")