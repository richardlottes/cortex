import uuid

import streamlit as st
import pandas as pd

from utils.streamlit import interactive_line_chart

#Initialize session ID and storage names inline; can't do this in a function as functions run after widget rehydration
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
db_name = f"""{st.session_state["session_id"]}.db"""
vector_index_name = f"""{st.session_state["session_id"]}.faiss"""

#Add session information to UI
with st.sidebar.expander("Session Info"):
    st.markdown(f"🔑 Session ID: `{st.session_state.session_id}`")
    st.markdown(f"🗂️ DB: `{db_name}`")
    st.markdown(f"💾 Index: `{vector_index_name}`")

st.title("Retrieval Evaluation")
st.markdown("The following is an evaluation of the retrieval efficacy of a variety of chunk sizes.")
st.markdown("""
- **Precision@k:** The proportion of retrieved chunks at rank <= *k* that are relevant
- **Recall@k:** The proportion of all relevant chunks that are retrieved within the top *k*
- **Average Relevant Similarity:** The average cosine similarity between the query and the of relevant retrieved chunks in the *k* retrieved chunks
- **Average Overall Similarity:** The average cosine similarity between the query and all *k* retrieved chunks
- **DCG@k:** A score that rewards retrieving relevant chunks earlier in the ranked list, using a logarithmic discount
- **nDCG@k:** The DCG@k score normalized by the ideal ordering of relevant chunks - capped at 1.0 for perfect ranking
""")
eval = pd.read_csv("evaluation/custom_eval.csv", index_col=0)
col1, col2 = st.columns(2)

with col1:
    interactive_line_chart(eval, "k", "precision@k", "chunk_size", "Precision@k", "Chunk Size")
    interactive_line_chart(eval, "k", "relevant_similarity", "chunk_size", "Average Relevant Similarity", "Chunk Size")
    interactive_line_chart(eval, "k", "dcg@k", "chunk_size", "DCG@k", "Chunk Size")
with col2:
    interactive_line_chart(eval, "k", "recall@k", "chunk_size", "Recall@k", "Chunk Size")
    interactive_line_chart(eval, "k", "overall_similarity", "chunk_size", "Average Overall Similarity", "Chunk Size")    
    interactive_line_chart(eval, "k", "ndcg@k", "chunk_size", "nDCG@k", "Chunk Size")