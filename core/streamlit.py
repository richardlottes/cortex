import os
import base64
from typing import List, Dict
from time import sleep

from textwrap import shorten
import streamlit as st
import pandas as pd
import plotly.express as px

from core.functionals import FAISSFunctional, SQLiteFunctional


def get_index(filename: str, dim: int=384) -> FAISSFunctional:
    """
    Returns existing vector index or creates and loads it.

    Parameters:
    ----------
    - filename (str): The file name of the FAISS index to be loaded.
    - dim (int): The dimensionality of the FAISS index to be loaded.

    Returns:
    -------
    - Instantiated FAISSFunctional object.
    """

    #Load indexes from session state (must be reloaded everytime - it doesn't update dynamically like a SQLite DB connection)
    idxs = st.session_state["faiss_indexes"]

    #Create a new index
    faiss_index = FAISSFunctional(embedding_dim=dim, filename=filename)
    
    #If the filename is in the indexes, load the most recent index; otherwise, save the newly instantiated index
    if filename in idxs:
        if os.path.exists(filename):
            faiss_index.load(filename)
    else:
        faiss_index.save(filename)

    #Set the new index and return it
    idxs[filename] = faiss_index

    return idxs[filename]


def get_db(filename: str, schemas: List[str]=None) -> SQLiteFunctional:
    """
    Returns existing DB or creates and loads it.

    Parameters:
    ----------
    - filename (str): The file name of the SQLite DB to connect to.
    - schemas (List(str)): The schemas to initialize; defaults to None.

    Returns:
    -------
    - Instantiated SQLiteFunctional object.
    """

    #Load DBs from session state
    dbs = st.session_state["sqlite_dbs"]
    
    #If the targeted DB doesn't exist then add it
    if filename not in dbs:
        if schemas is not None:
            dbs[filename] = SQLiteFunctional(filename, schemas)
        else:
            dbs[filename] = SQLiteFunctional(filename)

    #Return targeted DB
    return dbs[filename]


def display_pdf(file_path: str):
    """
    Displays PDF Documents.

    Parameters:
    ----------
    - file_path (str): The file path of the PDF to display.
    """

    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def st_success_reset(reset_key: str):
    """
    Displays success message and reruns page.

    Parameters:
    ----------
    - reset_key (str): The key that determines to create a new key for specific types of document ingestion.
    """

    st.success("Upload saved to storage!")
    st.session_state[reset_key] = True
    st.rerun()


def st_failure_reset(e: Exception, reset_key: str):
    """
    Displays error message and reruns page.

    Parameters:
    ----------
    - e (Exception): An Exception message.
    - reset_key (str): The key that determines to create a new key for specific types of document ingestion.
    """

    st.error(f"Upload failed: {e}")
    st.session_state[reset_key] = True
    sleep(5)
    st.rerun()


def view_context(context: List[Dict], message_n: int):
    if context:
        max_sim = max([chunk["similarity"] for chunk in context])
        if max_sim < 0.3:
            st.warning(f"Low confidence: Max similarity is {max_sim:.2f}")
        with st.expander("View context", expanded=False):
                for i, chunk in enumerate(context):
                    st.markdown(f"**Chunk {i+1} - Similarity: {chunk['similarity']:.2f}**")
                    preview = shorten(chunk["text"], width=300, placeholder="...")
                    st.text(preview)
                    if len(chunk["text"]) > 200:
                        if st.toggle(f"Show full Chunk {i+1}", key=f"message_{message_n}_chunk_{i}"):
                            st.markdown(chunk["text"])
    else:
        st.warning("Low confidence: No context retreived")


def view_message_history(messages: List[Dict], show_chunks: bool):
    """
    
    """

    for i, message in enumerate(messages):
        if message['role'] != "system":
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if show_chunks:

                        view_context(message.get("context", []), i)
                    st.markdown(message["content"][0]["text"])
                elif message["role"] == "user":
                    st.markdown(message["content"][0]["text"])


def interactive_line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, legend_title: str):
    """
    
    """

    fig = px.line(df, x=x, y=y, color=color)
    fig.update_layout(
        title=title,
        legend_title_text=legend_title
    )
    st.plotly_chart(fig)