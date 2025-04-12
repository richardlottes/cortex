import os
import base64
from typing import List
from time import sleep

import streamlit as st

from utils.functionals import FAISSFunctional, SQLiteFunctional


def get_index(filename:str, dim: int=384) -> FAISSFunctional:
    """
    Returns existing vector index or creates and loads it

    Parameters:
    ----------
    - filename (str):
    - dim (int): 

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
    - filename (str):
    - schemas (List(str))

    Returns:
    -------
    - 
    """

    #Load DBs from session state
    dbs = st.session_state["sqlite_dbs"]
    print(dbs)
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
    - file_path (str):
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
    - reset_key (str): 
    """

    st.success("Upload saved to storage!")
    st.session_state[reset_key] = True
    st.rerun()


def st_failure_reset(e: Exception, reset_key: str):
    """
    Displays error message and reruns page.

    Parameters:
    ----------
    - e (Exception):
    - reset_key (str):
    """

    st.error(f"Upload failed: {e}")
    st.session_state[reset_key] = True
    sleep(5)
    st.rerun()