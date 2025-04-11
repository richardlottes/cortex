import uuid
import streamlit as st

from utils.processing import (
    # add_chunks,
    # chunk_text,
    delete_document,
    # mistral_pdf_ocr,
    # create_title,
    # repunctuate,
    reset_storage,
    # retrieve_youtube_title,
    # retrieve_youtube_transcript,
    DocumentManager
)
from utils.streamlit_helpers import get_db, get_index, display_pdf, st_success_reset, st_failure_reset
from utils.config import schemas, dim, embed_model

#Initialize session ID and storage names inline; can't do this in a function as functions run after widget rehydration
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
db_name = f"""{st.session_state["session_id"]}.db"""
vector_index_name = f"""{st.session_state["session_id"]}.faiss"""

if "faiss_indexes" not in st.session_state:
    st.session_state["faiss_indexes"] = {} #maps index name -> FAISSFunctional
faiss_index = get_index(vector_index_name, dim)

if "sqlite_dbs" not in st.session_state:
    st.session_state["sqlite_dbs"] = {} #maps index name -> SQLiteFunctional
sqlite_db = get_db(db_name, schemas)

#Manually force resetting keys inline on each rerun to avoid duplication bugs; can't do this in a function as functions run after widget rehydration
keys = ["doc", "yt", "text"]
for key in keys:
    if f"{key}_upload_counter" not in st.session_state:
        st.session_state[f"{key}_upload_counter"] = 0
    if st.session_state.get(f"reset_{key}", False):
        st.session_state[f"{key}_upload_counter"] += 1
        st.session_state[f"reset_{key}"] = False

doc_upload_key = f"doc_upload_{st.session_state['doc_upload_counter']}"
yt_key = f"yt_upload_{st.session_state['yt_upload_counter']}"
text_key = f"text_upload_{st.session_state['text_upload_counter']}"

#Initialize sqlite DB to store chunk and document data
db_name = f"""{st.session_state["session_id"]}.db"""
sqlite_db = get_db(db_name)

#Initialize vector DB for RAG
vector_index_name = f"""{st.session_state["session_id"]}.faiss"""
faiss_index = get_index(vector_index_name, dim)

#Initialize document manager to process different types of docs
document_manager = DocumentManager(faiss_index, sqlite_db, embed_model)

#Upload and text-converted PDF 
uploaded_file = st.file_uploader("Save a file to Storage", type=["pdf", "txt"], key=doc_upload_key)
if uploaded_file is not None:
    try:
        #Write file to disk as bytes for later reading
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Processing PDF. This may take a minute...", show_time=True):
                document_manager.process_pdf(uploaded_file)
        else:
            with st.spinner("Processing TXT file. This may take a minute...", show_time=True):
                document_manager.process_txt(uploaded_file)
        st_success_reset("reset_doc")
    except Exception as e:
        st_failure_reset(e, "reset_doc")

#Transcribe YouTube video to text
with st.form("youtube_form"):
    text_input_raw =  st.text_input(
            "Save YouTube Video to Storage",
            key=yt_key,
        )
    yt_submitted = st.form_submit_button("Save Video")
    url = text_input_raw if text_input_raw.strip() != "" else None

    if yt_submitted and url is not None:
        try:
            with st.spinner("Transcribing YouTube video. This may take a minute...", show_time=True):
                document_manager.process_youtube(url)
            st_success_reset("reset_yt")
        except Exception as e:
            st_failure_reset(e, "reset_yt")

#Process raw text
with st.form("text_form"):
    text = st.text_input(
           "Save Text to Storage",
            key="txt_input",
        )
    txt_submitted = st.form_submit_button("Save Text")
    if txt_submitted and text is not None:
        try:
            with st.spinner("Adding text to storage. This may take a minute...", show_time=True):
                document_manager.process_text(text)
            st_success_reset("reset_text")
        except Exception as e:
            st_failure_reset(e, "reset_text")

#Reset storage 
if st.button("🧹 Reset Storage"):
    reset_storage(db_name, schemas, vector_index_name, faiss_index)

#Display storage
st.markdown("## 📚 Your Storage")
for doc_uuid, name, url, text, doc_type in sqlite_db.execute_query(
                                            """SELECT uuid, name, url, text, type 
                                               FROM documents 
                                               ORDER BY rowid DESC"""
                                            ):
    with st.expander(f"{'📄' if doc_type in ['.pdf','.txt','text'] else '📺'} {name} ({doc_type.title()})", expanded=False):
        if doc_type == ".pdf":
            display_pdf(name)  #Assumes the file is still saved by its original name
        elif doc_type == ".txt":
            st.markdown("**Plain Text File**")
        elif doc_type == "youtube":
            st.video(url)  #`name` here was stored as the video link
        elif doc_type == "text":
            st.markdown("**Raw Text Input**")
        st.markdown("**Extracted Text:**")
        st.text_area("Text", text, height=300, key=doc_uuid)
        if st.button("❌ Delete Document", key=f"delete_{doc_uuid}"):
            # Delete 
            delete_document(doc_uuid, faiss_index, sqlite_db)
            st.success(f"Deleted Document: {name}")
            st.rerun()
