import sys
import os
import uuid

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.streamlit_helpers import get_index, get_db
from utils.processing import llm_stream
from configs.schemas import schemas
from configs.llm import dim, MODELS, SYS_TEMPLATE
from configs.embed import load_embed_model


#Explicitly set to avoid forked process warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="Home"
)

st.title("Cortex")

with st.container():
    st.markdown("### 🧠 AI-Powered Knowledge Assistant")
    st.markdown(
        """
        This assistant uses a Retrieval-Augmented Generation (RAG) system to answer your questions using your uploaded documents.

        **How it works:**
        1. Upload documents on the "Storage" page.
        2. Ask questions below.
        3. Get context-aware answers powered by OpenAI + FAISS.

        _Your session is private and temporary._
        """
    )
    st.markdown("---")

#Initialize session ID and storage names
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

if "embedding_model" not in st.session_state:
    # embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    st.session_state["embedding_model"] = load_embed_model()

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

#
if "res" not in st.session_state:
    st.session_state["res"] = None


#Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message['role'] != "system":
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
            elif message["role"] == "user":
                st.markdown(message["content"][0]["text"])

#Accept inputs
if prompt := st.chat_input("What's up?"):
    #Model selector
    model = st.sidebar.selectbox("Select a model", MODELS.keys(), disabled=True)
    #Add checkbox to show retrieved context from RAG
    show_chunks = st.sidebar.toggle("Show retrieved context", value=False, disabled=True)


    #Add user message to chat history
    st.session_state["messages"].append(
        {
        "role":"user",
        "content":[
                {
                    "type": "input_text",
                    "text": prompt
                }
            ]
        }
    )
    
    #
    with st.chat_message("user"):
        st.markdown(prompt)

    #RAG
    with st.spinner("Retrieving context...", show_time=True):
        try:
            #Embed user prompt and query FAISS Index
            q_emb = st.session_state["embedding_model"].encode([prompt], convert_to_numpy=True)
            D, I = faiss_index.thread_controlled_query(q_emb, k=5)

            #Use retrieved IDs to query SQLite DB for text
            query_I = tuple([str(i) for i in I[0]])
            placeholders = ",".join(["?"] * len(query_I))
            res = sqlite_db.execute_query(f"""
                SELECT text
                FROM chunks
                WHERE vector_id in ({placeholders})
                """,
                query_I,)

            #
            st.session_state["res"] = res

            #Update context and system prompts
            context = "\n\n".join([f"[chunk-{i+1}] {chunk[0]}" for i, chunk in enumerate(res)])
            sys_prompt = f"{SYS_TEMPLATE}\n\n{context}"
        except Exception as e:
            sys_prompt = SYS_TEMPLATE
            st.error(f"No chunks retrieved during search")

    #Prepend/replace system prompt with updated context from RAG
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": sys_prompt
                }
            ]
        } 
    ] + st.session_state["messages"]

    #Hit LLM endpoint with messages
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        messages =  [
            {
                "role": message["role"],
                "content": message["content"]
            } for message in st.session_state["messages"]
        ]

        output = ""
        #Stream tokens
        for event in llm_stream(messages, llm=MODELS[model], model=model):
            output += event
            response_placeholder.markdown(output + "▌")            

    #Append response to messages
    st.session_state.messages.append({
        "role":"assistant",
        "content":output
    })
    st.rerun()
else:
    #Model selector
    model = st.sidebar.selectbox("Select a model", MODELS.keys(), disabled=False)
    #Add checkbox to show retrieved context from RAG
    show_chunks = st.sidebar.toggle("Show retrieved context", value=False, disabled=False)

#Show context chunks used in RAG
if show_chunks and st.session_state["res"] is not None:
    st.sidebar.markdown("### 📚 Context Chunks Used in Answer")
    for i, chunk in enumerate(st.session_state["res"]):
        with st.sidebar.expander(f"Chunk {i+1}", expanded=False):
            st.markdown(chunk[0])
            st.markdown("---")

st.sidebar.markdown("---")

#Add session information to UI
with st.sidebar.expander("Session Info"):
    st.markdown(f"🔑 Session ID: `{st.session_state.session_id}`")
    st.markdown(f"🗂️ DB: `{db_name}`")
    st.markdown(f"💾 Index: `{vector_index_name}`")
