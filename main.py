import sys
import os
import uuid

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.streamlit_helpers import get_index, get_db, view_context, view_message_history
from utils.processing import openai_message_template, llm_stream
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

###State management###

#Initialize session ID and storage names
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
db_name = f"""{st.session_state["session_id"]}.db"""
vector_index_name = f"""{st.session_state["session_id"]}.faiss"""

#
if "faiss_indexes" not in st.session_state:
    st.session_state["faiss_indexes"] = {} #maps index name -> FAISSFunctional
faiss_index = get_index(vector_index_name, dim)

#
if "sqlite_dbs" not in st.session_state:
    st.session_state["sqlite_dbs"] = {} #maps index name -> SQLiteFunctional
sqlite_db = get_db(db_name, schemas)

#
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = load_embed_model()

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = list()

#
if "model_selector" not in st.session_state:
    st.session_state["model_selector"] = list(MODELS.keys())[0]

#
if "show_rag_context" not in st.session_state:
    st.session_state["show_rag_context"] = False

#Accept inputs
prompt = st.chat_input("What's up?")

#Add model selector and toggle to show retrieved context from RAG in sidebar + display message history
processing = prompt is not None #Flag to enable/disables during query processing

#Model selector - Streamlit-managed persistence
model_selector = st.sidebar.selectbox(
    "Select a model",
    options=list(MODELS.keys()),
    index=list(MODELS.keys()).index(st.session_state["model_selector"]),
    disabled=processing
)
st.session_state["model_selector"] = model_selector 

#Context toggle - Streamlit-managed persistence
show_rag_context = st.sidebar.toggle(
    "Show retrieved context",
    value=st.session_state["show_rag_context"],
    disabled=processing
)
st.session_state["show_rag_context"] = show_rag_context
st.sidebar.markdown("---")

view_message_history(st.session_state["messages"], st.session_state["show_rag_context"])

if prompt is not None:
    #Add user message to chat history
    user_message = openai_message_template("user", prompt)
    st.session_state["messages"].append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    #RAG - consider only doing this if there's been an upload
    with st.spinner("Retrieving context...", show_time=True):
        #Embed user prompt and query FAISS Index
        q_emb = st.session_state["embedding_model"].encode([prompt], convert_to_numpy=True)
        D, I = faiss_index.thread_controlled_query(q_emb, k=5)

        #Use retrieved IDs to query SQLite DB for text and save result in session state so it's available in global context
        query_I = tuple([str(i) for i in I[0]])
        placeholders = ",".join(["?"] * len(query_I))
        res = sqlite_db.execute_query(f"""
            SELECT text, vector_id
            FROM chunks
            WHERE vector_id in ({placeholders})
            """,
            query_I,)
        
        index_text_map = {vector_id: text for text, vector_id in res}
        ordered_rag_context = [
            {
                "index": index,
                "similarity": distance,
                "text": index_text_map[index]
            }
            for distance, index in zip(D[0], I[0]) if index != -1
        ]

        #Update context and system prompts
        if len(ordered_rag_context) > 0:
            context = "\n\n".join([f"[chunk-{i+1}] {chunk['text']}" for i, chunk in enumerate(ordered_rag_context)])
            sys_prompt = f"{SYS_TEMPLATE}\n\n{context}"
        else:
            sys_prompt = SYS_TEMPLATE

    #Prepend/replace system prompt with updated context from RAG and format for LLM
    st.session_state["messages"] = [openai_message_template("system", sys_prompt)] + [message for message in st.session_state["messages"] if message["role"] != "system"]
    messages = [
        {
            "role": message["role"],
            "content": message["content"]
        } for message in st.session_state["messages"]
    ]

    #Hit LLM endpoint with messages
    with st.chat_message("assistant"):
        #Display RAG context inline
        if st.session_state["show_rag_context"]:
            view_context(ordered_rag_context, len(st.session_state["messages"]))
        #Stream tokens
        response_placeholder = st.empty()
        output = ""
        for event in llm_stream(messages, llm=MODELS[st.session_state["model_selector"]], model=st.session_state["model_selector"]):
            output += event
            response_placeholder.markdown(output + "▌")            
    
    assistant_message = openai_message_template("assistant", output)
    assistant_message["context"] = ordered_rag_context

    #Append response to messages
    st.session_state["messages"].append(assistant_message)

    #Rerun app to enable model selector and RAG toggle
    st.rerun()

#Add session information to UI
with st.sidebar.expander("Session Info"):
    st.markdown(f"🔑 Session ID: `{st.session_state.session_id}`")
    st.markdown(f"🗂️ DB: `{db_name}`")
    st.markdown(f"💾 Index: `{vector_index_name}`")
