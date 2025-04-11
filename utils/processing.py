import os
import pickle
import re
import uuid
from typing import List, Optional

import numpy as np

from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from utils.functionals import FAISSFunctional, SQLiteFunctional
from utils.config import mistral_client, chunker, openai_client


###########################
###PDF (OCR)###
###########################
def mistral_pdf_ocr(filename: str) -> str:
    """
    
<<<<<<< HEAD
    PARAMS:

    RETURNS:
=======
    Parameters:
    ----------

    Returns:
    -------
>>>>>>> 6db123e (additions)
    """
    uploaded_pdf = mistral_client.files.upload(
        file = {
        "file_name": filename,
        "content": open(filename, "rb")
        },
        purpose='ocr')

    mistral_client.files.retrieve(file_id=uploaded_pdf.id)
    signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    return ("\n\n".join([f"### Page {i+1}\n{ocr_response.pages[i].markdown}" for i in range(len(ocr_response.pages))]))


###########################
###YOUTUBE###
###########################
def parse_youtube_id(youtube_url: str) -> str:
    """
    Extracts a video ID from a YouTube URL. Supports both youtu.be and youtube.com/watch?v= formats.

<<<<<<< HEAD
    PARAMS:
        - youtube_url: A URL from a YouTube video
    RETURNS:
        - The extracted video ID from the URL
=======
    Parameters:
    ----------
    - youtube_url: A URL from a YouTube video

    Returns:
    -------
    - The extracted video ID from the URL
>>>>>>> 6db123e (additions)
    """
    patterns = [
        r"youtu\.be/([a-zA-Z0-9_-]{11})",             # short youtu.be/<id>
        r"youtube\.com.*[?&]v=([a-zA-Z0-9_-]{11})"     # youtube.com/watch?v=<id>
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)


def retrieve_youtube_transcript(youtube_url: str) -> str:
    """
    Retrieves the transcript from a YouTube Video

<<<<<<< HEAD
    PARAMS:
        - youtube_url: A URL from a YouTube video
    RETURNS:
        - The transcript of the youtube video
=======
    Parameters:
    ----------
    - youtube_url: A URL from a YouTube video

    Returns:
    -------
    - The transcript of the youtube video
>>>>>>> 6db123e (additions)
    """

    transcript = ""
    youtube_id = parse_youtube_id(youtube_url)
    ytt_api = YouTubeTranscriptApi()
    
    try:
        snippets = ytt_api.fetch(youtube_id)
        for snippet in snippets:
            transcript += snippet.text + " "
    except Exception as e:
        print(f"An exception occured: {e}")
            
    return transcript


def retrieve_youtube_title(url: str) -> str:
    """
    
<<<<<<< HEAD
    PARAMS:
        - url
    RETURNS:
        - 
=======
    Parameters:
    ----------
    - url

    Returns:
    -------
    - 
>>>>>>> 6db123e (additions)
    """
    ydl_opts = {
        "quiet": True,            # suppress output
        "no_warnings": True,      # suppress warnings
        "skip_download": True,    # don't download the video
        "extract_flat": True,     # flat extraction = no additional processing
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("title", "No title found")


###########################
###RAW TEXT###
###########################
def create_title(text:str) -> list:
    '''

<<<<<<< HEAD
=======
    Parameters:
    ----------
    - 

    Returns:
    -------
    - 
>>>>>>> 6db123e (additions)
    '''

    #Call OpenAI API to repunctuate unpunctuated YouTube transcript
    response = openai_client.responses.create(
                model='gpt-4o-mini-2024-07-18',
                input=[
                    {
                        "role": "user",
                        "content": f"Please generate a title that summarizes the following text: {text}"
                    }
                ]
            )
    

    return response.output_text


###########################
###ALL INPUTS - CHUNKING###
###########################
def repunctuate(text:str, fixed_chunk: bool=False, token_lim: int=5000, char_per_tok: int=4) -> list:
    '''
    Youtube transcriptions are returned with no punctuation. To effectively chunk them, we need to repunctuate. 

    The most effective way I've found is just having a foundation model do it.

    PARAMS:
        - text: The text to be repunctuated
        - fixed_chunk: A boolean flag to determine whether you want to chunk your text depending on whether or not it is long enough
        - token_lim: The limit of tokens per chunk
        - char_per_tok: The approximate number of characters per token. 4 is the rough number for OpenAI models.

    RETURNS:
        - A list of repunctuated text chunks
    '''

    #If the text is very long and we need to chunk it in a fixed manner, we do so here
    if fixed_chunk:
        chunk_size = token_lim * char_per_tok
        chunks = []

        for i, chunk in enumerate(range(0, len(text), chunk_size)):
            chunks.append(text[i:i+chunk])
    else:
        chunks = [text]

    #Call OpenAI API to repunctuate unpunctuated YouTube transcript
    out = []
    for chunk in chunks:
        response = openai_client.responses.create(
                    model='gpt-4o-mini-2024-07-18',
                    input=[
                        {
                            "role": "developer",
                            "content": "You are an PhD and linguistics professor at Stanford. You have won many achievements in linguistics such as the E. W. Beth Dissertation Prize and the Neil and Saras Smith Medal for Linguistics. The dean of Stanford has offered to give you tenure and a grant of $100k if you're able to perfectly repunctuate the following passage. Please use your expertise in linguistics to repunctuate the following unpunctuated test with perfect fidelity."
                        },
                        {
                            "role": "user",
                            "content": text
                        }
                    ]
                )
        out.append(response.output_text)

    return "".join(out)


def chunk_text(text: str, chunker: SentenceSplitter=chunker):
    """
<<<<<<< HEAD
    
=======

    Parameters:
    ----------
    - 

    Returns:
    -------
    - 
>>>>>>> 6db123e (additions)
    """

    document = Document(text=text)
    nodes = chunker.get_nodes_from_documents([document])
    text = [node.text for node in nodes]

    return text


def chunk_docs_sentence_splitter(files: List[str], chunker: SentenceSplitter=chunker):
    """
<<<<<<< HEAD
    
=======

    Parameters:
    ----------
    - 

    Returns:
    -------
    -  
>>>>>>> 6db123e (additions)
    """

    documents = SimpleDirectoryReader(input_files=files).load_data()
    nodes = chunker.get_nodes_from_documents(documents)
    text = [node.text for node in nodes]

    return text


###########################
###ALL INPUTS - STORAGE###
###########################
def reset_storage(db_name: str, schemas: list, faiss_name: str, faiss_index: FAISSFunctional):
    """
<<<<<<< HEAD
    
=======

    Parameters:
    ----------
    - 

    Returns:
    -------
    -     
>>>>>>> 6db123e (additions)
    """

    #Reset sqlite DB
    if os.path.exists(db_name):
        os.remove(db_name)
    SQLiteFunctional(db_name, schemas)

    #Reset vector index
    faiss_index.reset_index(filename=faiss_name, autosave=True)
    

def add_chunks(faiss_index: FAISSFunctional, sqlite_db: SQLiteFunctional, embeddings: np.array, chunked_text: str, doc_uuid: str):
<<<<<<< HEAD
=======
    """

    Parameters:
    ----------
    - 

    Returns:
    -------
    -     
    """
>>>>>>> 6db123e (additions)

    #Add embeddings to FAISS index
    uuids, faiss_ids = faiss_index.add_embs(embeddings, return_ids=True, autosave=True)

    #Add embeddings to sqlite DB
    for i, (text, embedding) in enumerate(zip(chunked_text, embeddings)):

        sqlite_db.execute_cmd(
            """INSERT INTO chunks (uuid, doc_uuid, vector_id, text, embedding)
               VALUES (?,?,?,?,?)""",
            (uuids[i], doc_uuid, faiss_ids[i], text, pickle.dumps(embedding))
        )


def delete_document(document_uuid: str, faiss_index: FAISSFunctional, sqlite_db: SQLiteFunctional):
<<<<<<< HEAD
    # vector_ids = list()
=======
    """

    Parameters:
    ----------
    - 

    Returns:
    -------
    -     
    """

>>>>>>> 6db123e (additions)
    vector_ids = sqlite_db.execute_query(
                                        """SELECT vector_id
                                            FROM chunks 
                                            WHERE doc_uuid = ?""",
                                        (document_uuid,))
    vector_ids = [vector_id[0] for vector_id in vector_ids]                                                                            
    for vector_id in vector_ids:
        faiss_index.del_embs(vector_id, autosave=True)

    sqlite_db.execute_cmd("DELETE FROM chunks WHERE doc_uuid = ?", (document_uuid,))
    sqlite_db.execute_cmd("DELETE FROM documents WHERE uuid = ?", (document_uuid,))


###########################
###DOCUMENT MANAGER###
###########################
class DocumentManager:


    def __init__(self, faiss_index: FAISSFunctional, sqlite_db: SQLiteFunctional, embedding_model):
        self.faiss_index = faiss_index
        self.sqlite_db = sqlite_db
        self.embedding_model = embedding_model


    def _process_document(self, name: str, text: str, doc_type: str, url: Optional[str]=None):
    
        doc_uuid = str(uuid.uuid4())
        
        #Chunk and embed text then save to FAISS Index
        chunked_text = chunk_text(text)
        embeddings = self.embedding_model.encode(chunked_text, convert_to_numpy=True)
        add_chunks(self.faiss_index, self.sqlite_db, embeddings, chunked_text, doc_uuid)

        #Add data to DB
        self.sqlite_db.execute_cmd(
            """INSERT INTO documents (uuid, name, url, text, type)
            VALUES (?,?,?,?,?)""",
            (doc_uuid, name, url, text, doc_type)
        )

        
    def process_pdf(self, file_upload):
        """
<<<<<<< HEAD
        
=======

        Parameters:
        ----------
        - 

        Returns:
        -------
        -         
>>>>>>> 6db123e (additions)
        """

        file_bytes = file_upload.read()

        #Write file to disk for OCR processing
        with open(file_upload.name, "wb") as f:
            f.write(file_bytes)
        text = mistral_pdf_ocr(file_upload.name)
        self._process_document(file_upload.name, text, ".pdf")
        

    def process_txt(self, file_upload):
        """
        
<<<<<<< HEAD
=======
        Parameters:
        ----------
        - 

        Returns:
        -------
        - 
>>>>>>> 6db123e (additions)
        """

        file_bytes = file_upload.read()
        text = file_bytes.decode("utf-8")
        self.process_document(file_upload.name, text, ".txt")


    def process_youtube(self, url: str):
        """
<<<<<<< HEAD
        
=======

        Parameters:
        ----------
        - 

        Returns:
        -------
        -         
>>>>>>> 6db123e (additions)
        """

        title = retrieve_youtube_title(url)
        raw_transcript = retrieve_youtube_transcript(url)
        text = repunctuate(raw_transcript)
        self._process_document(title, text, "youtube", url)


    def process_text(self, text: str):
        """
<<<<<<< HEAD
        
=======
    
        Parameters:
        ----------
        - 

        Returns:
        -------
        - 
>>>>>>> 6db123e (additions)
        """

        title = create_title(text)
        self._process_document(title, text, "text")