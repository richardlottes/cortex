import os
import uuid
from typing import Optional, Tuple, List, Any

import numpy as np
import faiss
import sqlite3


class FAISSFunctional:
    """
    Lightweight wrapper around FAISS to enhance usability in production-grade environments in which you may need to map UUIDs to internal FAISS IDs.
    """
    
    def __init__(self, embedding_dim: int=384, embeddings: np.ndarray=None, filename: str=None):
        self.index: faiss.IndexIDMap = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings[0].shape[0])) if embeddings is not None else faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        self.dim: int = embeddings[0].shape[0] if embeddings is not None else embedding_dim
        self.filename: Optional[str] = filename if filename else None


    def __repr__(self) -> str:

        return f"<FAISSFunctional ntotal={len(self)}, dim={self.dim}, file='{self.filename}'>"


    def __len__(self) -> int:

        return self.index.ntotal


    def __add__(self, other: "FAISSFunctional") -> "FAISSFunctional":

        if not isinstance(other, FAISSFunctional):
            raise ValueError("Can only merge with another FAISSFunctional object.")
        if self.dim != other.dim:
            raise ValueError("Embedding dimensions must match to merge indexes.")
        
        merged = FAISSFunctional(embedding_dim=self.dim)
        
        for faiss_id, emb in {**self.id_to_emb, **other.id_to_emb}.items():
            merged.index.add_with_ids(emb.reshape(1,-1), np.array([faiss_id]))

        return merged
        

    def __radd__(self, other) -> "FAISSFunctional":

        return other.__add__(self)


    @property
    def has_file(self) -> bool:
        """
        Whether the object has a filename or not.
        """

        return self.filename is not None and os.path.exists(self.filename)

    
    @property
    def id_to_emb(self) -> dict:
        """
        Custom ID to embedding map.
        """

        #Retrieve IDs
        ntotal = self.index.ntotal
        ids = faiss.vector_to_array(self.index.id_map).reshape(-1)

        #Retrieve vectors
        embeddings = np.array([self.index.index.reconstruct(i) for i in range(ntotal)])
        return {faiss_id: emb for faiss_id, emb in zip(ids, embeddings)}


    def _autosave(self):
        """
        Internal function to automatically save the FAISS Index under its filename.
        """

        if not self.filename:
            print("Autosave skipped: no filename set.")
            return
        try:
            faiss.write_index(self.index, self.filename)
        except Exception as e:
            print(f"Autosave failed: {e}")
   

    def save(self, filename: str):
        """
        Save the FAISS index to disk.

        Parameters:
        ----------
        - filename (str): The filename that the the index will be stored on disk under.
        """

        faiss.write_index(self.index, filename)
        self.filename = filename
   

    def load(self, filename: str):
        """
        Load a saved FAISS index into FAISSFuncitonal object.

        Parameters:
        ----------
        - filename (str): The filename of the index stored on disk to be loaded into the FAISSFunctional object.
        """

        loaded_index = faiss.read_index(filename)
        self.index = loaded_index
        self.dim = self.index.d
        self.filename = filename


    def add_embs(self, embeddings: np.ndarray, return_ids: bool=False, autosave: bool=False, custom: Optional[List[str]]=None) -> Optional[Tuple[List[str], List[int]]]:
        """
        Adds embeddings to the FAISS index.

        Parameters:
        ----------
        - embeddings (np.ndarray): The embeddings to be added to the index.

        Returns:
        -------
        - uuid_strs (List[str]): List of unique UUID values.
        - faiss_ids (List[int]): List of unique FAISS-compatible ID values.
        - autosave (bool): A flag to automatically save the index.
        """
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1,-1)

        faiss.normalize_L2(embeddings) #Normalize embedding for cosine similarity lookups
        if custom:
            faiss_ids = custom
        else:
            uuid_strs = [str(uuid.uuid4()) for _ in range(len(embeddings))] #Build UUIDs
            faiss_ids = [int(uuid.UUID(uuid_str).int & ((1 << 63) -1)) for uuid_str in uuid_strs] #Convert UUIDs into FAISS compatible 63-bit integers ID with extremely low collision probability
             
        self.index.add_with_ids(embeddings, faiss_ids)

        if autosave:
            self._autosave()

        #Return UUIDs and IDs if specified
        if return_ids:
            return uuid_strs, faiss_ids


    def del_embs(self, ids: np.ndarray, autosave: bool=False):
        """
        Deletes embeddings from the FAISS index.

        Parameters:
        ----------
        - ids (np.ndarray): The embeddings to be added to the index.
        - autosave (bool): A flag to automatically save the index.
        """

        if isinstance(ids, int):
            ids = np.array([ids], dtype='int64')
        elif isinstance(ids, list):
            ids = np.array(ids, dtype='int64')
        elif isinstance(ids, np.ndarray):
            ids = ids.astype('int64')
        else:
            raise TypeError("ids must be int, list of ints, or np.ndarray")

        self.index.remove_ids(ids)

        if autosave:
            self._autosave()


    def reset_index(self, embedding_dim: int=384, embeddings: np.ndarray=None, filename: str= None, autosave: bool=False):
        """
        Reset the index and FAISSFunctional object.

        Parameters:
        ----------
        - embedding_dim (int): The dimensionality of the embeddings the FAISS Index will store.
        - embeddings (np.ndarray): The embeddings themselves. If provided, the dimensionality of the embeddings the FAISS Index will store will be implicitly assumed using them.
        - filename (str): The filename under which the index will be saved.
        - autosave (bool): A flag to automatically save the index.
        """
        
        if self.has_file:
            os.remove(self.filename) 

        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings[0].shape[0])) if embeddings else faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        self.dim = embeddings[0].shape[0] if embeddings is not None else embedding_dim
        self.filename = filename if filename else None

        if autosave:
            self._autosave()


    def query(self, query_embedding: np.ndarray, k: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cosine similarity query (not native to FAISS) that returns k corresponding distances and IDs.

        Parameters:
        ----------
        - query_embedding (np.ndarray): The embedding of the query.
        - k (int): The number of similar vectors to be retrieve.
        
        Returns:
        -------
        - D (np.ndarray): The distances of each returned embedding from the query embedding.
        - I (np.ndarray): The IDs of the embeddings returned.
        """

        #Clone the query embedding to avoid modifying the original
        query_emb_copy = query_embedding.copy()

        #Normalize for cosine similarity
        faiss.normalize_L2(query_emb_copy)

        #Perform search
        D, I = self.index.search(query_embedding, k)
        
        return D, I


    def thread_controlled_query(self, query_embedding: np.ndarray, n_threads:int = 1, k: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cosine similarity query (not native to FAISS) that returns k corresponding distances and IDs. Allows user to set threads to use during query execution in case of parallelism errors.
        
        Parameters:
        ----------
        - query_embedding (np.ndarray): The embedding of the query.
        - n_threads (int): The number of threads you want to limit FAISS to use during the query.
        - k (int): The number of similar vectors to be retrieve.
        
        Returns:
        -------
        - D (np.ndarray): The distances of each returned embedding from the query embedding.
        - I (np.ndarray): The IDs of the embeddings returned.
        """
        
        #Set thread limit for FAISS to avoid excessive multiprocessing
        original_thread_count = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(n_threads)  #Default to single thread to avoid multiprocessing issues
        
        try:
            #Clone the query embedding to avoid modifying the original
            query_emb_copy = query_embedding.copy()
            
            #Normalize for cosine similarity
            faiss.normalize_L2(query_emb_copy)
            
            #Perform search
            D, I = self.index.search(query_emb_copy, k)
            
            return D, I
        finally:
            #Restore original thread settings
            faiss.omp_set_num_threads(original_thread_count)


class SQLiteFunctional:
    """
    Lightweight wrapper around SQLite3 to increase ease of use.
    """

    def __init__(self, filename: str, schemas: Optional[List[str]]=None):
        self.filename = filename
        self.schemas = schemas

        if schemas is not None:
            self._init_db()


    def __repr__(self):
        return f"<SQLiteFunctional db='{self.filename}'>"


    def _init_db(self):
        """
        Initializes a SQLite DB.
        """
        
        #Connects to db unless one doesn't exist in which case one is created
        with sqlite3.connect(self.filename) as conn:
            #Create cursor object to execute SQL with
            cur = conn.cursor()

            if self.schemas is not None:
                #Create table if one doesn't already exist
                for schema in self.schemas:
                    cur.execute(schema)

            #Save DB
            conn.commit()


    def execute_cmd(self, command: str, objects: Tuple[Any, ...]):
        """
        Executes an arbitrary command.

        Parameters:
        ----------
        - command (str): The command to be executed.
        - objects (Tuple[any]): Any objects that will be dynamically used in the command.
        """

        #Connects to db unless one doesn't exist in which case one is created
        with sqlite3.connect(self.filename) as conn:
            cur = conn.cursor()
            cur.execute(command, objects)


    def execute_query(self, command: str, objects: Tuple[any]=()) -> List[Tuple[Any, ...]]:
        """
        Executes a query.

        Parameters:
        ----------
        - command (str): The query to be executed.
        - objects (Tuple[any]): Any objects that will be dynamically used in the command.

        Returns:
        -------
        - Query results.
        """
        
        #Connects to db unless one doesn't exist in which case one is created
        with sqlite3.connect(self.filename) as conn:
            cur = conn.cursor()
            cur.execute(command, objects)
            results = cur.fetchall()

        return results