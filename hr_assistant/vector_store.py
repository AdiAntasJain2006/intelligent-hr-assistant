"""Step4 - Store cunks embeddings into vector store"""

from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embedding_model
import os 
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

# build vector store
def build_vec_store(chunks) :
    """ Convert text to vectors then storing in vector store FAISS"""
    embedding_model = get_embedding_model()
    return FAISS.from_documents(documents = chunks, embedding = embedding_model)

# Save vector store 
def save_vec_store(vec_store, path : str = config.vec_store_path) -> None:
    """ Saving vector store into disk memory"""
    vec_store.save_local(path)
    logger.info("Vector store saved successfully")
    
# Load vector store
def load_vec_store(path : str = config.vec_store_path) -> FAISS:
    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    logger.info("Vector store loaded successfully")
    # allow_dangerous_deserialization=True means ignore warnings 
    return vector_store
    
# Does vector store exists
def is_vector_store(path : str = config.vec_store_path) :
    """Check if FAISS exists in our memory"""
    return os.path.exists(os.path.join(path, "index.faiss"))
    
def get_retriever(vector_store, top_k : int = config.top_k) :
    """Convert vector store into retriever that returns top k matching chunks"""
    logger.info("Retreival process starts")
    return vector_store.as_retriever(search_kwargs = {"k" : top_k})