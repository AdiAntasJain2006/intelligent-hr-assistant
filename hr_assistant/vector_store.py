"""Step4 - Store cunks embeddings into vector store"""

from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embedding_model
import os 
from hr_assistant.logger import get_logger
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient # python library to communicate to qdrant page

logger = get_logger(__name__)

# build vector store
def build_vec_store(chunks) :
    """Embed every chunk and upload it into a Qdrant Cloud collection."""
    logger.info(
        "Embedding %d chunk(s) and uploading to Qdrant collection '%s'...",
        len(chunks),
        config.QDRANT_COLLECTION_NAME,
    )
    embeddings_model = get_embedding_model()
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    logger.info("Uploaded to Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    return vector_store
    
# Load vector store
def load_vec_store(path : str = config.vec_store_path):
    """Connect to a Qdrant Cloud collection that was already built before."""
    logger.info("Connecting to existing Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    embeddings_model = get_embedding_model()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    
# Does vector store exists
def is_vector_store(path : str = config.vec_store_path) :
    """Check if FAISS exists in our memory"""
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)
    
def get_retriever(vector_store, top_k : int = config.top_k) :
    """Convert vector store into retriever that returns top k matching chunks"""
    logger.info("Retreival process starts")
    return vector_store.as_retriever(search_kwargs = {"k" : top_k})