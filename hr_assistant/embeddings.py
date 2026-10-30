"""Step3 - converting chunks into embeddings"""

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_embedding_model():
    """Return JINA embedding model. Reads JINA api key from the env because in config env is already loaded"""
    logger.info("Initializing embeddings model '%s'", config.Embeddings_model)
    return JinaEmbeddings(model_name = config.Embeddings_model)