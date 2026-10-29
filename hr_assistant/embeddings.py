"""Step3 - converting chunks into embeddings"""

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def get_embedding_model():
    """Return JINA embedding model. Reads JINA api key from the env because in config env is already loaded"""
    return JinaEmbeddings(model_name = config.Embeddings_model)