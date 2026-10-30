"""Step1 - Read raw document from folder"""

from typing import Text
from langchain_community.document_loaders import TextLoader
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def load_document(file_path : str = config.file_path) -> list[str] :
    """Load a .txt file and return it as a list of LangChain Document objects."""
    logger.info("Loading document from '%s'", file_path)
    loader = TextLoader(file_path)
    docs = loader.load()
    logger.info("Loaded %d document(s)", len(docs))
    return docs
    