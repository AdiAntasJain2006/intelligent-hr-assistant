"""Step1 - Read raw document from folder"""

from typing import Text
from langchain_community.document_loaders import TextLoader
from hr_assistant import config

def load_document(file_path : str = config.file_path) -> list[str] :
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs
    