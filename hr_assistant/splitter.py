"""Step2 - Code to split loaded text"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config

def split_into_chunks(doc : str) -> list[str]:
    """This function splits our document into small overlapping chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = config.chunk_size, chunk_overlap = config.chunk_overlap)
    return text_splitter.split_documents(documents=doc)

