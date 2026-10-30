"""Step2 - Code to split loaded text"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def split_into_chunks(doc : str) -> list[str]:
    """This function splits our document into small overlapping chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = config.chunk_size, chunk_overlap = config.chunk_overlap)
    chunks = text_splitter.split_documents(documents=doc)
    logger.info("Split document(s) into %d chunk(s)", len(chunks))
    return chunks

