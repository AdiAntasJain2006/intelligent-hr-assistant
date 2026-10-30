"""Wires all the components together into one ready-to-use agent.

This is the single entry point that main.py (CLI) and app.py (Streamlit)
both call. Each step is handled by its own small module.
"""

from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.llm import get_llm
from hr_assistant.document_loader import load_document
from hr_assistant.splitter import split_into_chunks
from hr_assistant.tools import create_search_tool
from hr_assistant.vector_store import build_vec_store, save_vec_store, get_retriever, load_vec_store, is_vector_store
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def build_vec_store_for_docs(file_path : str = config.file_path) :
    """Vector store is being build inside this fxn"""
    if is_vector_store(config.vec_store_path) :
        logger.info("Found saved vector store")
        print("Found saved vector store on disk")
        return load_vec_store(config.vec_store_path)
    
    logger.info("No saved vector store found; building from source documents")
    print("No vector store saved")
    document = load_document(file_path)
    chunks = split_into_chunks(document)
    print(f"Data loaded and and split into {len(chunks)} chunks")
    
    vector_store = build_vec_store(chunks)
    save_vec_store(vector_store, config.vec_store_path)
    print("Vector store build and saved")
    return vector_store

def build_hr_assistant(file_path : str = config.vec_store_path) :
    """Using this fxn we are trying to build agent with tools and brain"""
    config.check_api_key()
    
    vector_store = build_vec_store_for_docs()
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)
    
    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])
    
    return agent

def ask_agent(agent, question : str) -> str :
    """This function is use to call agent and ask query"""
    logger.info("Generating response to HR assistant question")
    
    agent = build_hr_assistant()
    
    response = agent.invoke(
        {
            "messages" : [
                {
                    "role" : "user",
                    "content" :  question
                }
            ]
        }
    )
    answer = response['messages'][-1].content
    logger.info("HR assistant response generated")
    return answer
    
    