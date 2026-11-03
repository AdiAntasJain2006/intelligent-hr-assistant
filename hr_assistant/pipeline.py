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
from hr_assistant.vector_store import build_vec_store, get_retriever, load_vec_store, is_vector_store
from hr_assistant.logger import get_logger
from hr_assistant.tracing import check_langsmith_tracing
from hr_assistant.guardrails import check_input, check_output, REFUSAL_MESSAGE

logger = get_logger(__name__)

# Data Ingestion and Vector Store Building

def build_vec_store_for_docs(file_path : str = config.file_path) :
    """Load + split + embed the document, 
    reusing the Qdrant Cloud collection if we have one."""
    if is_vector_store():
        print("Found an existing Qdrant Cloud collection, connecting to it (fast, no re-embedding).")
        logger.info("Qdrant Cloud collection already exists, reusing it")
        return load_vec_store()

    print("No Qdrant Cloud collection found, building one from scratch...")
    logger.info("No Qdrant Cloud collection found, building one from scratch")
    documents = load_document(file_path)
    chunks = split_into_chunks(documents)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store = build_vec_store(chunks)
    print("Vector store built and uploaded to Qdrant Cloud.")
    return vector_store

# Data Retrieval and Agent Building

def build_hr_assistant(file_path : str = config.file_path) :
    """Using this fxn we are trying to build agent with tools and brain"""
    config.check_api_key()
    check_langsmith_tracing()
    
    vector_store = build_vec_store_for_docs(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)
    
    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])
    
    return agent

def ask_agent(
    agent,
    question: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """This function is use to call agent and ask query"""
    logger.info("Generating response to HR assistant question")
    
    # applying input guard 
    input_is_safe, _ = check_input(question)
    if not input_is_safe :
        logger.warning("Input failed guard check")
        return REFUSAL_MESSAGE
    
    messages = list(chat_history or [])
    messages.append({"role": "user", "content": question})
    response = agent.invoke({"messages": messages})
    
    # applying output guard
    output_is_safe, _ = check_output(response['messages'][-1].content)
    if not output_is_safe :
        logger.warning("Output failed guard check")
        return REFUSAL_MESSAGE
     
    answer = response['messages'][-1].content
    logger.info("HR assistant response generated")
    return answer
    
    