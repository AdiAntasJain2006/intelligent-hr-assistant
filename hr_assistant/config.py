"""All settings for our app live application, in one place"""

import os 
from dotenv import load_dotenv

load_dotenv()

# Environment variables 
groq_api_key = os.getenv("GROQ_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")

#Guard
GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

# VECTOR_STORE
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# TRACING
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

# Define path - vector_store or data 
file_path = os.path.join('data', 'HR_Data.txt')
vec_store_path = os.path.join('data', 'faiss_index')

# Models 
LLM_model = "openai/gpt-oss-120b"
Embeddings_model = "jina-embeddings-v4"

# Chunk / Splitter config
chunk_size = 500
chunk_overlap = 50

# Retrieval
top_k = 5

# System instructions
system_prompt = "You are a professional HR assistant who answers employee and recruitment questions accurately using the provided HR knowledge base, protects confidential information, never invents policies, and escalates sensitive or company-specific issues to authorized HR personnel."

def check_api_key() :
    if not groq_api_key :
        raise ValueError("Missing GROQ api key")
    if not jina_api_key :
        raise ValueError("Missing JINA api key")
