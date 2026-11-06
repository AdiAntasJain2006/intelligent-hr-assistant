# HR Assistant

An AI-powered assistant for answering employee and recruitment questions from an HR knowledge base. It retrieves relevant passages from `data/HR_Data.txt` before generating an answer, and checks both incoming questions and generated answers with safety guardrails.

> **Important:** This project is a demonstration, not an official HR channel. Answers depend on the contents of the configured knowledge base and may not reflect current company policy. Confirm important or sensitive matters with authorized HR personnel.

## Features

- Ask HR questions in a Streamlit chat interface or an interactive command-line app.
- Retrieve relevant knowledge-base passages using Jina embeddings and Qdrant Cloud.
- Use a Groq-hosted language model to generate answers grounded in retrieved information.
- Apply input and output safety checks, including checks for prompt injection and sensitive personal information.
- Optionally trace model and agent activity with LangSmith.
- Run a LangSmith evaluation for answer correctness and retrieval groundedness.

## How it works

1. The assistant loads `data/HR_Data.txt` and splits it into overlapping text chunks.
2. Jina creates embeddings for the chunks, which are stored in the configured Qdrant collection. An existing collection is reused.
3. Before answering, the assistant checks the question with a safety model.
4. The agent retrieves relevant chunks and uses them as context for its response.
5. A second safety check runs before the response is returned.

## Requirements

- Python 3.10 or later
- A [Groq](https://console.groq.com/) API key
- A [Jina AI](https://jina.ai/) API key
- A Qdrant Cloud cluster, API key, and collection name
- Optional: a [LangSmith](https://smith.langchain.com/) API key for tracing and evaluation

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and fill in the required values:

```dotenv
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_CLUSTER_ENDPOINT=https://your-cluster-endpoint
QDRANT_COLLECTION_NAME=hr-assistant
```

To enable LangSmith tracing, add:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=hr-assistant
```

`LANGSMITH_ENDPOINT` can also be set if you use a non-default LangSmith endpoint. Keep `.env` private; do not commit API keys.

## Run

Start the web app:

```powershell
streamlit run app.py
```

Or start the interactive CLI:

```powershell
python main.py
```

On its first run, the assistant creates and populates the configured Qdrant collection from `data/HR_Data.txt`. Later runs connect to the existing collection.

## Evaluation

The evaluation uses a LangSmith dataset and LLM judges to assess correctness and groundedness. Configure LangSmith credentials, then run:

```powershell
python evaluation.py
```

## Project structure

```text
.
├── app.py                    # Streamlit chat interface
├── main.py                   # Interactive CLI
├── evaluation.py             # LangSmith evaluation entry point
├── requirements.txt
├── data/
│   └── HR_Data.txt           # HR knowledge base
└── hr_assistant/
    ├── config.py
    ├── pipeline.py           # Assistant and retrieval setup
    ├── document_loader.py
    ├── splitter.py
    ├── embeddings.py
    ├── vector_store.py
    ├── tools.py
    ├── agent.py
    ├── guardrails.py
    └── run_eval.py
```

## Configuration notes

Model names, chunking settings, retrieval count, and the system prompt are defined in `hr_assistant/config.py`. The default pipeline retrieves up to five chunks per question. Update the knowledge-base document and configuration to fit your authorized use case.
