# RAG Chatbot with Multiple User and Session Management

A prototype LLM-based RAG (Retrieval-Augmented Generation) chatbot that supports multiple users and continuous chat sessions.

## Features

- Multi-user and multi-session support
- Continuous conversation management
- Local LLM inference using Transformers
- Vector database for knowledge base
- Chat history storage in JSON format
- Clean responses without reasoning artifacts

## Components

1. **LLM**: DeepSeek R1 distilled model (local storage)
2. **Embeddings**: Sentence transformers for vector embeddings
3. **Vector Database**: FAISS for similarity search
4. **Storage**: JSON files for chat history
5. **Session Management**: In-memory session handling

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the chatbot:
```bash
python main.py
```

## Model and Resource Usage

1. "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" runs on CPU (12GB RAM), or T4 GPU (16GB VRAM)
2. "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" , "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" runs on L4 (24GB VRAM)

## Usage

The chatbot supports multiple users and chat sessions through a simple command-line interface. Each user can have multiple concurrent chat sessions.

## Project Structure

```
├── main.py                 # Main application entry point
├── src/
│   ├── llm_manager.py      # LLM model management
│   ├── vector_db.py        # Vector database operations
│   ├── session_manager.py  # User session management
│   └── rag_pipeline.py     # RAG retrieval pipeline
├── data/
│   ├── knowledge_base/     # Documents for RAG
│   ├── sessions/           # JSON session and chat logs
│   └── models/             # Local model storage
└── requirements.txt
```