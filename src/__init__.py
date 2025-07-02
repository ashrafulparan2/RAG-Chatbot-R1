"""
RAG Chatbot Package
Multi-user, multi-session chatbot with local LLM and vector database
"""

__version__ = "1.0.0"
__author__ = "RAG Chatbot Team"

from .llm_manager import LLMManager
from .vector_db import VectorDatabase
from .session_manager import SessionManager, ChatMessage, ChatSession
from .rag_pipeline import RAGPipeline

__all__ = [
    "LLMManager",
    "VectorDatabase", 
    "SessionManager",
    "ChatMessage",
    "ChatSession",
    "RAGPipeline"
]
