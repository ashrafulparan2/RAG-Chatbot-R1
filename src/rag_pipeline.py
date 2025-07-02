"""
RAG Pipeline for combining retrieval and generation
"""
import logging
from typing import Optional, List, Dict, Any
from .vector_db import VectorDatabase
from .llm_manager import LLMManager
from .session_manager import SessionManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 vector_db: VectorDatabase,
                 llm_manager: LLMManager,
                 session_manager: SessionManager,
                 retrieval_threshold: float = 0.3,
                 max_context_length: int = 1000):
        """
        Initialize RAG pipeline
        
        Args:
            vector_db: Vector database for retrieval
            llm_manager: LLM manager for generation
            session_manager: Session manager for conversation context
            retrieval_threshold: Minimum similarity score for retrieval
            max_context_length: Maximum length of retrieved context
        """
        self.vector_db = vector_db
        self.llm_manager = llm_manager
        self.session_manager = session_manager
        self.retrieval_threshold = retrieval_threshold
        self.max_context_length = max_context_length
        
        logger.info("RAG Pipeline initialized")
    
    def generate_response(self, 
                         user_id: str,
                         session_id: str,
                         user_input: str,
                         use_retrieval: bool = True,
                         use_conversation_context: bool = True) -> Dict[str, Any]:
        """
        Generate a response using RAG pipeline
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_input: User's input message
            use_retrieval: Whether to use document retrieval
            use_conversation_context: Whether to include conversation history
              Returns:
            Dictionary containing response and metadata
        """
        try:
            retrieved_context = ""
            retrieval_results = []
            
            if use_retrieval:
                retrieval_results = self.vector_db.search(user_input, k=3)
                if retrieval_results:
                    relevant_docs = [doc for doc, score in retrieval_results 
                                   if score >= self.retrieval_threshold]
                    
                    if relevant_docs:
                        retrieved_context = "\n\n".join(relevant_docs)
                        if len(retrieved_context) > self.max_context_length:
                            retrieved_context = retrieved_context[:self.max_context_length] + "..."
            
            conversation_context = ""
            if use_conversation_context:
                conversation_context = self.session_manager.get_conversation_context(
                    user_id, session_id, max_messages=4
                )
            
            combined_context = self._combine_contexts(
                retrieved_context, conversation_context
            )
            
            response = self.llm_manager.generate_response(
                prompt=user_input,
                context=combined_context,
                max_new_tokens=256,
                temperature=0.7
            )
            user_msg_id = self.session_manager.add_message(
            user_id=user_id,
            session_id=session_id,
            content=user_input,
            is_user=True
        )
            
            bot_msg_id = self.session_manager.add_message(
                user_id=user_id,
                session_id=session_id,
                content=response,
                is_user=False,
                context_used=retrieved_context if retrieved_context else None
            )
            
            response_metadata = {
                "response": response,
                "user_message_id": user_msg_id,
                "bot_message_id": bot_msg_id,
                "retrieval_used": bool(retrieved_context),
                "conversation_context_used": bool(conversation_context),
                "retrieved_documents": len(retrieval_results),
                "retrieval_scores": [score for _, score in retrieval_results],
                "context_length": len(combined_context)
            }
            
            logger.debug(f"Generated response for user {user_id}, session {session_id}")
            return response_metadata
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "error": str(e),
                "retrieval_used": False,
                "conversation_context_used": False
            }
    
    def _combine_contexts(self, retrieved_context: str, conversation_context: str) -> str:
        """
        Combine retrieved and conversation contexts intelligently
        
        Args:
            retrieved_context: Context from document retrieval
            conversation_context: Context from conversation history
        Returns:
            Combined context string
        """
        contexts = []
        
        if conversation_context:
            contexts.append(f"Previous conversation:\n{conversation_context}")
        
        if retrieved_context:
            contexts.append(f"Relevant information:\n{retrieved_context}")
        return "\n\n".join(contexts)
    
    def add_knowledge(self, documents: List[str]):
        """
        Add new documents to the knowledge base
        
        Args:
            documents: List of document texts
        """
        try:
            self.vector_db.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to knowledge base")
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
    
    def search_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            results = self.vector_db.search(query, k)
            return [{"document": doc, "score": score} for doc, score in results]
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        try:
            return {
                "vector_db": self.vector_db.get_stats(),
                "llm": self.llm_manager.get_model_info(),
                "sessions": self.session_manager.get_stats(),
                "retrieval_threshold": self.retrieval_threshold,
                "max_context_length": self.max_context_length
            }
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {"error": str(e)}
    
    def update_settings(self, **kwargs):
        """
        Update pipeline settings
        
        Args:
            **kwargs: Settings to update (retrieval_threshold, max_context_length, etc.)
        """
        if 'retrieval_threshold' in kwargs:
            self.retrieval_threshold = kwargs['retrieval_threshold']
            logger.info(f"Updated retrieval threshold to {self.retrieval_threshold}")
        
        if 'max_context_length' in kwargs:
            self.max_context_length = kwargs['max_context_length']
            logger.info(f"Updated max context length to {self.max_context_length}")
