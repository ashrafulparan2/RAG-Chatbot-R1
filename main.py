import os
import sys
import logging
import time
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm_manager import LLMManager
from src.vector_db import VectorDatabase
from src.session_manager import SessionManager
from src.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot system"""
        logger.info("Initializing RAG Chatbot...")
        
        self.llm_manager = None
        self.vector_db = None
        self.session_manager = None
        self.rag_pipeline = None
        
        self.current_user_id = None
        self.current_session_id = None
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Loading LLM...")
            self.llm_manager = LLMManager()
            
            logger.info("Initializing vector database...")
            self.vector_db = VectorDatabase()
            
            knowledge_dir = "data/knowledge_base"
            if os.path.exists(knowledge_dir):
                self.vector_db.add_knowledge_from_directory(knowledge_dir)
            
            logger.info("Initializing session manager...")
            self.session_manager = SessionManager()
            
            logger.info("Initializing RAG pipeline...")
            self.rag_pipeline = RAGPipeline(
                vector_db=self.vector_db,
                llm_manager=self.llm_manager,
                session_manager=self.session_manager
            )
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def create_user_session(self, user_id: str) -> str:
        """Create a new session for a user"""
        session_id = self.session_manager.create_user_session(user_id)
        logger.info(f"Created new session {session_id} for user {user_id}")
        return session_id
    
    def chat(self, user_id: str, session_id: str, message: str) -> Dict[str, Any]:
        """
        Process a chat message and generate response
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            message: User message
              Returns:
            Response dictionary with bot reply and metadata
        """
        try:
            if not self.session_manager._validate_session(user_id, session_id):
                return {
                    "error": "Invalid session. Please start a new session.",
                    "response": None
                }
            
            result = self.rag_pipeline.generate_response(
                user_id=user_id,
                session_id=session_id,
                user_input=message
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your message."
            }
    
    def get_user_sessions(self, user_id: str) -> list:
        """Get all sessions for a user"""
        sessions = self.session_manager.get_user_sessions(user_id)
        return [session.to_dict() for session in sessions]
    
    def get_session_history(self, user_id: str, session_id: str) -> list:
        """Get chat history for a session"""
        messages = self.session_manager.get_session_history(user_id, session_id)
        return [msg.to_dict() for msg in messages]
    
    def shutdown(self):
        """Shutdown the chatbot system"""
        logger.info("Shutting down RAG Chatbot...")
        
        if self.session_manager:
            self.session_manager.save_all_data()
        
        logger.info("Shutdown complete")

def display_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("RAG CHATBOT")
    print("="*60)
    print("1. Start new chat session")
    print("2. Continue existing session")
    print("3. List user sessions")
    print("4. Switch user")
    print("5. Exit")
    print("="*60)

def main():
    try:
        chatbot = RAGChatbot()
        
        current_user_id = None
        current_session_id = None
        
        while True:
            if not current_user_id:
                current_user_id = input("\nEnter your User ID: ").strip()
                if not current_user_id:
                    print("User ID cannot be empty!")
                    continue
                print(f"Welcome, {current_user_id}!")
            
            display_menu()
            
            try:
                choice = input("\nSelect an option (1-5): ").strip()
                
                if choice == "1":
                    session_id = chatbot.create_user_session(current_user_id)
                    current_session_id = session_id
                    print(f"Started new session: {session_id}")
                    
                    print("\nChat mode activated. Type 'quit' to return to menu.")
                    while True:
                        user_input = input(f"\n{current_user_id}: ").strip()
                        if user_input.lower() in ['quit', 'exit', 'back']:
                            break
                        
                        if user_input:
                            result = chatbot.chat(current_user_id, current_session_id, user_input)
                            if result.get('error'):
                                print(f"Error: {result['error']}")
                            else:
                                print(f"Bot: {result['response']}")
                
                elif choice == "2":
                    sessions = chatbot.get_user_sessions(current_user_id)
                    if not sessions:
                        print("No existing sessions found. Create a new session first.")
                        continue
                    
                    print("\nYour sessions:")
                    for i, session in enumerate(sessions[:5]):
                        created_time = time.ctime(session['created_at'])
                        print(f"{i+1}. {session['session_id'][:8]}... (Created: {created_time})")
                    
                    try:
                        session_choice = int(input("Select session number: ")) - 1
                        if 0 <= session_choice < len(sessions):
                            current_session_id = sessions[session_choice]['session_id']
                            print(f"Resumed session: {current_session_id}")
                            
                            history = chatbot.get_session_history(current_user_id, current_session_id)
                            if history:
                                print("\nRecent messages:")
                                for msg in history[-10:]:
                                    role = "User" if msg['is_user'] else "Bot"
                                    timestamp = time.ctime(msg['timestamp'])
                                    print(f"[{timestamp}] {role}: {msg['content'][:100]}...")
                            
                            while True:
                                user_input = input(f"\n{current_user_id}: ").strip()
                                if user_input.lower() in ['quit', 'exit', 'back']:
                                    break
                                
                                if user_input:
                                    result = chatbot.chat(current_user_id, current_session_id, user_input)
                                    if result.get('error'):
                                        print(f"Error: {result['error']}")
                                    else:
                                        print(f"Bot: {result['response']}")
                        else:
                            print("Invalid session number")
                    except ValueError:
                        print("Please enter a valid number")
                
                elif choice == "3":
                    sessions = chatbot.get_user_sessions(current_user_id)
                    if sessions:
                        print(f"\nSessions for {current_user_id}:")
                        for session in sessions:
                            created_time = time.ctime(session['created_at'])
                            last_activity = time.ctime(session['last_activity'])
                            print(f"- {session['session_id'][:8]}... | Created: {created_time} | Last: {last_activity} | Messages: {session['message_count']}")
                    else:
                        print("No sessions found for this user")
                
                elif choice == "4":
                    current_user_id = None
                    current_session_id = None
                    print("Switched user. Please enter new User ID.")
                
                elif choice == "5":
                    print("\nGoodbye!\n")
                    chatbot.shutdown()
                    break
                
                else:
                    print("Invalid option. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!\n")
                chatbot.shutdown()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"An error occurred: {e}")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
