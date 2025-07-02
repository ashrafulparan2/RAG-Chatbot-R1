import uuid
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a single chat message"""
    message_id: str
    user_id: str
    session_id: str
    content: str
    is_user: bool  
    timestamp: float
    context_used: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Create ChatMessage from dictionary"""
        return cls(**data)

@dataclass 
class ChatSession:
    """Represents a chat session"""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    message_count: int = 0
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        """Create ChatSession from dictionary"""
        return cls(**data)

class SessionManager:
    def __init__(self, max_sessions_per_user: int = 10, session_timeout: int = 3600, 
                 data_dir: str = "data/sessions"):
        """
        Initialize the session manager
        
        Args:
            max_sessions_per_user: Maximum concurrent sessions per user
            session_timeout: Session timeout in seconds (default: 1 hour)
            data_dir: Directory to store session data
        """
        self.max_sessions_per_user = max_sessions_per_user
        self.session_timeout = session_timeout
        self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions_file = self.data_dir / "sessions.json"
        self.user_sessions_file = self.data_dir / "user_sessions.json"
        self.messages_dir = self.data_dir / "messages"
        self.messages_dir.mkdir(exist_ok=True)
        
        self.sessions: Dict[str, ChatSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}
        self.session_messages: Dict[str, List[ChatMessage]] = {}
        
        self._load_data()
        
        logger.info("Session Manager initialized with persistent storage")
    
    def create_user_session(self, user_id: str) -> str:
        """
        Create a new chat session for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            session_id: New session identifier
        """
        # Clean up expired sessions first
        self._cleanup_expired_sessions()
        
        # Check session limit for user
        if user_id in self.user_sessions:
            active_sessions = [sid for sid in self.user_sessions[user_id] 
                             if sid in self.sessions and self.sessions[sid].is_active]
            
            if len(active_sessions) >= self.max_sessions_per_user:
                # Deactivate oldest session
                oldest_session_id = min(active_sessions, 
                                      key=lambda sid: self.sessions[sid].last_activity)                
                self.sessions[oldest_session_id].is_active = False
                logger.info(f"Deactivated oldest session {oldest_session_id} for user {user_id}")
        
        session_id = str(uuid.uuid4())
        current_time = time.time()
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            message_count=0,
            is_active=True
        )
        
        self.sessions[session_id] = session
        self.session_messages[session_id] = []
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        self._save_sessions()
        self._save_user_sessions()
        self._save_session_messages(session_id)
        
        logger.info(f"Created new session {session_id} for user {user_id}")
        return session_id
    
    def add_message(self, user_id: str, session_id: str, content: str, 
                   is_user: bool, context_used: Optional[str] = None) -> str:
        """
        Add a message to a session
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            content: Message content
            is_user: True if user message, False if bot response
            context_used: RAG context used for the response
            
        Returns:
            message_id: Message identifier
        """
        if not self._validate_session(user_id, session_id):
            raise ValueError(f"Invalid session {session_id} for user {user_id}")
        
        message_id = str(uuid.uuid4())
        current_time = time.time()
        
        message = ChatMessage(
            message_id=message_id,
            user_id=user_id,
            session_id=session_id,
            content=content,
            is_user=is_user,
            timestamp=current_time,
            context_used=context_used
        )
        
        # Add message to session
        self.session_messages[session_id].append(message)
        
        # Update session metadata
        session = self.sessions[session_id]
        session.last_activity = current_time
        session.message_count += 1
        
        # Save to persistent storage
        self._save_sessions()
        self._save_session_messages(session_id)
        
        logger.debug(f"Added message {message_id} to session {session_id}")
        return message_id
    
    def get_session_history(self, user_id: str, session_id: str, 
                           limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get chat history for a session
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of messages to return (None for all)
            
        Returns:
            List of chat messages
        """
        if not self._validate_session(user_id, session_id):
            return []
        
        messages = self.session_messages.get(session_id, [])
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_context(self, user_id: str, session_id: str, 
                               max_messages: int = 6) -> str:
        """
        Get recent conversation context for the LLM
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        messages = self.get_session_history(user_id, session_id, max_messages)
        
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages:
            role = "User" if msg.is_user else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user's chat sessions
        """
        if user_id not in self.user_sessions:
            return []
        
        session_ids = self.user_sessions[user_id]
        sessions = []
        
        for session_id in session_ids:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.is_active:
                    sessions.append(session)
          # Sort by last activity (most recent first)
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions
    
    def _validate_session(self, user_id: str, session_id: str) -> bool:
        """
        Validate that a session belongs to a user and is active
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            True if session is valid
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        return (session.user_id == user_id and 
                session.is_active and
                not self._is_session_expired(session))
    
    def _is_session_expired(self, session: ChatSession) -> bool:
        """
        Check if a session has expired
        
        Args:
            session: Chat session
            
        Returns:
            True if session has expired
        """
        return (time.time() - session.last_activity) > self.session_timeout
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            session = self.sessions[session_id]
            session.is_active = False
            logger.info(f"Session {session_id} expired for user {session.user_id}")
        
        if expired_sessions:
            self._save_sessions()
    
    def get_stats(self) -> Dict:
        """Get session manager statistics"""
        active_sessions = sum(1 for s in self.sessions.values() if s.is_active)
        total_users = len(self.user_sessions)
        total_messages = sum(len(msgs) for msgs in self.session_messages.values())
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_users": total_users,
            "total_messages": total_messages,
            "sessions_per_user": {
                user_id: len([sid for sid in session_ids 
                            if sid in self.sessions and self.sessions[sid].is_active])
                for user_id, session_ids in self.user_sessions.items()
            }
        }
    def _load_data(self):
        """Load session data from persistent storage"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    sessions_data = json.load(f)
                    self.sessions = {
                        session_id: ChatSession.from_dict(data)
                        for session_id, data in sessions_data.items()
                    }
                logger.info(f"Loaded {len(self.sessions)} sessions from storage")
            
            if self.user_sessions_file.exists():
                with open(self.user_sessions_file, 'r', encoding='utf-8') as f:
                    self.user_sessions = json.load(f)
                logger.info(f"Loaded user sessions mapping for {len(self.user_sessions)} users")
            
            for session_id in self.sessions.keys():
                message_file = self.messages_dir / f"{session_id}.json"
                if message_file.exists():
                    with open(message_file, 'r', encoding='utf-8') as f:
                        messages_data = json.load(f)
                        self.session_messages[session_id] = [
                            ChatMessage.from_dict(msg_data)
                            for msg_data in messages_data
                        ]
                else:
                    self.session_messages[session_id] = []
            
            logger.info("Session data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            # Initialize empty data structures if loading fails
            self.sessions = {}
            self.user_sessions = {}
            self.session_messages = {}
    
    def _save_sessions(self):
        """Save sessions to persistent storage"""
        try:
            sessions_data = {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            }
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, indent=2, ensure_ascii=False)
            logger.debug("Sessions saved to storage")
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _save_user_sessions(self):
        """Save user sessions mapping to persistent storage"""
        try:
            with open(self.user_sessions_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_sessions, f, indent=2, ensure_ascii=False)
            logger.debug("User sessions mapping saved to storage")
        except Exception as e:
            logger.error(f"Error saving user sessions: {e}")
    
    def _save_session_messages(self, session_id: str):
        """Save messages for a specific session"""
        try:
            if session_id in self.session_messages:
                message_file = self.messages_dir / f"{session_id}.json"
                messages_data = [
                    msg.to_dict() for msg in self.session_messages[session_id]
                ]
                with open(message_file, 'w', encoding='utf-8') as f:
                    json.dump(messages_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Messages saved for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving messages for session {session_id}: {e}")
    
    def save_all_data(self):
        """Save all session data to persistent storage"""
        try:
            self._save_sessions()
            self._save_user_sessions()
            for session_id in self.session_messages.keys():
                self._save_session_messages(session_id)
            logger.info("All session data saved successfully")
        except Exception as e:
            logger.error(f"Error saving all session data: {e}")
    
    def __del__(self):
        """Cleanup method to save data when object is destroyed"""
        try:
            self.save_all_data()
        except:
            pass  # Ignore errors during cleanup
