"""
Memory Management Module - Handles conversation history and context.

This module provides functionality for:
- Storing and managing conversation history
- Generating context from previous messages
- Managing conversation sessions
- Clearing and resetting memory
"""

from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import json
from .logging_config import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Manages conversation history for RAG chatbot.
    
    Stores user queries and assistant responses with timestamps.
    Can generate context from conversation history for LLM.
    """
    
    def __init__(self, max_history: int = 10, session_id: Optional[str] = None):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of message exchanges to keep in memory
            session_id: Optional session identifier for tracking conversations
        """
        self.max_history = max_history
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history: List[Dict[str, str]] = []
        self.session_start = datetime.now()
        
        logger.info("Initialized ConversationMemory with max_history=%d, session_id=%s", max_history, self.session_id)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata dict (e.g., model used, tokens, etc.)
        """
        if role not in ["user", "assistant"]:
            logger.warning("Invalid role: %s, must be 'user' or 'assistant'", role)
            return
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.history.append(message)
        logger.info("Added %s message (length: %d) to history. Total messages: %d", 
                   role, len(content), len(self.history))
        
        # Keep only last max_history*2 messages (user + assistant pairs)
        if len(self.history) > self.max_history * 2:
            removed_count = len(self.history) - (self.max_history * 2)
            self.history = self.history[removed_count:]
            logger.info("Pruned history: removed %d oldest messages", removed_count)
    
    def get_history_context(self, num_messages: int = 6) -> str:
        """
        Generate conversation context from recent messages.
        
        Useful for providing conversation context to LLM for better continuity.
        
        Args:
            num_messages: Number of recent messages to include (default: 6, i.e., 3 exchanges)
        
        Returns:
            Formatted string with recent conversation history
        """
        if not self.history:
            logger.debug("No history available for context generation")
            return ""
        
        recent_messages = self.history[-num_messages:]
        logger.debug("Generating context from %d recent messages", len(recent_messages))
        
        context_parts = []
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content'][:200]}...")  # Truncate long messages
        
        context = "\n".join(context_parts)
        logger.debug("Generated context (length: %d)", len(context))
        
        return context
    
    def get_recent_user_queries(self, num_queries: int = 3) -> List[str]:
        """
        Get recent user queries from history.
        
        Args:
            num_queries: Number of recent queries to retrieve
        
        Returns:
            List of recent user queries
        """
        queries = [msg["content"] for msg in self.history if msg["role"] == "user"]
        recent = queries[-num_queries:] if len(queries) > num_queries else queries
        logger.debug("Retrieved %d recent user queries", len(recent))
        return recent
    
    def get_last_exchange(self) -> Optional[Dict]:
        """
        Get the last user-assistant exchange.
        
        Returns:
            Dict with 'user_message' and 'assistant_response', or None if no exchange
        """
        # Find last user message and its following assistant response
        user_idx = None
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "user":
                user_idx = i
                break
        
        if user_idx is None:
            logger.debug("No user message found in history")
            return None
        
        # Find assistant response after user message
        assistant_response = None
        for i in range(user_idx + 1, len(self.history)):
            if self.history[i]["role"] == "assistant":
                assistant_response = self.history[i]
                break
        
        if not assistant_response:
            logger.debug("No assistant response found after last user message")
            return None
        
        exchange = {
            "user_message": self.history[user_idx]["content"],
            "assistant_response": assistant_response["content"],
            "timestamp": assistant_response["timestamp"]
        }
        
        logger.debug("Retrieved last exchange")
        return exchange
    
    def clear(self) -> None:
        """Clear all conversation history."""
        cleared_count = len(self.history)
        self.history.clear()
        logger.info("Cleared conversation history (%d messages removed)", cleared_count)
    
    def get_full_history(self) -> List[Dict[str, str]]:
        """
        Get complete conversation history.
        
        Returns:
            List of message dicts
        """
        logger.debug("Retrieved full history (%d messages)", len(self.history))
        return self.history.copy()
    
    def get_message_count(self) -> Dict[str, int]:
        """
        Get counts of different message types.
        
        Returns:
            Dict with 'user', 'assistant', and 'total' counts
        """
        user_count = len([m for m in self.history if m["role"] == "user"])
        assistant_count = len([m for m in self.history if m["role"] == "assistant"])
        total_count = len(self.history)
        
        stats = {
            "user": user_count,
            "assistant": assistant_count,
            "total": total_count
        }
        
        logger.debug("Message counts: %s", stats)
        return stats
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export conversation history to JSON file.
        
        Args:
            filepath: Path to save JSON file. If None, uses default logs directory.
        
        Returns:
            Path to saved JSON file
        """
        if filepath is None:
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            filepath = log_dir / f"conversation_{self.session_id}.json"
        
        export_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_messages": len(self.history),
            "history": self.history
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info("Exported conversation history to %s", filepath)
            return str(filepath)
        except Exception as e:
            logger.error("Failed to export conversation history: %s", e, exc_info=True)
            raise
    
    def import_from_json(self, filepath: str) -> None:
        """
        Import conversation history from JSON file.
        
        Args:
            filepath: Path to JSON file to import
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.history = data.get("history", [])
            self.session_id = data.get("session_id", self.session_id)
            
            logger.info("Imported %d messages from %s", len(self.history), filepath)
        except Exception as e:
            logger.error("Failed to import conversation history: %s", e, exc_info=True)
            raise
    
    def get_session_stats(self) -> Dict:
        """
        Get statistics about current conversation session.
        
        Returns:
            Dict with session metrics
        """
        session_duration = datetime.now() - self.session_start
        message_counts = self.get_message_count()
        
        stats = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "duration_seconds": session_duration.total_seconds(),
            "total_messages": message_counts["total"],
            "user_messages": message_counts["user"],
            "assistant_messages": message_counts["assistant"],
            "avg_message_length": (sum(len(m["content"]) for m in self.history) / len(self.history)) if self.history else 0
        }
        
        logger.debug("Generated session stats: %s", stats)
        return stats


class MemoryManager:
    """
    Manages multiple conversation sessions and memory storage.
    
    Can store and retrieve different conversation memories by session ID.
    """
    
    def __init__(self, storage_dir: str = "./data/memories"):
        """
        Initialize memory manager.
        
        Args:
            storage_dir: Directory to store memory files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, ConversationMemory] = {}
        
        logger.info("Initialized MemoryManager with storage: %s", self.storage_dir)
    
    def create_session(self, session_id: Optional[str] = None, max_history: int = 10) -> ConversationMemory:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional session ID (auto-generated if not provided)
            max_history: Maximum history size for this session
        
        Returns:
            ConversationMemory instance for the session
        """
        memory = ConversationMemory(max_history=max_history, session_id=session_id)
        self.sessions[memory.session_id] = memory
        
        logger.info("Created new session: %s", memory.session_id)
        return memory
    
    def get_session(self, session_id: str) -> Optional[ConversationMemory]:
        """Get existing session by ID."""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session from memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("Deleted session: %s", session_id)
    
    def list_sessions(self) -> List[str]:
        """Get list of all active session IDs."""
        return list(self.sessions.keys())
    
    def export_all_sessions(self) -> Dict[str, str]:
        """
        Export all sessions to JSON files.
        
        Returns:
            Dict mapping session_id to file path
        """
        results = {}
        for session_id, memory in self.sessions.items():
            try:
                filepath = self.storage_dir / f"session_{session_id}.json"
                saved_path = memory.export_to_json(str(filepath))
                results[session_id] = saved_path
            except Exception as e:
                logger.error("Failed to export session %s: %s", session_id, e)
        
        logger.info("Exported %d sessions", len(results))
        return results


# Example usage and testing
if __name__ == "__main__":
    # Create a memory instance
    memory = ConversationMemory(max_history=5)
    
    # Add some messages
    memory.add_message("user", "What is the capital of France?")
    memory.add_message("assistant", "The capital of France is Paris.")
    
    memory.add_message("user", "What is its population?")
    memory.add_message("assistant", "Paris has a population of approximately 2.2 million people.")
    
    # Get statistics
    print("\n=== Memory Statistics ===")
    print(f"Session ID: {memory.session_id}")
    print(f"Message Count: {memory.get_message_count()}")
    print(f"Session Stats: {memory.get_session_stats()}")
    
    # Get context
    print("\n=== Conversation Context ===")
    print(memory.get_history_context())
    
    # Export
    print("\n=== Export ===")
    filepath = memory.export_to_json()
    print(f"Exported to: {filepath}")
