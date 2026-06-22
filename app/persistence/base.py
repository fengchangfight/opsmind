"""
Abstract Session Repository interface.
Concrete implementations: SqliteSessionRepository, PostgresSessionRepository.
Pattern: Repository + Factory, config-switchable via DB_BACKEND setting.
"""
from abc import ABC, abstractmethod
from typing import Optional


class SessionRepository(ABC):
    """Abstract interface for session + message persistence."""

    @abstractmethod
    def init(self):
        """Initialize storage (create tables, run migrations)."""
        ...

    @abstractmethod
    def create_session(self, user_id: str = "default") -> str:
        """Create new session, return session_id."""
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session metadata."""
        ...

    @abstractmethod
    def list_sessions(self, user_id: str = "default") -> list[dict]:
        """List sessions for user, newest first."""
        ...

    @abstractmethod
    def save_message(
        self, session_id: str, role: str, content: str,
        citations: list | None = None,
    ):
        """Append a message to session."""
        ...

    @abstractmethod
    def get_messages(self, session_id: str) -> list[dict]:
        """Get all messages for session with full metadata."""
        ...

    @abstractmethod
    def get_messages_for_llm(self, session_id: str) -> list[dict]:
        """Get messages in LLM-compatible format [{role, content}, ...]."""
        ...

    @abstractmethod
    def auto_title(self, session_id: str, first_message: str):
        """Derive session title from first user message."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str):
        """Delete session and its messages."""
        ...
