"""
Persistence layer: abstract Repository pattern, SQLite + PostgreSQL backends.
Usage:
    from app.persistence.factory import get_repo
    repo = get_repo()
    repo.init()
    sid = repo.create_session()
    repo.save_message(sid, "user", "Hello")
"""
from app.persistence.base import SessionRepository
from app.persistence.factory import get_repo

__all__ = ["SessionRepository", "get_repo"]
