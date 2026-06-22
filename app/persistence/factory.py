"""
Factory: return the appropriate SessionRepository based on DB_BACKEND config.
"""
from app.config import settings
from app.persistence.base import SessionRepository

_repo: SessionRepository | None = None


def get_repo() -> SessionRepository:
    global _repo
    if _repo is not None:
        return _repo

    backend = settings.db_backend.lower()
    if backend == "sqlite":
        from app.persistence.sqlite_impl import SqliteSessionRepository
        _repo = SqliteSessionRepository()
    elif backend == "postgres":
        from app.persistence.postgres_impl import PostgresSessionRepository
        _repo = PostgresSessionRepository()
    else:
        raise ValueError(f"Unknown DB_BACKEND: {backend} (use 'sqlite' or 'postgres')")

    return _repo
