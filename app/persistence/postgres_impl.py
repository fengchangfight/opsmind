"""
PostgreSQL implementation of SessionRepository.
Uses asyncpg for async access, suitable for production multi-node deployments.
Swap via: DB_BACKEND=postgres
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.persistence.base import SessionRepository
from app.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PostgresSessionRepository(SessionRepository):
    """
    PostgreSQL session store via asyncpg.
    
    Table schema (auto-created on init):
    
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title      TEXT DEFAULT '',
            status     TEXT DEFAULT 'active',
            user_id    TEXT DEFAULT 'default',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(session_id),
            role       TEXT NOT NULL,
            content    TEXT NOT NULL DEFAULT '',
            citations  JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.postgres_dsn
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)
        return self._pool

    async def _execute(self, query: str, *args):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def _fetch(self, query: str, *args):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def _fetchrow(self, query: str, *args):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    def init(self):
        import asyncio
        asyncio.get_event_loop().run_until_complete(self._async_init())

    async def _async_init(self):
        await self._execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title      TEXT DEFAULT '',
                status     TEXT DEFAULT 'active',
                user_id    TEXT DEFAULT 'default',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role       TEXT NOT NULL,
                content    TEXT NOT NULL DEFAULT '',
                citations  JSONB DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
        """)

    def create_session(self, user_id: str = "default") -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._create_session(user_id)
        )

    async def _create_session(self, user_id: str) -> str:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        await self._execute(
            "INSERT INTO sessions (session_id, user_id, created_at, updated_at) VALUES ($1,$2,$3,$3)",
            sid, user_id, _now(),
        )
        return sid

    def get_session(self, session_id: str) -> Optional[dict]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._get_session(session_id))

    async def _get_session(self, session_id: str) -> Optional[dict]:
        row = await self._fetchrow(
            "SELECT * FROM sessions WHERE session_id = $1", session_id,
        )
        return dict(row) if row else None

    def list_sessions(self, user_id: str = "default") -> list[dict]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._list_sessions(user_id)
        )

    async def _list_sessions(self, user_id: str) -> list[dict]:
        rows = await self._fetch(
            "SELECT * FROM sessions WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 50",
            user_id,
        )
        return [dict(r) for r in rows]

    def save_message(
        self, session_id: str, role: str, content: str,
        citations: list | None = None,
    ):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self._save_message(session_id, role, content, citations)
        )

    async def _save_message(self, session_id, role, content, citations):
        citations_json = json.dumps(citations or [])
        await self._execute(
            "INSERT INTO messages (session_id, role, content, citations, created_at) VALUES ($1,$2,$3,$4,$5)",
            session_id, role, content, citations_json, _now(),
        )
        await self._execute(
            "UPDATE sessions SET updated_at = $1 WHERE session_id = $2",
            _now(), session_id,
        )

    def get_messages(self, session_id: str) -> list[dict]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._get_messages(session_id)
        )

    async def _get_messages(self, session_id: str) -> list[dict]:
        rows = await self._fetch(
            "SELECT * FROM messages WHERE session_id = $1 ORDER BY id ASC",
            session_id,
        )
        results = []
        for r in rows:
            d = dict(r)
            d["citations"] = json.loads(d["citations"]) if d["citations"] else []
            results.append(d)
        return results

    def get_messages_for_llm(self, session_id: str) -> list[dict]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._get_messages_for_llm(session_id)
        )

    async def _get_messages_for_llm(self, session_id: str) -> list[dict]:
        rows = await self._fetch(
            "SELECT role, content FROM messages WHERE session_id = $1 ORDER BY id ASC",
            session_id,
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def auto_title(self, session_id: str, first_message: str):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self._auto_title(session_id, first_message)
        )

    async def _auto_title(self, session_id: str, first_message: str):
        title = first_message.strip()[:60]
        if len(first_message.strip()) > 60:
            title += "..."
        await self._execute(
            "UPDATE sessions SET title = $1 WHERE session_id = $2",
            title, session_id,
        )

    def delete_session(self, session_id: str, user_id: str = ""):
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self._delete_session(session_id, user_id)
        )

    async def _delete_session(self, session_id: str, user_id: str = ""):
        if user_id:
            await self._execute("DELETE FROM messages WHERE session_id = $1", session_id)
            await self._execute("DELETE FROM sessions WHERE session_id = $1 AND user_id = $2", session_id, user_id)
        else:
            await self._execute("DELETE FROM messages WHERE session_id = $1", session_id)
            await self._execute("DELETE FROM sessions WHERE session_id = $1", session_id)

    def verify_user(self, username: str, password: str) -> dict | None:
        import asyncio, hashlib
        return asyncio.get_event_loop().run_until_complete(self._verify_user(username, password))

    async def _verify_user(self, username: str, password: str) -> dict | None:
        import hashlib
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        row = await self._fetchrow(
            "SELECT user_id, username, display_name, role FROM users WHERE username = $1 AND password_hash = $2",
            username, pw_hash,
        )
        return dict(row) if row else None

    def get_user(self, user_id: str) -> dict | None:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._get_user(user_id))

    async def _get_user(self, user_id: str) -> dict | None:
        row = await self._fetchrow(
            "SELECT user_id, username, display_name, role FROM users WHERE user_id = $1",
            user_id,
        )
        return dict(row) if row else None
