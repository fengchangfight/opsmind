"""
SQLite implementation of SessionRepository.
Single-file, zero-dependency, WAL mode.
Shareable between SQLite and PostgreSQL via the SessionRepository interface.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.persistence.base import SessionRepository
from app.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteSessionRepository(SessionRepository):
    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or settings.sqlite_path)

    def _get_conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def init(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       TEXT PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name  TEXT NOT NULL DEFAULT '',
                role          TEXT NOT NULL DEFAULT 'user',
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title      TEXT DEFAULT '',
                status     TEXT DEFAULT 'active',
                user_id    TEXT NOT NULL REFERENCES users(user_id),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role       TEXT NOT NULL,
                content    TEXT NOT NULL DEFAULT '',
                citations  TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at);
        """)

        # Seed demo users (idempotent)
        import hashlib
        demos = [
            ("alice", "opsmind123", "Alice Wang", "sre"),
            ("bob",   "opsmind123", "Bob Chen",   "devops"),
        ]
        for username, password, display, role in demos:
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, username, password_hash, display_name, role, created_at) VALUES (?,?,?,?,?,?)",
                (username, username, pw_hash, display, role, _now()),
            )

        conn.commit()
        conn.close()

    def create_session(self, user_id: str = "default") -> str:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, user_id, _now(), _now()),
        )
        conn.commit()
        conn.close()
        return sid

    def get_session(self, session_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_sessions(self, user_id: str = "default") -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50",
            (user_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def save_message(
        self, session_id: str, role: str, content: str,
        citations: list | None = None,
    ):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, citations, created_at) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(citations or []), _now()),
        )
        conn.commit()
        # Touch session updated_at
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        conn.commit()
        conn.close()

    def get_messages(self, session_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            d["citations"] = json.loads(d["citations"])
            results.append(d)
        return results

    def get_messages_for_llm(self, session_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        conn.close()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def auto_title(self, session_id: str, first_message: str):
        title = first_message.strip()[:60]
        if len(first_message.strip()) > 60:
            title += "..."
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, session_id),
        )
        conn.commit()
        conn.close()

    def delete_session(self, session_id: str, user_id: str = ""):
        conn = self._get_conn()
        if user_id:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)",
                (session_id, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        else:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    def verify_user(self, username: str, password: str) -> dict | None:
        import hashlib
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT user_id, username, display_name, role FROM users WHERE username = ? AND password_hash = ?",
            (username, pw_hash),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_user(self, user_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT user_id, username, display_name, role FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None
