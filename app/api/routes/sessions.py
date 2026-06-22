from fastapi import APIRouter
from app.persistence import get_repo

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
async def list_sessions(user_id: str = "default"):
    """List recent sessions."""
    repo = get_repo()
    sessions = repo.list_sessions(user_id)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a session with all messages."""
    repo = get_repo()
    session = repo.get_session(session_id)
    if not session:
        return {"error": "Session not found"}, 404
    messages = repo.get_messages(session_id)
    return {"session": session, "messages": messages}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its messages."""
    repo = get_repo()
    repo.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
