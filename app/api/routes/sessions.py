from fastapi import APIRouter, Request, HTTPException
from app.persistence import get_repo

router = APIRouter(tags=["sessions"])


def _user_id(request: Request) -> str:
    uid = getattr(request.state, "user_id", None)
    if not uid:
        raise HTTPException(401)
    return uid


@router.get("/sessions")
async def list_sessions(request: Request):
    repo = get_repo()
    sessions = repo.list_sessions(_user_id(request))
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    repo = get_repo()
    session = repo.get_session(session_id)
    if not session or session.get("user_id") != _user_id(request):
        raise HTTPException(404, "Session not found")
    messages = repo.get_messages(session_id)
    return {"session": session, "messages": messages}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    repo = get_repo()
    repo.delete_session(session_id, _user_id(request))
    return {"status": "deleted", "session_id": session_id}
