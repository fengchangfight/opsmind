from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.persistence import get_repo
from app.api.auth import create_token

router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


@router.post("/login")
async def login(req: LoginRequest):
    repo = get_repo()
    user = repo.verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(
        user_id=user["user_id"],
        username=user["username"],
        role=user["role"],
    )
    return {
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
        },
    }


@router.get("/me")
async def me(request: "Request"):
    """Get current user from JWT token (middleware injects request.state.user)."""
    from fastapi import Request
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401)
    return {
        "user_id": user["sub"],
        "username": user["username"],
        "role": user["role"],
    }
