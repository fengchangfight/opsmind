"""
Simple JWT auth middleware. No external deps — uses HMAC-SHA256.
Demo-only; production would use PyJWT + OAuth2.
"""
import hashlib
import hmac
import json
import time
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config import settings

JWT_SECRET = "opsmind-demo-secret-key-2026"
AUTH_WHITELIST = ["/api/login", "/api/docs", "/api/openapi.json", "/health"]


def _b64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    import base64
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_token(user_id: str, username: str, role: str, ttl: int = 86400) -> str:
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(json.dumps({
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl,
    }).encode())
    signature = hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(signature)}"


def verify_token(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, sig = parts
        expected_sig = _b64url_encode(
            hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig, expected_sig):
            return None
        data = json.loads(_b64url_decode(payload))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if any(request.url.path.startswith(p) for p in AUTH_WHITELIST):
            return await call_next(request)

        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        # Fallback: query param for SSE (EventSource can't set headers)
        if not token:
            token = request.query_params.get("_token", "")

        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")

        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        request.state.user = payload
        request.state.user_id = payload["sub"]
        return await call_next(request)
