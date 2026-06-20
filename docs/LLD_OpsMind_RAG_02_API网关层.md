# OpsMind-RAG 详细设计 — API 网关层 (LLD-02)

**版本**: v1.0
**日期**: 2026-06-20
**对应 HLD**: HLD_OpsMind_RAG.md §2.1 API 网关层

---

## 1. 模块职责

| 子模块 | 职责 |
|--------|------|
| **FastAPI App** | 应用生命周期管理、路由注册、中间件挂载 |
| **Query Router** | `/api/query` SSE 流式端点，触发完整 RAG 链路 |
| **Retrieve Router** | `/api/retrieve` 独立检索端点 |
| **Resume Router** | `/api/resume` 中断恢复端点 |
| **Admin Router** | `/api/admin/*` 调试与监控端点 |
| **Auth Middleware** | JWT Token 或 API Key 认证 |
| **Rate Limiter** | 令牌桶算法限流 |
| **Tracing Middleware** | OpenTelemetry 追踪，注入 trace_id |
| **Exception Handler** | 全局异常捕获与统一错误响应格式 |

---

## 2. 目录结构

```
opmind/api/
├── __init__.py
├── main.py                       # FastAPI 实例 + 生命周期
├── routes/
│   ├── __init__.py
│   ├── query.py                  # POST /api/query (SSE)
│   ├── retrieve.py               # POST /api/retrieve
│   ├── resume.py                 # POST /api/resume
│   └── admin.py                  # /api/admin/*
├── middleware/
│   ├── __init__.py
│   ├── auth.py                   # JWT / API Key 认证
│   ├── rate_limiter.py           # 令牌桶限流
│   └── tracing.py                # OpenTelemetry 注入
├── schemas/
│   ├── __init__.py
│   ├── request.py                # 请求体 Pydantic 模型
│   └── response.py               # 响应体 Pydantic 模型
└── dependencies.py               # FastAPI 依赖注入（Session 获取等）
```

---

## 3. 应用入口 (main.py)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opsmind.api.routes import query, retrieve, resume, admin
from opsmind.api.middleware import auth, rate_limiter, tracing
from opsmind.observability.tracer import init_tracer
from opsmind.observability.logger import init_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # Startup
    init_tracer()
    init_logger()
    # 预加载：AgentRuntime、ToolRegistry、模型等
    await app.state.agent_runtime.initialize()
    yield
    # Shutdown
    await app.state.agent_runtime.shutdown()
    await app.state.message_bus.close()


app = FastAPI(
    title="OpsMind RAG",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

# 中间件（执行顺序：从下到上）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(rate_limiter.RateLimitMiddleware)
app.add_middleware(tracing.TracingMiddleware)
app.add_middleware(auth.AuthMiddleware)

# 路由注册
app.include_router(query.router, prefix="/api")
app.include_router(retrieve.router, prefix="/api")
app.include_router(resume.router, prefix="/api")
app.include_router(admin.router, prefix="/api/admin")

# 全局异常处理
app.add_exception_handler(Exception, global_exception_handler)
```

---

## 4. 请求/响应 Schema

### 4.1 QueryRequest

```python
from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: Optional[str] = Field(
        default=None,
        description="已有 session ID，不传则创建新 session"
    )
    context: Optional[QueryContext] = Field(
        default=None,
        description="用户上下文信息"
    )
    options: Optional[QueryOptions] = Field(
        default=None,
        description="查询选项"
    )

class QueryContext(BaseModel):
    user_id: Optional[str] = None
    team: Optional[str] = None
    env: Optional[str] = None

class QueryOptions(BaseModel):
    max_iterations: int = Field(default=3, ge=1, le=5)
    enable_interrupt: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=True)
    tools: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
```

### 4.2 ResumeRequest

```python
class ResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    human_input: str = Field(..., min_length=1, max_length=4096)
    option: str = Field(
        default="continue",
        pattern="^(continue|modify|transfer)$"
    )
```

### 4.3 RetrieveRequest

```python
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: Optional[dict] = Field(
        default=None,
        description="元数据过滤条件: {team: sre, doc_type: runbook}"
    )
```

### 4.4 SSE 事件响应

```python
from pydantic import BaseModel
from typing import Any

class SSEMessage(BaseModel):
    event: str              # 事件类型
    data: dict[str, Any]    # 事件数据
    timestamp: str          # ISO 8601 时间戳

# 具体事件
class AgentStartEvent(SSEMessage):
    event: str = "agent_start"
    data: dict  # {agent_id, description, step}

class RetrievalResultEvent(SSEMessage):
    event: str = "retrieval_result"
    data: dict  # {chunks: [...], citations: [...]}

class ReasoningStepEvent(SSEMessage):
    event: str = "reasoning_step"
    data: dict  # {step, hypothesis, confidence, evidence}

class ToolCallEvent(SSEMessage):
    event: str = "tool_call"
    data: dict  # {tool_name, params, trace_id}

class ToolResultEvent(SSEMessage):
    event: str = "tool_result"
    data: dict  # {tool_name, result, duration, trace_id}

class InterruptedEvent(SSEMessage):
    event: str = "interrupted"
    data: dict  # {reason, confidence, options, session_id}

class FinalAnswerEvent(SSEMessage):
    event: str = "final_answer"
    data: dict  # {answer, citations, tools_called, confidence, iterations}

class ChunkEvent(SSEMessage):
    event: str = "chunk"
    data: dict  # {content: "token..."}

class ErrorEvent(SSEMessage):
    event: str = "error"
    data: dict  # {code, message, trace_id}
```

### 4.5 统一错误响应

```python
class ErrorResponse(BaseModel):
    error: ErrorDetail
    trace_id: str
    timestamp: str

class ErrorDetail(BaseModel):
    code: str           # 错误码: INVALID_QUERY, UNAUTHORIZED, RATE_LIMITED, etc.
    message: str        # 人类可读的错误描述
    details: dict | None = None
```

---

## 5. 核心端点实现

### 5.1 POST /api/query (SSE)

```python
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
import json
import asyncio

router = APIRouter(tags=["query"])

@router.post("/query")
async def query(
    req: QueryRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),  # 认证依赖
):
    """
    主查询接口。返回 Server-Sent Events 流。
    """
    # 1. 获取 AgentRuntime 实例
    runtime: AgentRuntime = http_request.app.state.agent_runtime

    # 2. 获取或创建 Session
    session = await runtime.get_or_create_session(
        session_id=req.session_id,
        user_id=user["user_id"],
        context=req.context,
    )

    # 3. 创建 Trace（OpenTelemetry）
    tracer = get_tracer()
    trace_ctx = tracer.start_span("api.query", attributes={
        "query": req.query[:100],
        "session_id": session.session_id,
        "user_id": user["user_id"],
    })

    # 4. 创建 SSE 事件队列
    event_queue: asyncio.Queue[SSEMessage] = asyncio.Queue()

    # 5. 启动后台推理任务
    inference_task = asyncio.create_task(
        runtime.execute_query(
            query=req.query,
            session=session,
            options=req.options,
            event_queue=event_queue,
            trace_ctx=trace_ctx,
        )
    )

    # 6. 返回 SSE 流
    async def event_generator():
        try:
            while True:
                # 等待事件或推理完成
                try:
                    event = await asyncio.wait_for(
                        event_queue.get(), timeout=0.1
                    )
                    yield f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
                    event_queue.task_done()

                    if event.event in ("final_answer", "error"):
                        break
                except asyncio.TimeoutError:
                    # 发送心跳避免超时
                    yield f": heartbeat\n\n"

            # 等待推理任务完成
            await inference_task
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'code': 'INTERNAL', 'message': str(e)})}\n\n"
        finally:
            trace_ctx.end()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",       # Nginx 代理不缓冲
            "Access-Control-Allow-Origin": "*",
        },
    )
```

### 5.2 POST /api/retrieve

```python
from fastapi import APIRouter, Depends

router = APIRouter(tags=["retrieve"])

@router.post("/retrieve")
async def retrieve(
    req: RetrieveRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
):
    """
    纯检索接口，不触发推理链路。用于调试和独立检索场景。
    """
    runtime: AgentRuntime = http_request.app.state.agent_runtime
    retrieve_agent = runtime.get_agent("retrieve")

    result = await retrieve_agent.retrieve_only(
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
        user_context={"user_id": user["user_id"]},
    )

    return {
        "query": req.query,
        "results": [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "score": c.score,
                "doc_title": c.doc_title,
                "section_path": c.section_path,
            }
            for c in result.chunks
        ],
        "latency_ms": result.latency_ms,
        "trace_id": get_current_trace_id(),
    }
```

### 5.3 POST /api/resume

```python
@router.post("/resume")
async def resume(
    req: ResumeRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
):
    """
    从中断点恢复执行。会返回新的 SSE 流。
    """
    runtime: AgentRuntime = http_request.app.state.agent_runtime

    # 1. 校验 session 是否存在且处于中断状态
    session = await runtime.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "interrupted":
        raise HTTPException(status_code=409, detail="Session not in interrupted state")
    if session.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="Session belongs to another user")

    # 2. 创建 SSE 事件队列
    event_queue: asyncio.Queue = asyncio.Queue()

    # 3. 启动恢复任务
    tracer = get_tracer()
    trace_ctx = tracer.start_span("api.resume", attributes={
        "session_id": session.session_id,
    })

    resume_task = asyncio.create_task(
        runtime.resume_session(
            session=session,
            human_input=req.human_input,
            option=req.option,
            event_queue=event_queue,
            trace_ctx=trace_ctx,
        )
    )

    # 4. SSE 流（同 query）
    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
                if event.event in ("final_answer", "error"):
                    break
        finally:
            trace_ctx.end()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 6. 中间件设计

### 6.1 认证中间件 (auth.py)

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import jwt

AUTH_WHITELIST = [
    "/api/docs", "/api/openapi.json", "/health", "/api/admin/metrics",
]

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # 白名单跳过
        if any(request.url.path.startswith(p) for p in AUTH_WHITELIST):
            return await call_next(request)

        token = request.headers.get("Authorization", "").removeprefix("Bearer ")

        if not token:
            # 发育阶段：允许 API Key fallback
            api_key = request.headers.get("X-API-Key")
            if api_key:
                token = api_key
            else:
                raise HTTPException(status_code=401, detail="Missing authentication")

        try:
            # 优先 JWT 解码，失败则尝试 API Key 校验
            if token.startswith("eyJ"):
                payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
                user_info = {"user_id": payload["sub"], "role": payload.get("role", "user")}
            else:
                user_info = await self.validate_api_key(token)

            request.state.user = user_info
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid authentication")

        return await call_next(request)

    async def validate_api_key(self, api_key: str) -> dict:
        # Demo 阶段：硬编码或 SQLite 查表
        # 生产级：从 Redis/DB 查
        return {"user_id": "demo-user", "role": "user"}
```

### 6.2 限流中间件 (rate_limiter.py)

```python
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class TokenBucket:
    """内存级令牌桶（单机），生产级升级为 Redis Lua 脚本"""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate            # 令牌生成速率（个/秒）
        self.capacity = capacity    # 桶容量
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def consume(self, count: int = 1) -> bool:
        """尝试消费 count 个令牌，成功返回 True"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate: float = 10.0, capacity: int = 20):
        super().__init__(app)
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(rate, capacity)
        )

    async def dispatch(self, request: Request, call_next):
        user_id = getattr(request.state, "user", {}).get("user_id", "anonymous")
        bucket = self.buckets[user_id]

        if not bucket.consume():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
            )

        return await call_next(request)
```

### 6.3 追踪中间件 (tracing.py)

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace
import uuid

class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tracer = trace.get_tracer(__name__)
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))

        with tracer.start_as_current_span(
            f"http.{request.method} {request.url.path}",
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.trace_id": trace_id,
            },
        ) as span:
            request.state.trace_id = trace_id
            request.state.span = span
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            response.headers["X-Trace-ID"] = trace_id
            return response
```

### 6.4 全局异常处理

```python
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

async def global_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger = get_logger()

    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
    else:
        status_code = 500
        message = "Internal server error"
        logger.error(f"Unhandled exception | trace_id={trace_id}", exc_info=exc)

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code="HTTP_" + str(status_code),
                message=message,
            ),
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
    )
```

---

## 7. 依赖注入 (dependencies.py)

```python
from fastapi import Request, Depends, HTTPException

async def get_current_user(request: Request) -> dict:
    """从请求状态中提取已认证的用户信息"""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

async def get_runtime(request: Request) -> "AgentRuntime":
    """获取 AgentRuntime 实例"""
    return request.app.state.agent_runtime

async def get_tool_registry(request: Request) -> "ToolRegistry":
    """获取 ToolRegistry 实例"""
    return request.app.state.tool_registry
```

---

## 8. 错误码定义

| HTTP 状态码 | 错误码 | 说明 |
|-------------|--------|------|
| 400 | `INVALID_QUERY` | 查询参数不合法（空、过长） |
| 400 | `INVALID_SESSION` | Session ID 格式错误 |
| 401 | `UNAUTHORIZED` | 未提供认证凭据 |
| 401 | `TOKEN_EXPIRED` | JWT 过期 |
| 403 | `FORBIDDEN` | 无权访问该 Session |
| 404 | `SESSION_NOT_FOUND` | Session 不存在 |
| 409 | `SESSION_NOT_INTERRUPTED` | Session 不在中断状态，无法恢复 |
| 429 | `RATE_LIMITED` | 请求过于频繁 |
| 500 | `INTERNAL_ERROR` | 服务器内部错误 |
| 503 | `SERVICE_UNAVAILABLE` | LLM / Milvus 等外部服务不可用 |

---

## 9. CORS 配置

```python
# 开发环境：允许 localhost
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]

# 生产环境：仅允许白名单域名（从环境变量读取）
if settings.ENV == "production":
    CORS_ORIGINS = settings.CORS_ORIGINS.split(",")
```

---

## 10. 接口依赖（对其他层）

| 依赖对象 | 获取方式 | 用途 |
|----------|---------|------|
| `AgentRuntime` | `app.state.agent_runtime` | 执行查询、管理 session |
| `RetrieveAgent` | `runtime.get_agent("retrieve")` | 独立检索 |
| `ToolRegistry` | `app.state.tool_registry` | 工具管理接口（可选暴露） |
| `SessionManager` | `runtime.session_manager` | Session CRUD（admin 接口） |

---

## 11. 变更日志

| 版本 | 日期 | 变更 | 作者 |
|------|------|------|------|
| v1.0 | 2026-06-20 | 初始版本 | AI-assisted Design |
