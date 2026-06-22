from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.persistence import get_repo
from app.retrieval import Embedder, VectorStore
from app.agents import RetrieveAgent, ReasonAgent
from app.api.routes import query, retrieve, resume, sessions, auth, mcp
from app.api.auth import AuthMiddleware
from app.mcp import McpManager
from app.tools import create_default_registry


def _load_demo_mcp_servers(manager: McpManager):
    """Load demo MCP servers. Always includes the built-in demo server for testing."""
    # Built-in demo server
    from app.mcp.config import McpServerConfig, StdioConfig
    manager.add_server(McpServerConfig(
        name="demo",
        description="Demo MCP server with echo + sysinfo tools",
        transport=StdioConfig(command="python", args=["scripts/demo_mcp_server.py"]),
    ))

    # Additional servers from config
    for cfg_data in settings.mcp_servers:
        try:
            cfg = McpServerConfig(**cfg_data)
            manager.add_server(cfg)
        except Exception as e:
            print(f"[OpsMind] Failed to load MCP server: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[OpsMind] Starting up...")
    print(f"[OpsMind] Vector store: Milvus {settings.milvus_host}:{settings.milvus_port}")

    repo = get_repo()
    repo.init()
    print(f"[OpsMind] DB backend: {settings.db_backend}")

    embedder = Embedder()
    vector_store = VectorStore()
    retrieve_agent = RetrieveAgent(embedder, vector_store)

    # MCP Manager
    mcp_manager = McpManager()
    _load_demo_mcp_servers(mcp_manager)
    tool_registry = create_default_registry()
    reason_agent = ReasonAgent(mcp_manager=mcp_manager, tool_registry=tool_registry)
    retrieve_agent.set_llm_client(reason_agent.client)

    app.state.runtime = {
        "embedder": embedder,
        "vector_store": vector_store,
        "retrieve": retrieve_agent,
        "reason": reason_agent,
        "mcp": mcp_manager,
    }

    # Start MCP servers in background (non-blocking)
    asyncio.create_task(mcp_manager.start_all())

    doc_count = vector_store.count()
    print(f"[OpsMind] Vector store contains {doc_count} chunks")
    print(f"[OpsMind] Ready on http://{settings.api_host}:{settings.api_port}")

    yield

    print("[OpsMind] Shutting down...")


app = FastAPI(
    title="OpsMind RAG",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
)

app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api")
app.include_router(retrieve.router, prefix="/api")
app.include_router(resume.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(mcp.router, prefix="/api")
app.include_router(auth.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "docs_indexed": app.state.runtime["vector_store"].count()}
