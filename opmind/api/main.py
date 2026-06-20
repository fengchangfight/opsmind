from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opmind.config import settings
from opmind.retrieval import Embedder, VectorStore
from opmind.agents import RetrieveAgent, ReasonAgent
from opmind.api.routes import query, retrieve, resume


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[OpsMind] Starting up...")
    print(f"[OpsMind] Vector store: {settings.chroma_persist_dir}")

    embedder = Embedder()
    vector_store = VectorStore()
    retrieve_agent = RetrieveAgent(embedder, vector_store)
    reason_agent = ReasonAgent()

    app.state.runtime = {
        "embedder": embedder,
        "vector_store": vector_store,
        "retrieve": retrieve_agent,
        "reason": reason_agent,
    }

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


@app.get("/health")
async def health():
    return {"status": "ok", "docs_indexed": app.state.runtime["vector_store"].count()}
