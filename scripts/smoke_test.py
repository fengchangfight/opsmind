"""
Quick smoke test for OpsMind RAG backend.
Tests: config loading, vector store read, retrieve API
"""
import asyncio
import sys
sys.path.insert(0, ".")

from app.config import settings
from app.retrieval.vector_store import VectorStore
from app.retrieval.embedder import Embedder
from app.agents.retrieve_agent import RetrieveAgent


async def main():
    print("=" * 50)
    print("OpsMind RAG - Smoke Test")
    print("=" * 50)

    # 1. Config
    print(f"\n1. Config: model={settings.llm_model}, dense={settings.embedding_dense_model}, sparse={settings.embedding_sparse_model}")
    print(f"   Milvus: {settings.milvus_host}:{settings.milvus_port}")

    # 2. Vector Store
    vs = VectorStore()
    count = vs.count()
    print(f"\n2. Vector Store: {count} chunks indexed")
    if count == 0:
        print("   WARNING: No data. Run 'python scripts/ingest.py' first.")
        return

    # 3. Embedding
    print("\n3. Embedding test...")
    embedder = Embedder()
    emb = await embedder.embed_single("test query")
    dims = len(emb)
    print(f"   Embedding dims: {dims}, first 5: {emb[:5]}")

    # 4. Search
    print("\n4. Search test...")
    results = vs.search(emb, top_k=3)
    for i, r in enumerate(results):
        print(f"   [{i+1}] {r.doc_title[:60]}... (score: {r.score:.3f})")

    # 5. Retrieve Agent
    print("\n5. Retrieve Agent test...")
    ra = RetrieveAgent(embedder, vs)
    results, citations, latency = await ra.retrieve("How to troubleshoot MySQL replication lag?", top_k=3)
    print(f"   Latency: {latency*1000:.0f}ms, Results: {len(results)}")
    for c in citations:
        print(f"   [{c.citation_id}] {c.doc_title[:60]}... (score: {c.relevance_score:.3f})")

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
