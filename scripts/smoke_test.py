"""
Quick smoke test for OpsMind RAG backend.
Tests: config loading, vector store read, retrieve API
"""
import asyncio
import sys
sys.path.insert(0, ".")

from app.config import settings
from app.retrieval.vector_store import VectorStore
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

    # 3. LlamaIndex Retriever
    print("\n3. LlamaIndex hybrid retriever test...")
    from app.retrieval.embedder import Embedder
    embedder = Embedder()
    ra = RetrieveAgent(embedder, vs)
    ra.init_li_retriever()
    results, citations, latency = await ra.retrieve("retention policy", top_k=3)
    print(f"   Latency: {latency*1000:.0f}ms, Results: {len(results)}")
    for i, r in enumerate(results):
        print(f"   [{i+1}] {r.doc_title[:60]}... (score: {r.score:.3f})")

    print("\n" + "=" * 50)
    print("All smoke tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
