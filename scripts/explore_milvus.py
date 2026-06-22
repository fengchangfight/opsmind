"""Milvus collection explorer - quick CLI to browse indexed chunks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retrieval.vector_store import VectorStore
from app.retrieval.embedder import Embedder


def show_stats(vs: VectorStore):
    print(f"\n{'='*60}")
    print(f"  Collection: opsmind_chunks  |  Chunks: {vs.count()}")
    print(f"{'='*60}")

def show_random(vs: VectorStore, n: int = 5):
    from pymilvus import MilvusClient
    from app.config import settings
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    results = client.query(
        collection_name=settings.milvus_collection_name,
        filter="id >= 0",
        output_fields=["chunk_id", "doc_title", "content", "category"],
        limit=n,
    )
    for i, r in enumerate(results):
        print(f"\n--- [{i+1}] {r.get('doc_title', 'N/A')[:80]} ---")
        print(f"  category: {r.get('category', 'N/A')}  id: {r.get('id', '?')}")
        print(f"  chunk_id: {r.get('chunk_id', '')}")
        content = r.get("content", "")
        print(f"  content:  {content[:200]}{'...' if len(content) > 200 else ''}")

def show_search(vs: VectorStore, query: str, n: int = 5):
    import asyncio
    emb = Embedder()
    vec = asyncio.run(emb.embed_single(query))
    results = vs.search(vec, top_k=n)
    for i, r in enumerate(results):
        print(f"\n--- [{i+1}] {r.doc_title[:80]} ---  score: {r.score:.4f}")
        print(f"  chunk: {r.content[:200]}{'...' if len(r.content) > 200 else ''}")

if __name__ == "__main__":
    vs = VectorStore()
    show_stats(vs)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "random":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            show_random(vs, n)
        elif cmd == "search":
            query = " ".join(sys.argv[2:]) or "troubleshooting"
            n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
            show_search(vs, query, n)
        else:
            show_search(vs, " ".join(sys.argv[1:]))
    else:
        show_random(vs, 5)
