"""Hybrid ingest: TxtConnector → SentenceSplitter → dual embedding → Milvus.
LlamaIndex handles retrieval/search; manual ingest for schema control (sparse field)."""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from app.config import settings
from app.connectors import TxtConnector
from app.retrieval import Embedder, VectorStore
from app.models import Chunk

CACHE_FILE = Path("data/ingest_cache.json")


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists(): return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(c: dict[str, str]):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(c))


def _hash_content(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def main():
    data_path = settings.sample_data_path
    if not data_path.exists():
        print(f"[ERROR] Sample data not found at {data_path}")
        return

    print(f"[Ingest] Dense: {settings.embedding_dense_model}, Sparse: {settings.embedding_sparse_model}")
    print(f"[Ingest] Max docs/category: {settings.demo_max_docs_per_category}")

    connector = TxtConnector()
    embedder = Embedder()
    vector_store = VectorStore()
    vector_store.clear()
    vector_store._ensure_li_store()  # LlamaIndex creates schema
    print("[Ingest] Collection ready")

    cache = _load_cache()

    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    cats = set(settings.demo_categories) if settings.demo_categories else {"confluence", "github"}
    li_docs: list[Document] = []
    cat_counts: dict[str, int] = {}
    new_cache: dict[str, str] = {}
    skipped = 0

    for cat in cats:
        cat_dir = data_path / cat
        if not cat_dir.exists(): continue
        async for doc in connector.extract(str(cat_dir)):
            if cat_counts.get(cat, 0) >= settings.demo_max_docs_per_category: break
            h = _hash_content(doc.content)
            new_cache[doc.doc_id] = h
            if cache.get(doc.doc_id) == h: skipped += 1; continue
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            li_docs.append(Document(text=doc.content, doc_id=doc.doc_id,
                                     metadata={"doc_title": doc.title, "category": cat}))

    print(f"[Ingest] {len(li_docs)} new docs, {skipped} skipped ({dict(cat_counts)})")

    nodes = splitter(li_docs) if li_docs else []
    print(f"[Ingest] {len(li_docs)} docs → {len(nodes)} nodes")

    if nodes:
        for i in range(0, len(nodes), 32):
            batch = nodes[i:i + 32]
            chunks = [Chunk(chunk_id=n.node_id, doc_id=n.metadata.get("doc_id", ""), content=n.text,
                            metadata={"doc_id": n.metadata.get("doc_id", ""), "doc_title": n.metadata.get("doc_title", ""),
                                      "title": n.metadata.get("doc_title", ""), "category": n.metadata.get("category", "")})
                      for n in batch]
            texts = [f"{c.metadata.get('doc_title','')}\n{c.content}" for c in chunks]
            dense = await embedder.embed(texts)
            sparse = await embedder.embed_sparse(texts)
            for c, d, s in zip(chunks, dense, sparse): c.embedding = d; c.sparse_embedding = s
            vector_store.add_chunks(chunks)
            if (i + 32) % 128 == 0: print(f"[Ingest] {min(i + 32, len(nodes))}/{len(nodes)} chunks")

    removed = set(cache.keys()) - set(new_cache.keys())
    for rid in removed: vector_store.delete_by_doc_id(rid)
    if removed: print(f"[Ingest] Removed {len(removed)} stale docs")

    _save_cache(new_cache)
    print(f"[Ingest] Done! {len(li_docs)} new, {skipped} skipped, Milvus: {vector_store.count}")


if __name__ == "__main__":
    asyncio.run(main())
