"""Hybrid ingest with incremental indexing — skips unchanged docs, deletes removed docs, caches via content hash."""
import asyncio
import json
import hashlib
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from app.config import settings
from app.connectors import TxtConnector
from app.retrieval import SimpleChunker, Embedder, VectorStore
from app.models import Chunk

CACHE_FILE = Path("data/ingest_cache.json")


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict[str, str]):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


def _hash_content(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def main():
    data_path = settings.sample_data_path
    if not data_path.exists():
        print(f"[ERROR] Sample data not found at {data_path}")
        return

    print(f"[Ingest] Source: {data_path}")
    print(f"[Ingest] Dense: {settings.embedding_dense_model}, Sparse: {settings.embedding_sparse_model}")

    connector = TxtConnector()
    chunker = SimpleChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    embedder = Embedder()
    vector_store = VectorStore()

    cache = _load_cache()
    print(f"[Ingest] Cache: {len(cache)} previously indexed docs")

    cats_to_scan = set(settings.demo_categories) if settings.demo_categories else {"confluence", "github"}
    cat_counts: dict[str, int] = {}
    batch_size = 10
    chunk_buf: list[Chunk] = []
    text_buf: list[str] = []
    total_new, total_skip, total_delete = 0, 0, 0
    new_cache: dict[str, str] = {}

    for cat_name in cats_to_scan:
        cat_dir = data_path / cat_name
        if not cat_dir.exists():
            continue
        async for doc in connector.extract(str(cat_dir)):
            if cat_counts.get(cat_name, 0) >= settings.demo_max_docs_per_category:
                break
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

            # Q2: dedup — skip if content unchanged
            doc_hash = _hash_content(doc.content)
            new_cache[doc.doc_id] = doc_hash
            if cache.get(doc.doc_id) == doc_hash:
                total_skip += 1
                continue

            total_new += 1
            for c in chunker.chunk(doc):
                chunk_buf.append(c)
                text_buf.append(f"{doc.title}\n{c.content}")
                if len(text_buf) >= batch_size:
                    await _embed_batch(text_buf, chunk_buf, embedder, vector_store)
                    chunk_buf.clear()
                    text_buf.clear()

            if (total_new + total_skip) % 10 == 0:
                print(f"[Ingest] {total_new} new, {total_skip} skipped, cats: {dict(cat_counts)}")

    # Q1: delete chunks for docs no longer in source
    removed_ids = set(cache.keys()) - set(new_cache.keys())
    for doc_id in removed_ids:
        vector_store.delete_by_doc_id(doc_id)
        total_delete += 1
    if total_delete:
        print(f"[Ingest] Removed {total_delete} stale documents")

    # Final batch
    if chunk_buf:
        await _embed_batch(text_buf, chunk_buf, embedder, vector_store)

    _save_cache(new_cache)

    print(f"\n[Ingest] Done! {total_new} new, {total_skip} skipped, {total_delete} deleted, Milvus: {vector_store.count()}")


async def _embed_batch(texts: list[str], chunks: list[Chunk], embedder: Embedder, vector_store: VectorStore):
    dense = await embedder.embed(texts)
    sparse = await embedder.embed_sparse(texts)
    for c, d, s in zip(chunks, dense, sparse):
        c.embedding = d
        c.sparse_embedding = s
    vector_store.add_chunks(chunks)


if __name__ == "__main__":
    asyncio.run(main())
