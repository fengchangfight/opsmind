"""Hybrid ingest: fast filtered loading + SentenceSplitter (LlamaIndex) + dual embedding (dense + BM25 sparse)."""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from app.config import settings
from app.connectors import TxtConnector
from app.retrieval import SimpleChunker, Embedder, VectorStore
from app.models import Chunk


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
    vector_store.clear()
    print("[Ingest] Cleared Milvus collection")

    cats_to_scan = set(settings.demo_categories) if settings.demo_categories else {"confluence", "github"}
    cat_counts: dict[str, int] = {}
    batch_size = 10
    chunk_buf: list[Chunk] = []
    text_buf: list[str] = []
    total_chunks = 0

    for cat_name in cats_to_scan:
        cat_dir = data_path / cat_name
        if not cat_dir.exists():
            continue
        async for doc in connector.extract(str(cat_dir)):
            if cat_counts.get(cat_name, 0) >= settings.demo_max_docs_per_category:
                break
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

            for c in chunker.chunk(doc):
                chunk_buf.append(c)
                text_buf.append(f"{doc.title}\n{c.content}")

                if len(text_buf) >= batch_size:
                    await _embed_batch(text_buf, chunk_buf, embedder, vector_store)
                    total_chunks += len(chunk_buf)
                    chunk_buf.clear()
                    text_buf.clear()

            if sum(cat_counts.values()) % 10 == 0:
                print(f"[Ingest] {sum(cat_counts.values())} docs, ~{total_chunks} chunks, cats: {dict(cat_counts)}")

    if chunk_buf:
        await _embed_batch(text_buf, chunk_buf, embedder, vector_store)
        total_chunks += len(chunk_buf)

    print(f"\n[Ingest] Done! {sum(cat_counts.values())} docs → {total_chunks} chunks, Milvus: {vector_store.count()}")


async def _embed_batch(texts: list[str], chunks: list[Chunk], embedder: Embedder, vector_store: VectorStore):
    dense = await embedder.embed(texts)
    sparse = await embedder.embed_sparse(texts)
    for c, d, s in zip(chunks, dense, sparse):
        c.embedding = d
        c.sparse_embedding = s
    vector_store.add_chunks(chunks)


if __name__ == "__main__":
    asyncio.run(main())
