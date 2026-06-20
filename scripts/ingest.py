import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opmind.config import settings
from opmind.connectors import TxtConnector
from opmind.retrieval import SimpleChunker, Embedder, VectorStore
from opmind.models import Chunk


async def main():
    data_path = settings.sample_data_path
    if not data_path.exists():
        print(f"[ERROR] Sample data not found at {data_path}")
        return

    print(f"[Ingest] Source: {data_path}")
    print(f"[Ingest] Categories: {settings.demo_categories or 'all'}")
    print(f"[Ingest] Max docs per category: {settings.demo_max_docs_per_category}")

    connector = TxtConnector()
    chunker = SimpleChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = Embedder()
    vector_store = VectorStore()

    vector_store.clear()
    print("[Ingest] Cleared existing collection")

    total_docs = 0
    total_chunks = 0
    cat_counts: dict[str, int] = {}

    all_chunks: list[Chunk] = []
    batch_size = 20

    async for doc in connector.extract(str(data_path)):
        category = doc.metadata.get("category", "unknown")

        if settings.demo_categories and category not in settings.demo_categories:
            continue

        if cat_counts.get(category, 0) >= settings.demo_max_docs_per_category:
            continue

        cat_counts[category] = cat_counts.get(category, 0) + 1
        total_docs += 1

        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        total_chunks += len(chunks)

        if len(all_chunks) >= batch_size:
            print(f"[Ingest] Embedding batch of {len(all_chunks)} chunks...")
            texts = [c.content for c in all_chunks]
            embeddings = await embedder.embed(texts)
            for c, emb in zip(all_chunks, embeddings):
                c.embedding = emb
            vector_store.add_chunks(all_chunks)
            all_chunks = []

        if total_docs % 10 == 0:
            print(f"[Ingest] Progress: {total_docs} docs, {total_chunks} chunks, categories: {dict(cat_counts)}")

    if all_chunks:
        print(f"[Ingest] Embedding final batch of {len(all_chunks)} chunks...")
        texts = [c.content for c in all_chunks]
        embeddings = await embedder.embed(texts)
        for c, emb in zip(all_chunks, embeddings):
            c.embedding = emb
        vector_store.add_chunks(all_chunks)

    print(f"\n[Ingest] Complete!")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks: {total_chunks}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  Store count: {vector_store.count()}")


if __name__ == "__main__":
    asyncio.run(main())
