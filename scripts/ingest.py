"""Quick hybrid ingest: sparse + dense embeddings into Milvus."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    chunker = SimpleChunker(chunk_size=512, chunk_overlap=64)
    embedder = Embedder()
    vector_store = VectorStore()

    vector_store.clear()
    print("[Ingest] Cleared collection")

    total_docs = 0
    cat_counts: dict[str, int] = {}
    batch_size = 10
    chunk_buffer: list[Chunk] = []
    text_buffer: list[str] = []

    async for doc in connector.extract(str(data_path)):
        cat = doc.metadata.get("category", "unknown")
        if settings.demo_categories and cat not in settings.demo_categories:
            continue
        if cat_counts.get(cat, 0) >= settings.demo_max_docs_per_category:
            continue
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        total_docs += 1

        chunks = chunker.chunk(doc)
        chunk_buffer.extend(chunks)
        text_buffer.extend([f"{doc.title}\n{c.content}" for c in chunks])

        if len(text_buffer) >= batch_size:
            # Generate both embeddings
            dense_embs = await embedder.embed(text_buffer)
            sparse_embs = await embedder.embed_sparse(text_buffer)
            for c, d, s in zip(chunk_buffer, dense_embs, sparse_embs):
                c.embedding = d
                c.sparse_embedding = s
            vector_store.add_chunks(chunk_buffer)
            chunk_buffer.clear()
            text_buffer.clear()

        if total_docs % 10 == 0:
            print(f"[Ingest] {total_docs} docs, {sum(cat_counts.values())} chunks, cats: {dict(cat_counts)}")

    # Final batch
    if chunk_buffer:
        dense_embs = await embedder.embed(text_buffer)
        sparse_embs = await embedder.embed_sparse(text_buffer)
        for c, d, s in zip(chunk_buffer, dense_embs, sparse_embs):
            c.embedding = d
            c.sparse_embedding = s
        vector_store.add_chunks(chunk_buffer)

    print(f"\n[Ingest] Done! {total_docs} docs, {vector_store.count()} chunks in Milvus")
    print(f"[Ingest] Categories: {dict(cat_counts)}")


if __name__ == "__main__":
    asyncio.run(main())
