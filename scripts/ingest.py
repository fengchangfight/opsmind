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
from app.retrieval import Embedder, VectorStore
from app.models import Chunk
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

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
    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    embedder = Embedder()
    vector_store = VectorStore()

    cache = _load_cache()
    print(f"[Ingest] Cache: {len(cache)} previously indexed docs")

    cats_to_scan = set(settings.demo_categories) if settings.demo_categories else {"confluence", "github"}
    cat_counts: dict[str, int] = {}
    li_docs: list[Document] = []
    new_doc_ids: list[str] = []
    total_skip = 0
    new_cache: dict[str, str] = {}

    # Phase 1: collect documents + filter
    for cat_name in cats_to_scan:
        cat_dir = data_path / cat_name
        if not cat_dir.exists():
            continue
        async for doc in connector.extract(str(cat_dir)):
            if cat_counts.get(cat_name, 0) >= settings.demo_max_docs_per_category:
                break
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

            doc_hash = _hash_content(doc.content)
            new_cache[doc.doc_id] = doc_hash
            if cache.get(doc.doc_id) == doc_hash:
                total_skip += 1
                continue

            li_docs.append(Document(
                text=doc.content,
                doc_id=doc.doc_id,
                metadata={"doc_title": doc.title, "category": cat_name, "doc_id": doc.doc_id},
            ))
            new_doc_ids.append(doc.doc_id)

    print(f"[Ingest] Phase 1: {len(li_docs)} new docs, {total_skip} skipped")

    # Phase 2: chunk with LlamaIndex SentenceSplitter
    print(f"[Ingest] Phase 2: chunking {len(li_docs)} docs...")
    nodes = splitter(li_docs) if li_docs else []
    print(f"[Ingest] {len(li_docs)} docs → {len(nodes)} nodes")

    # Phase 3: embed + insert
    if nodes:
        print(f"[Ingest] Phase 3: embedding {len(nodes)} nodes...")
        batch_size = 32
        for start in range(0, len(nodes), batch_size):
            batch_nodes = nodes[start:start + batch_size]
            chunk_batch = [
                Chunk(
                    chunk_id=n.node_id,
                    doc_id=n.metadata.get("doc_id", ""),
                    content=n.text,
                    metadata={
                        "doc_id": n.metadata.get("doc_id", ""),
                        "doc_title": n.metadata.get("doc_title", ""),
                        "title": n.metadata.get("doc_title", ""),
                        "category": n.metadata.get("category", ""),
                    },
                )
                for n in batch_nodes
            ]
            texts = [f"{c.metadata.get('doc_title','')}\n{c.content}" for c in chunk_batch]
            dense = await embedder.embed(texts)
            sparse = await embedder.embed_sparse(texts)
            for c, d, s in zip(chunk_batch, dense, sparse):
                c.embedding = d
                c.sparse_embedding = s
            vector_store.add_chunks(chunk_batch)
            if (start + batch_size) % 128 == 0:
                print(f"[Ingest] Embedded {min(start + batch_size, len(nodes))}/{len(nodes)}")

    # Phase 4: cleanup deleted docs
    removed_ids = set(cache.keys()) - set(new_cache.keys())
    for doc_id in removed_ids:
        vector_store.delete_by_doc_id(doc_id)
    total_delete = len(removed_ids)
    if total_delete:
        print(f"[Ingest] Removed {total_delete} stale documents")

    _save_cache(new_cache)

    print(f"\n[Ingest] Done! {len(new_doc_ids)} new, {total_skip} skipped, {total_delete} deleted")
    print(f"[Ingest] {len(nodes)} chunks in Milvus: {vector_store.count()}")


if __name__ == "__main__":
    asyncio.run(main())
