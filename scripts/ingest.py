"""
Full LlamaIndex ingestion pipeline: load → chunk → embed → Milvus.
With incremental indexing via docstore + cache.
"""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.schema import Document

from app.config import settings
from app.connectors import TxtConnector


async def main():
    data_path = settings.sample_data_path
    if not data_path.exists():
        print(f"[ERROR] Sample data not found at {data_path}")
        return

    print(f"[Ingest] Source: {data_path}")
    print(f"[Ingest] Dense: {settings.embedding_dense_model}, Sparse: off in LlamaIndex pipeline")

    connector = TxtConnector()

    # LlamaIndex IngestionPipeline
    cache_dir = Path("data/ingest_cache_li")
    embed_model = FastEmbedEmbedding(model_name=settings.embedding_dense_model)
    vector_store = MilvusVectorStore(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
        collection_name=settings.milvus_collection_name,
        dim=settings.milvus_dim,
        similarity_metric="COSINE",
        overwrite=False,
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
            embed_model,
        ],
        vector_store=vector_store,
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.UPSERTS,
        cache=IngestionCache(cache_dir=str(cache_dir)),
    )

    # Phase 1: load documents
    cats_to_scan = set(settings.demo_categories) if settings.demo_categories else {"confluence", "github"}
    docs: list[Document] = []
    cat_counts: dict[str, int] = {}

    for cat_name in cats_to_scan:
        cat_dir = data_path / cat_name
        if not cat_dir.exists():
            continue
        async for doc in connector.extract(str(cat_dir)):
            if cat_counts.get(cat_name, 0) >= settings.demo_max_docs_per_category:
                break
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
            docs.append(Document(
                text=doc.content,
                doc_id=doc.doc_id,
                metadata={"doc_title": doc.title, "category": cat_name},
            ))

    print(f"[Ingest] Loaded {len(docs)} docs ({dict(cat_counts)})")

    # Phase 2: run pipeline (chunk + embed + insert)
    print(f"[Ingest] Running pipeline...")
    nodes = pipeline.run(documents=docs, show_progress=True)
    print(f"[Ingest] {len(docs)} docs → {len(nodes)} nodes")

    pipeline.docstore.persist(str(cache_dir / "docstore.json"))

    print(f"[Ingest] Done!")


if __name__ == "__main__":
    asyncio.run(main())
