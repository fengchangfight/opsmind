"""LlamaIndex IngestionPipeline — load → chunk → embed → insert."""
import asyncio, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.schema import Document

from app.config import settings
from app.connectors import TxtConnector
from app.retrieval import VectorStore


async def main():
    data_path = settings.sample_data_path
    if not data_path.exists():
        return print(f"[ERROR] Sample data not found at {data_path}")

    connector = TxtConnector()
    cats = set(settings.demo_categories) if settings.demo_categories else {"confluence", "github"}
    li_docs, cat_counts = [], {}
    for cat in cats:
        cat_dir = data_path / cat
        if not cat_dir.exists(): continue
        async for doc in connector.extract(str(cat_dir)):
            if cat_counts.get(cat, 0) >= settings.demo_max_docs_per_category: break
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            li_docs.append(Document(text=doc.content, doc_id=doc.doc_id,
                                     metadata={"doc_title": doc.title, "category": cat}))
    print(f"[Ingest] {len(li_docs)} docs ({dict(cat_counts)})")
    if not li_docs: return

    vs = VectorStore()
    vs.clear()
    vs._ensure_collection()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
            FastEmbedEmbedding(model_name=settings.embedding_dense_model),
        ],
        vector_store=vs.get_li_store(),
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )

    print("[Ingest] Running IngestionPipeline...")
    nodes = pipeline.run(documents=li_docs, show_progress=True)
    print(f"[Ingest] {len(li_docs)} docs → {len(nodes)} nodes ingested via LlamaIndex")


if __name__ == "__main__":
    asyncio.run(main())
