"""
Document ingestion pipeline using LlamaIndex.
Features: auto-chunking, incremental caching, batch embedding into Milvus.
"""
import asyncio
import sys
from pathlib import Path
from hashlib import md5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from app.config import settings
from app.retrieval import VectorStore
from app.models import Chunk


async def main():
    data_path = settings.sample_data_path
    if not data_path.exists():
        print(f"[ERROR] Sample data not found at {data_path}")
        return

    print(f"[Ingest] Source: {data_path}")
    print(f"[Ingest] Categories: {settings.demo_categories or 'all'}")
    print(f"[Ingest] Max docs per category: {settings.demo_max_docs_per_category}")

    # LlamaIndex pipeline
    embed_model = FastEmbedEmbedding(model_name=settings.embedding_model)
    cache_dir = Path(settings.sqlite_path).parent / "ingest_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
            embed_model,
        ],
        docstore=SimpleDocumentStore.from_persist_dir(str(cache_dir)) if cache_dir.exists() else None,
        docstore_strategy=DocstoreStrategy.UPSERTS,
        cache=IngestionCache(
            cache_dir=str(cache_dir),
            cache_hash=f"ingest-v1-{settings.embedding_model}",
        ),
    )

    # Load documents
    reader = SimpleDirectoryReader(
        input_dir=str(data_path),
        recursive=True,
        required_exts=[".txt", ".md", ".json"],
    )
    docs = reader.load_data()
    print(f"[Ingest] Loaded {len(docs)} raw documents")

    # Apply category filters
    cat_counts: dict[str, int] = {}
    filtered_docs = []
    for doc in docs:
        cat = _infer_category(doc.metadata.get("file_path", ""))
        if settings.demo_categories and cat not in settings.demo_categories:
            continue
        if cat_counts.get(cat, 0) >= settings.demo_max_docs_per_category:
            continue
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        doc.metadata["category"] = cat
        doc.metadata["doc_title"] = _extract_title(doc.text) or doc.metadata.get("file_name", "untitled")
        filtered_docs.append(doc)
    print(f"[Ingest] Filtered to {len(filtered_docs)} docs ({dict(cat_counts)})")

    # Run pipeline → produces nodes with embeddings
    print(f"[Ingest] Running ingestion pipeline (chunking + embedding)...")
    nodes = pipeline.run(documents=filtered_docs, show_progress=True)
    print(f"[Ingest] Produced {len(nodes)} nodes")

    # Convert to OpsMind Chunks and insert into Milvus
    vector_store = VectorStore()
    vector_store.clear()
    print("[Ingest] Cleared existing Milvus collection")

    chunks = []
    for node in nodes:
        chunk_id = node.node_id
        doc_id = md5(str(node.metadata.get("file_path", "")).encode()).hexdigest()
        embedding = node.embedding

        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=node.text,
            embedding=embedding,
            metadata={
                "doc_id": doc_id,
                "doc_title": node.metadata.get("doc_title", ""),
                "title": node.metadata.get("doc_title", ""),
                "category": node.metadata.get("category", ""),
                "file_path": node.metadata.get("file_path", ""),
            },
        ))

        # Batch insert
        if len(chunks) >= 50:
            vector_store.add_chunks(chunks)
            chunks = []

    if chunks:
        vector_store.add_chunks(chunks)

    pipeline.docstore.persist(str(cache_dir))

    print(f"\n[Ingest] Complete!")
    print(f"  Documents: {len(filtered_docs)}")
    print(f"  Chunks: {len(nodes)}")
    print(f"  Milvus count: {vector_store.count()}")


def _infer_category(file_path: str) -> str:
    for cat in ["confluence", "github", "gmail", "jira", "hubspot", "google_drive", "fireflies"]:
        if cat in file_path:
            return cat
    return "unknown"


def _extract_title(text: str) -> str | None:
    first_line = text.split("\n", 1)[0].strip()
    if first_line and len(first_line) < 200 and not first_line.startswith("---"):
        return first_line
    return None


if __name__ == "__main__":
    asyncio.run(main())
