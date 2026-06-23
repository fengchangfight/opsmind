"""Hybrid ingest: TxtConnector → LlamaIndex SentenceSplitter → dual embedding → Milvus.
LlamaIndex handles chunking; manual embed+insert for schema compatibility."""
import asyncio, hashlib, json, os, sys
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


async def main():
    data_path = settings.sample_data_path
    if not data_path.exists():
        return print(f"[ERROR] Sample data not found at {data_path}")

    connector = TxtConnector()
    embedder = Embedder()
    vs = VectorStore()
    vs.clear(); vs._ensure_collection()

    cache = _load_cache()
    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    cats = set(settings.demo_categories) if settings.demo_categories else {"confluence", "github"}
    li_docs, cat_counts, new_cache, skipped = [], {}, {}, 0

    for cat in cats:
        cat_dir = data_path / cat
        if not cat_dir.exists(): continue
        async for doc in connector.extract(str(cat_dir)):
            if cat_counts.get(cat, 0) >= settings.demo_max_docs_per_category: break
            h = hashlib.md5(doc.content.encode()).hexdigest()
            new_cache[doc.doc_id] = h
            if cache.get(doc.doc_id) == h: skipped += 1; continue
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            li_docs.append(Document(text=doc.content, doc_id=doc.doc_id,
                                     metadata={"doc_title": doc.title, "category": cat}))

    nodes = splitter(li_docs) if li_docs else []
    print(f"[Ingest] {len(li_docs)} new + {skipped} skipped = {len(nodes)} nodes")

    for i in range(0, len(nodes), 32):
        batch = nodes[i:i+32]
        chunks = [Chunk(chunk_id=n.node_id, doc_id=n.metadata.get("doc_id",""), content=n.text,
                         metadata={"doc_id": n.metadata.get("doc_id",""), "doc_title": n.metadata.get("doc_title",""),
                                   "title": n.metadata.get("doc_title",""), "category": n.metadata.get("category","")})
                  for n in batch]
        texts = [f"{c.metadata.get('doc_title','')}\n{c.content}" for c in chunks]
        d = await embedder.embed(texts); s = await embedder.embed_sparse(texts)
        for c, dd, ss in zip(chunks, d, s): c.embedding = dd; c.sparse_embedding = ss
        vs.add_chunks(chunks)

    for rid in set(cache.keys()) - set(new_cache.keys()): vs.delete_by_doc_id(rid)
    _save_cache(new_cache)
    print(f"[Ingest] Done! {vs.count()} chunks in Milvus")


def _load_cache(): return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
def _save_cache(c): CACHE_FILE.parent.mkdir(parents=True, exist_ok=True); CACHE_FILE.write_text(json.dumps(c))


if __name__ == "__main__":
    asyncio.run(main())
