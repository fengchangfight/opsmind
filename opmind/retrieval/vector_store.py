import chromadb
from chromadb.config import Settings as ChromaSettings
from opmind.config import settings
from opmind.models import Chunk, SearchResult


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [{k: str(v) for k, v in c.metadata.items()} for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding]

        if embeddings:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        where = None
        if filters:
            where = {k: str(v) for k, v in filters.items()}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                search_results.append(SearchResult(
                    chunk_id=str(chunk_id),
                    doc_id=str(metadata.get("doc_id", "")),
                    content=str(results["documents"][0][i]),
                    doc_title=str(metadata.get("doc_title", metadata.get("title", ""))),
                    score=1.0 - float(distance),
                    metadata=metadata,
                ))
        return search_results

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        try:
            self._client.delete_collection(settings.chroma_collection_name)
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass
