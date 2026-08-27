import asyncio
import hashlib
from typing import Any, Awaitable, Callable

import chromadb

from repo_lens.rag.base_vector_index import BaseVectorIndex
from repo_lens.rag.types import IndexedDocument


class ChromaVectorIndex(BaseVectorIndex):
    def __init__(
        self,
        *,
        collection_name: str,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        host: str | None = None,
        port: int = 8000,
        path: str | None = None,
    ) -> None:
        super().__init__(embedding_fn=embedding_fn)

        if path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.HttpClient(host=host, port=port)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _doc_id(content: str) -> str:
        """
        Generate a stable ID from chunk content.
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _extract_metadata(self, document: IndexedDocument) -> dict[str, str]:
        return {key: str(value) for key, value in document.items() if key != "content"}

    async def _store(self, vector: list[float], document: IndexedDocument) -> None:
        content = document["content"]

        await asyncio.to_thread(
            self._collection.upsert,
            ids=[self._doc_id(content)],
            embeddings=[vector],
            documents=[content],
            metadatas=[self._extract_metadata(document)],
        )

    async def _store_batch(
        self, vectors: list[list[float]], documents: list[IndexedDocument]
    ) -> None:
        contents = [doc["content"] for doc in documents]
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[self._doc_id(c) for c in contents],
            embeddings=vectors,
            documents=contents,
            metadatas=[self._extract_metadata(doc) for doc in documents],
        )

    def _build_result_doc(
        self, content: str, metadata: dict[str, str]
    ) -> IndexedDocument:
        doc: IndexedDocument = {"content": content}
        doc.update(metadata)  # type: ignore[typeddict-item]
        return doc

    async def search(
        self, *, query: Any, k: int = 1, repo: str | None = None
    ) -> list[tuple[IndexedDocument, float]]:
        if await asyncio.to_thread(self._collection.count) == 0:
            return []

        query_vector = await self._resolve_query_vector(query)

        where = {"repo": repo} if repo is not None else None

        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
        )

        return [
            (self._build_result_doc(doc, meta), dist)
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    async def get_all_documents(self) -> list[IndexedDocument]:
        if await asyncio.to_thread(self._collection.count) == 0:
            return []

        results = await asyncio.to_thread(
            self._collection.get, include=["documents", "metadatas"]
        )

        return [
            self._build_result_doc(content=doc_text, metadata=metadata)
            for doc_text, metadata in zip(results["documents"], results["metadatas"])
        ]

    async def get_metadata(
        self, metadata_key: str, metadata_value: str, field: str
    ) -> str | None:
        results = await asyncio.to_thread(
            self._collection.get,
            where={metadata_key: metadata_value},
            limit=1,
            include=["metadatas"],
        )

        if not results["ids"]:
            return None

        metadatas = results["metadatas"]
        if not metadatas:
            return None
        value = metadatas[0].get(field)
        return str(value) if value is not None else None

    async def exists_in_collection(
        self, metadata_key: str, metadata_value: str
    ) -> bool:
        results = await asyncio.to_thread(
            self._collection.get, where={metadata_key: metadata_value}, limit=1
        )
        if not results["ids"]:
            return False
        return True

    async def remove_from_collection(
        self, metadata_key: str, metadata_value: str
    ) -> int:
        results = await asyncio.to_thread(
            self._collection.get, where={metadata_key: metadata_value}
        )
        if not results["ids"]:
            return 0
        await asyncio.to_thread(self._collection.delete, ids=results["ids"])
        return len(results["ids"])

    def __len__(self) -> int:
        return int(self._collection.count())

    def __repr__(self) -> str:
        return (
            f"ChromaVectorIndex(count={len(self)}, "
            f"collection='{self._collection.name}')"
        )
