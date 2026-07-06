import hashlib
from typing import Any, Callable

import chromadb

from base_vector_index import BaseVectorIndex


class ChromaVectorIndex(BaseVectorIndex):
    METADATA_KEYS = ("repo", "section", "url")

    def __init__(
        self,
        path: str,
        collection_name: str = "repo_chunks",
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        super().__init__(embedding_fn=embedding_fn)
        self._client = chromadb.PersistentClient(path=path)
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

    def _extract_metadata(self, document: dict[str, Any]) -> dict[str, str]:
        return {key: document[key] for key in self.METADATA_KEYS if key in document}

    def _store(self, vector: list[float], document: dict[str, Any]) -> None:
        content = document["content"]

        self._collection.upsert(
            ids=[self._doc_id(content)],
            embeddings=[vector],
            documents=[content],
            metadatas=[self._extract_metadata(document)],
        )

    def _store_batch(
        self, vectors: list[list[float]], documents: list[dict[str, Any]]
    ) -> None:
        contents = [doc["content"] for doc in documents]
        self._collection.upsert(
            ids=[self._doc_id(c) for c in contents],
            embeddings=vectors,
            documents=contents,
            metadatas=[self._extract_metadata(doc) for doc in documents],
        )

    def _build_result_doc(
        self, content: str, metadata: dict[str, str]
    ) -> dict[str, Any]:
        doc: dict[str, Any] = {"content": content}
        doc.update(metadata)
        return doc

    def search(self, query: Any, k: int = 1) -> list[tuple[dict[str, Any], float]]:
        if self._collection.count() == 0:
            return []

        query_vector = self._resolve_query_vector(query)

        results = self._collection.query(query_embeddings=[query_vector], n_results=k)

        return [
            (self._build_result_doc(doc, meta), dist)
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_all_documents(self) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["documents", "metadatas"])

        return [
            self._build_result_doc(content=doc_text, metadata=metadata)
            for doc_text, metadata in zip(results["documents"], results["metadatas"])
        ]

    def has_repo(self, repo: str) -> bool:
        results = self._collection.get(where={"repo": repo}, limit=1)

        return len(results["ids"]) > 0

    def __len__(self) -> int:
        return int(self._collection.count())

    def __repr__(self) -> str:
        return (
            f"ChromaVectorIndex(count={len(self)}, "
            f"collection='{self._collection.name}')"
        )
