from abc import ABC, abstractmethod
from typing import Any, Callable

from repo_lens.rag.hybrid_retriever import validate_document
from repo_lens.rag.types import IndexedDocument


class BaseVectorIndex(ABC):
    def __init__(
        self,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn

    def _require_embedding_fn(self) -> Callable[[list[str]], list[list[float]]]:
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        return self._embedding_fn

    def add_document(self, document: IndexedDocument) -> None:
        validate_document(document)
        embed = self._require_embedding_fn()
        content = document["content"]
        vector = embed([content])[0]
        self._store(vector, document)

    def add_documents(self, documents: list[IndexedDocument]) -> None:
        embed = self._require_embedding_fn()

        if not documents:
            return

        contents = []
        for i, document in enumerate(documents):
            validate_document(document=document, index=i)
            contents.append(document["content"])

        vectors = embed(contents)
        self._store_batch(vectors, documents)

    def _resolve_query_vector(self, query: Any) -> list[float]:
        if isinstance(query, str):
            embed = self._require_embedding_fn()
            return embed([query])[0]
        elif isinstance(query, list):
            return query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

    @abstractmethod
    def _store(self, vector: list[float], document: IndexedDocument) -> None: ...

    @abstractmethod
    def _store_batch(
        self, vectors: list[list[float]], documents: list[IndexedDocument]
    ) -> None: ...
