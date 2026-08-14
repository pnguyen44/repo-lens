from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from repo_lens.rag.types import IndexedDocument, validate_document


class BaseVectorIndex(ABC):
    def __init__(
        self,
        embedding_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn

    def _require_embedding_fn(
        self,
    ) -> Callable[[list[str]], Awaitable[list[list[float]]]]:
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        return self._embedding_fn

    async def add_document(self, document: IndexedDocument) -> None:
        validate_document(document)
        embed = self._require_embedding_fn()
        content = document["content"]
        vector = (await embed([content]))[0]
        await self._store(vector, document)

    async def add_documents(self, documents: list[IndexedDocument]) -> None:
        embed = self._require_embedding_fn()

        if not documents:
            return

        contents = []
        for i, document in enumerate(documents):
            validate_document(document=document, index=i)
            contents.append(document["content"])

        vectors = await embed(contents)
        await self._store_batch(vectors, documents)

    async def _resolve_query_vector(self, query: Any) -> list[float]:
        if isinstance(query, str):
            embed = self._require_embedding_fn()
            return (await embed([query]))[0]
        elif isinstance(query, list):
            return query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

    @abstractmethod
    async def _store(self, vector: list[float], document: IndexedDocument) -> None: ...

    @abstractmethod
    async def _store_batch(
        self, vectors: list[list[float]], documents: list[IndexedDocument]
    ) -> None: ...
