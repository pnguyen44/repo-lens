from typing import Any
from chroma_index import ChromaVectorIndex
from hybrid_retriever import HybridRetriever


class DocumentIndexer:
    def __init__(
        self, *, vector_index: ChromaVectorIndex, retriever: HybridRetriever
    ) -> None:
        self._vector_index = vector_index
        self._retriever = retriever

    def load_from_store(self) -> int:
        documents = self._vector_index.get_all_documents()
        if documents:
            self._retriever.reload(documents)

        return len(documents)

    def exits(self, key: str, value: str) -> bool:
        return self._vector_index.exists_in_collection(key, value)

    def index(self, documents: list[dict[str, Any]]) -> int:
        self._retriever.add_documents(documents)
        return len(documents)

    def reindex(self, key: str, value: str, documents: list[dict[str, Any]]) -> int:
        self._vector_index.remove_from_collection(key, value)
        self._retriever.add_documents(documents)  # persist new docs
        self._retriever.reload(
            self._vector_index.get_all_documents()
        )  # rebuild BM25 from truth
        return len(documents)

    def search(self, query: str, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        return self._retriever.search(query_text=query, k=k)
