from typing import Any, Protocol


def validate_document(document: dict[str, Any], index: int | None = None) -> None:
    prefix = f"Document at index {index}: " if index is not None else ""

    if not isinstance(document, dict):
        raise TypeError(f"{prefix}Document must be a dictionary.")

    if "content" not in document:
        raise ValueError(f"{prefix}Document dictionary must contain a 'content' key.")

    if not isinstance(document["content"], str):
        raise TypeError(f"{prefix}Document 'content' must be a string.")


class SearchIndex(Protocol):
    def add_document(self, document: dict[str, Any]) -> None: ...

    def add_documents(self, documents: list[dict[str, Any]]) -> None: ...

    def search(self, query: Any, k: int = 1) -> list[tuple[dict[str, Any], float]]: ...

    def __len__(self) -> int: ...


class HybridRetriever:
    def __init__(self, *indexes: SearchIndex) -> None:
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    def add_document(self, document: dict[str, Any]) -> None:
        for index in self._indexes:
            index.add_document(document)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for index in self._indexes:
            index.add_documents(documents)

    def search(
        self, query_text: str, k: int = 1, k_rrf: int = 60
    ) -> list[tuple[dict[str, Any], float]]:
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")

        # Query each index and collect ranked results
        all_results = [
            index.search(query=query_text, k=k * 5) for index in self._indexes
        ]

        doc_ranks: dict[str, dict[str, Any]] = {}
        # For each unique document, gather its rank from each index
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results, start=1):
                match_key = doc["content"]
                if match_key not in doc_ranks:
                    doc_ranks[match_key] = {
                        "doc_obj": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                doc_ranks[match_key]["ranks"][idx] = rank

        # Compute RRF score for each document using calc_rrf_score
        scored_docs: list[tuple[dict[str, Any], float]] = [
            (entry["doc_obj"], self._calc_rrf_score(entry["ranks"], k_rrf))
            for entry in doc_ranks.values()
        ]

        filtered_docs = [(doc, score) for doc, score in scored_docs if score > 0]

        # Sort by RRF score (highest first) and return top k
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        return filtered_docs[:k]

    def _calc_rrf_score(self, ranks: list[float], k_rrf: int = 60) -> float:
        # RRF formula: sum( 1 / (k_rrf + rank) ) for each rank
        # Higher score = more relevant (appeared high across multiple indexes)
        return sum(1 / (k_rrf + rank) for rank in ranks if rank != float("inf"))

    def __len__(self) -> int:
        return len(self._indexes[0])
