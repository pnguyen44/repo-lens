from typing import Any

from repo_lens.rag.bm25_index import BM25Index
from repo_lens.rag.protocols import SearchIndex
from repo_lens.rag.types import IndexedDocument


class HybridRetriever:
    def __init__(self, *indexes: SearchIndex) -> None:
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    async def add_document(self, document: IndexedDocument) -> None:
        for index in self._indexes:
            await index.add_document(document)

    async def add_documents(self, documents: list[IndexedDocument]) -> None:
        for index in self._indexes:
            await index.add_documents(documents)

    async def search(
        self,
        *,
        query_text: str,
        k: int = 1,
        k_rrf: int = 60,
        repo: str | None = None,
    ) -> list[tuple[IndexedDocument, float]]:
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")

        # Query each index and collect ranked results
        all_results = [
            await index.search(query=query_text, k=k * 5, repo=repo)
            for index in self._indexes
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
        scored_docs: list[tuple[IndexedDocument, float]] = [
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

    async def reload_bm25(self, documents: list[IndexedDocument]) -> None:
        for index in self._indexes:
            if isinstance(index, BM25Index):
                index.clear()
                await index.add_documents(documents)
                return
        raise RuntimeError("BM25 index not found")

    def __len__(self) -> int:
        return len(self._indexes[0])
