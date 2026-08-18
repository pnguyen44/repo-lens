from unittest.mock import AsyncMock, MagicMock

import pytest

from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.types import IndexedDocument


class TestInit:
    def test_raises_with_no_indexes(self) -> None:
        with pytest.raises(ValueError, match="At least one index must be provided"):
            HybridRetriever()

    def test_stores_indexes(self) -> None:
        index1 = AsyncMock()
        index2 = AsyncMock()
        retriever = HybridRetriever(index1, index2)

        assert len(retriever._indexes) == 2


class TestAddDocuments:
    async def test_delegates_to_all_indexes(self) -> None:
        index1 = AsyncMock()
        index2 = AsyncMock()
        retriever = HybridRetriever(index1, index2)
        docs: list[IndexedDocument] = [{"content": "hello"}, {"content": "world"}]

        await retriever.add_documents(docs)

        index1.add_documents.assert_awaited_once_with(docs)
        index2.add_documents.assert_awaited_once_with(docs)

    async def test_add_document_delegates_to_all_indexes(self) -> None:
        index1 = AsyncMock()
        index2 = AsyncMock()
        retriever = HybridRetriever(index1, index2)
        doc: IndexedDocument = {"content": "hello"}

        await retriever.add_document(doc)

        index1.add_document.assert_awaited_once_with(doc)
        index2.add_document.assert_awaited_once_with(doc)


class TestSearch:
    async def test_returns_top_k_results(self) -> None:
        doc_a = {"content": "a"}
        doc_b = {"content": "b"}
        doc_c = {"content": "c"}

        index = AsyncMock()
        index.search.return_value = [(doc_a, 0.1), (doc_b, 0.2), (doc_c, 0.3)]

        retriever = HybridRetriever(index)
        results = await retriever.search(query_text="test", k=2)

        assert len(results) == 2

    async def test_ranks_documents_appearing_in_multiple_indexes_higher(self) -> None:
        doc_shared = {"content": "shared"}
        doc_only_vector = {"content": "only vector"}
        doc_only_bm25 = {"content": "only bm25"}

        vector_index = AsyncMock()
        vector_index.search.return_value = [
            (doc_shared, 0.1),
            (doc_only_vector, 0.2),
        ]

        bm25_index = AsyncMock()
        bm25_index.search.return_value = [
            (doc_shared, 0.1),
            (doc_only_bm25, 0.2),
        ]

        retriever = HybridRetriever(vector_index, bm25_index)
        results = await retriever.search(query_text="test", k=3)

        assert results[0][0]["content"] == "shared"

    async def test_returns_empty_when_no_results(self) -> None:
        index = AsyncMock()
        index.search.return_value = []

        retriever = HybridRetriever(index)
        results = await retriever.search(query_text="test", k=3)

        assert results == []

    async def test_rejects_non_string_query(self) -> None:
        retriever = HybridRetriever(AsyncMock())

        with pytest.raises(TypeError, match="Query text must be a string"):
            await retriever.search(query_text=123, k=1)  # type: ignore[arg-type]

    async def test_rejects_non_positive_k(self) -> None:
        retriever = HybridRetriever(AsyncMock())

        with pytest.raises(ValueError, match="k must be a positive integer"):
            await retriever.search(query_text="test", k=0)


class TestCalcRrfScore:
    def test_multiple_ranks(self) -> None:
        retriever = HybridRetriever(MagicMock())
        ranks: list[float] = [1, 3]
        result = retriever._calc_rrf_score(ranks=ranks)

        expected = 1 / (60 + 1) + 1 / (60 + 3)
        assert result == pytest.approx(expected)

    def test_inf_ranks_are_ignored(self) -> None:
        retriever = HybridRetriever(MagicMock())
        ranks = [2, float("inf")]
        result = retriever._calc_rrf_score(ranks=ranks)

        expected = 1 / (60 + 2)
        assert result == pytest.approx(expected)

    def test_custom_k_rrf(self) -> None:
        retriever = HybridRetriever(MagicMock())
        ranks: list[float] = [1, 1]
        result = retriever._calc_rrf_score(ranks=ranks, k_rrf=10)

        expected = 1 / (10 + 1) + 1 / (10 + 1)
        assert result == pytest.approx(expected)
