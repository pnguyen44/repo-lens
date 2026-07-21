from unittest.mock import MagicMock

import pytest

from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.types import IndexedDocument


class TestInit:
    def test_raises_with_no_indexes(self) -> None:
        with pytest.raises(ValueError, match="At least one index must be provided"):
            HybridRetriever()

    def test_stores_indexes(self) -> None:
        index1 = MagicMock()
        index2 = MagicMock()
        retriever = HybridRetriever(index1, index2)

        assert len(retriever._indexes) == 2


class TestAddDocuments:
    def test_delegates_to_all_indexes(self) -> None:
        index1 = MagicMock()
        index2 = MagicMock()
        retriever = HybridRetriever(index1, index2)
        docs: list[IndexedDocument] = [{"content": "hello"}, {"content": "world"}]

        retriever.add_documents(docs)

        index1.add_documents.assert_called_once_with(docs)
        index2.add_documents.assert_called_once_with(docs)

    def test_add_document_delegates_to_all_indexes(self) -> None:
        index1 = MagicMock()
        index2 = MagicMock()
        retriever = HybridRetriever(index1, index2)
        doc: IndexedDocument = {"content": "hello"}

        retriever.add_document(doc)

        index1.add_document.assert_called_once_with(doc)
        index2.add_document.assert_called_once_with(doc)


class TestSearch:
    def test_returns_top_k_results(self) -> None:
        doc_a = {"content": "a"}
        doc_b = {"content": "b"}
        doc_c = {"content": "c"}

        index = MagicMock()
        index.search.return_value = [(doc_a, 0.1), (doc_b, 0.2), (doc_c, 0.3)]

        retriever = HybridRetriever(index)
        results = retriever.search(query_text="test", k=2)

        assert len(results) == 2

    def test_ranks_documents_appearing_in_multiple_indexes_higher(self) -> None:
        doc_shared = {"content": "shared"}
        doc_only_vector = {"content": "only vector"}
        doc_only_bm25 = {"content": "only bm25"}

        vector_index = MagicMock()
        vector_index.search.return_value = [
            (doc_shared, 0.1),
            (doc_only_vector, 0.2),
        ]

        bm25_index = MagicMock()
        bm25_index.search.return_value = [
            (doc_shared, 0.1),
            (doc_only_bm25, 0.2),
        ]

        retriever = HybridRetriever(vector_index, bm25_index)
        results = retriever.search(query_text="test", k=3)

        assert results[0][0]["content"] == "shared"

    def test_returns_empty_when_no_results(self) -> None:
        index = MagicMock()
        index.search.return_value = []

        retriever = HybridRetriever(index)
        results = retriever.search(query_text="test", k=3)

        assert results == []

    def test_rejects_non_string_query(self) -> None:
        retriever = HybridRetriever(MagicMock())

        with pytest.raises(TypeError, match="Query text must be a string"):
            retriever.search(query_text=123, k=1)  # type: ignore[arg-type]

    def test_rejects_non_positive_k(self) -> None:
        retriever = HybridRetriever(MagicMock())

        with pytest.raises(ValueError, match="k must be a positive integer"):
            retriever.search(query_text="test", k=0)


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
