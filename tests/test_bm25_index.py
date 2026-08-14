import pytest

from repo_lens.rag.bm25_index import BM25Index
from repo_lens.rag.types import IndexedDocument

DOCS: list[IndexedDocument] = [
    {"content": "Python is a popular programming language for data science"},
    {"content": "JavaScript powers most web applications and frontend frameworks"},
    {"content": "Python and JavaScript are the most popular languages in 2024"},
    {"content": "Machine learning models are built with Python and TensorFlow"},
    {"content": "CSS and HTML are used alongside JavaScript for web development"},
]


@pytest.fixture
async def loaded_index() -> BM25Index:
    index = BM25Index()

    for doc in DOCS:
        await index.add_document(doc)
    return index


class TestAddDocument:
    async def test_stores_document(self) -> None:
        index = BM25Index()
        doc: IndexedDocument = {"content": "hello world"}
        await index.add_document(doc)

        assert len(index.documents) == 1
        assert index.documents[0] == doc

    async def test_updates_doc_stats(self) -> None:
        index = BM25Index()
        await index.add_document({"content": "the cat sat on the cat"})

        assert index._doc_len == [6]
        assert index._doc_freqs == {"the": 1, "cat": 1, "sat": 1, "on": 1}
        assert len(index.documents) == 1

    async def test_rejects_non_dict(self) -> None:
        index = BM25Index()
        with pytest.raises(TypeError):
            await index.add_document("not a dict")  # type: ignore[arg-type]

    async def test_rejects_missing_content_key(self) -> None:
        index = BM25Index()
        with pytest.raises(ValueError):
            await index.add_document({"title": "no content key"})  # type: ignore[typeddict-item]

    async def test_rejects_non_string_content(self) -> None:
        index = BM25Index()
        with pytest.raises(TypeError):
            await index.add_document({"content": 123})  # type: ignore[typeddict-item]


class TestSearch:
    async def test_returns_most_relevant_document_first(
        self, loaded_index: BM25Index
    ) -> None:
        query = "Python programming"
        results = await loaded_index.search(query=query, k=3)

        assert results[0][0]["content"] == DOCS[0]["content"]
        assert results[0][1] < results[-1][1]

    async def test_returns_empty_list_when_no_documents(self) -> None:
        index = BM25Index()
        results = await index.search(query="anything", k=3)
        assert results == []

    async def test_returns_empty_list_when_no_matching_terms(
        self, loaded_index: BM25Index
    ) -> None:
        results = await loaded_index.search(query="Rust concurrency", k=3)
        assert results == []

    @pytest.mark.parametrize("k", [1, 3])
    async def test_respects_k_parameter(self, loaded_index: BM25Index, k: int) -> None:
        query = "Python programming"
        results = await loaded_index.search(query=query, k=k)

        assert len(results) == k

    async def test_lower_score_means_more_relevant(
        self, loaded_index: BM25Index
    ) -> None:
        results = await loaded_index.search(query="TensorFlow models", k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores)

    async def test_rejects_non_string_query(self, loaded_index: BM25Index) -> None:
        with pytest.raises(TypeError):
            await loaded_index.search(query=123, k=3)  # type: ignore[arg-type]
