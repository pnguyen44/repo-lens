import pytest

from repo_lens.rag.types import IndexedDocument
from repo_lens.rag.vector_index import VectorIndex

vec1 = [1.0, 0.0, 0.0]
vec2 = [0.0, 1.0, 0.0]
doc1: IndexedDocument = {"content": "doc A"}
doc2: IndexedDocument = {"content": "doc B"}


async def test_search_returns_closest_match() -> None:
    index = VectorIndex()

    index.add_vector(vec1, doc1)
    index.add_vector(vec2, doc2)

    results = await index.search(query=vec1, k=1)

    assert len(results) == 1
    assert results[0][0]["content"] == doc1["content"]


async def test_search_rejects_positional_query() -> None:
    index = VectorIndex()
    index.add_vector(vec1, doc1)

    with pytest.raises(TypeError):
        await index.search(vec1, k=1)  # type: ignore[misc]


COSINE_CASES = [
    {
        "name": "identical vectors is zero",
        "vec1": [1.0, 2.0, 3.0],
        "vec2": [1.0, 2.0, 3.0],
        "expected": 0.0,
    },
    {
        "name": "orthogonal vectors is one",
        "vec1": [1.0, 0.0],
        "vec2": [0.0, 1.0],
        "expected": 1.0,
    },
    {
        "name": "opposite vectors is two",
        "vec1": [1.0, 0.0],
        "vec2": [-1.0, 0.0],
        "expected": 2.0,
    },
]

EUCLIDEAN_CASES = [
    {
        "name": "3-4-5 triangle",
        "vec1": [0.0, 0.0],
        "vec2": [3.0, 4.0],
        "expected": 5.0,
    },
    {
        "name": "identical vectors is zero",
        "vec1": [1.0, 2.0],
        "vec2": [1.0, 2.0],
        "expected": 0.0,
    },
]


@pytest.mark.parametrize("case", COSINE_CASES, ids=lambda c: c["name"])
def test_cosine_distance(case) -> None:
    index = VectorIndex()
    distance = index._cosine_distance(case["vec1"], case["vec2"])

    assert distance == pytest.approx(case["expected"])


@pytest.mark.parametrize("case", EUCLIDEAN_CASES, ids=lambda c: c["name"])
def test_euclidean_distance(case) -> None:
    index = VectorIndex()
    distance = index._euclidean_distance(case["vec1"], case["vec2"])

    assert distance == pytest.approx(case["expected"])


async def test_search_empty_index_returns_empty() -> None:
    index = VectorIndex()
    results = await index.search(query=vec1, k=1)

    assert results == []


async def test_search_returns_results_in_order() -> None:
    index = VectorIndex()
    index.add_vector([1.0, 0.0, 0.0], {"content": "closest"})
    index.add_vector([0.9, 0.1, 0.0], {"content": "middle"})
    index.add_vector([0.0, 1.0, 0.0], {"content": "farthest"})

    results = await index.search(query=[1.0, 0.0, 0.0], k=3)

    assert results[0][0]["content"] == "closest"
    assert results[1][0]["content"] == "middle"
    assert results[2][0]["content"] == "farthest"


async def test_search_respects_k() -> None:
    index = VectorIndex()
    index.add_vector([1.0, 0.0], {"content": "a"})
    index.add_vector([0.0, 1.0], {"content": "b"})
    index.add_vector([0.5, 0.5], {"content": "c"})

    results = await index.search(query=[1.0, 0.0], k=2)

    assert len(results) == 2


def test_add_vector_rejects_mismatched_dimensions() -> None:
    index = VectorIndex()
    index.add_vector([1.0, 0.0, 0.0], {"content": "3d"})

    with pytest.raises(ValueError, match="Inconsistent vector dimension"):
        index.add_vector([1.0, 0.0], {"content": "2d"})


async def test_add_document_raises_without_embedding_fn() -> None:
    index = VectorIndex()

    with pytest.raises(ValueError, match="Embedding function not provided"):
        await index.add_document({"content": "hello"})


async def test_add_document_uses_embedding_fn() -> None:
    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]

    index = VectorIndex(embedding_fn=fake_embed)

    await index.add_document({"content": "hello"})

    assert len(index) == 1
    assert index.vectors[0] == [1.0, 2.0, 3.0]


async def test_add_documents_stores_all_documents() -> None:
    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[float(i), 0.0, 0.0] for i in range(len(texts))]

    index = VectorIndex(embedding_fn=fake_embed)
    docs: list[IndexedDocument] = [
        {"content": "first", "section": "intro"},
        {"content": "second", "section": "body"},
        {"content": "third", "section": "end"},
    ]

    await index.add_documents(docs)

    assert len(index) == 3
    assert index.documents[0]["content"] == "first"
    assert index.documents[2]["section"] == "end"
    assert index.vectors[1] == [1.0, 0.0, 0.0]


async def test_add_documents_raises_without_embedding_fn() -> None:
    index = VectorIndex()

    with pytest.raises(ValueError, match="Embedding function not provided"):
        await index.add_documents([{"content": "hello"}])


async def test_add_documents_validates_documents() -> None:
    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    index = VectorIndex(embedding_fn=fake_embed)

    with pytest.raises(ValueError, match="Document at index 1"):
        await index.add_documents(
            [{"content": "valid"}, {"wrong_key": "invalid"}]  # type: ignore[typeddict-item,typeddict-unknown-key]
        )

    assert len(index) == 0


async def test_add_documents_empty_list() -> None:
    async def fake_embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("Should not be called")

    index = VectorIndex(embedding_fn=fake_embed)

    await index.add_documents([])

    assert len(index) == 0
