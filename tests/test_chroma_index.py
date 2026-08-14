import pytest

from repo_lens.rag.chroma_index import ChromaVectorIndex
from repo_lens.rag.types import IndexedDocument

fake_embedding = [1.0, 0.0, 0.0]


async def fake_embedding_fn(texts: list[str]) -> list[list[float]]:
    return [fake_embedding] * len(texts)


doc1: IndexedDocument = {
    "content": "How to install the API",
    "repo": "owner/repo-a",
    "section": "Installation",
    "url": "https://github.com/owner/repo-a",
}
doc2: IndexedDocument = {
    "content": "Authentication uses bearer tokens",
    "repo": "owner/repo-a",
    "section": "Auth",
    "url": "https://github.com/owner/repo-a",
}
doc3: IndexedDocument = {
    "content": "Deployment guide for production",
    "repo": "owner/repo-b",
    "section": "Deploy",
    "url": "https://github.com/owner/repo-b",
}

collection_name = "repo_chunks"


def create_chroma_index(path: str) -> ChromaVectorIndex:
    return ChromaVectorIndex(
        collection_name=collection_name,
        embedding_fn=fake_embedding_fn,
        path=path,
    )


@pytest.fixture
def chroma_index(tmp_path):
    return create_chroma_index(str(tmp_path))


async def test_add_documents_and_search(chroma_index) -> None:
    await chroma_index.add_documents([doc1, doc2, doc3])
    results = await chroma_index.search(query="Install instruction for API", k=1)

    assert len(results) == 1
    assert isinstance(results[0], tuple)
    assert "content" in results[0][0]
    assert isinstance(results[0][1], float)
    assert len(chroma_index) == 3


async def test_exists_in_collection(chroma_index) -> None:
    await chroma_index.add_documents([doc1])
    assert (
        await chroma_index.exists_in_collection(
            metadata_key="repo", metadata_value="owner/repo-a"
        )
        is True
    )


async def test_remove_from_collection(chroma_index) -> None:
    await chroma_index.add_documents([doc1, doc2, doc3])
    removed = await chroma_index.remove_from_collection("repo", "owner/repo-a")

    assert removed == 2
    assert await chroma_index.exists_in_collection("repo", "owner/repo-a") is False
    assert await chroma_index.exists_in_collection("repo", "owner/repo-b") is True
    assert len(chroma_index) == 1


async def test_persistence_across_instances(tmp_path) -> None:
    index1 = create_chroma_index(str(tmp_path))
    await index1.add_documents([doc1, doc2])

    index2 = create_chroma_index(str(tmp_path))

    assert len(index2) == 2
