import asyncio
from unittest.mock import AsyncMock

import pytest

from repo_lens.rag.document_indexer import DocumentIndexer


@pytest.fixture
def indexer_parts():
    vector_index = AsyncMock()
    retriever = AsyncMock()
    indexer = DocumentIndexer(vector_index=vector_index, retriever=retriever)
    return indexer, vector_index, retriever


async def test_sync_bm25_from_store_with_docs(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    docs = [
        {"content": "chunk one", "repo": "org/repo"},
        {"content": "chunk two", "repo": "org/repo"},
    ]
    vector_index.get_all_documents.return_value = docs

    count = await indexer.sync_bm25_from_store()

    assert count == 2
    retriever.reload_bm25.assert_awaited_once_with(docs)


async def test_sync_bm25_from_store_empty(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    vector_index.get_all_documents.return_value = []

    count = await indexer.sync_bm25_from_store()

    assert count == 0
    retriever.reload_bm25.assert_not_awaited()


async def test_file_is_indexed_true(indexer_parts):
    indexer, vector_index, _ = indexer_parts
    vector_index.exists_in_collection.return_value = True

    assert await indexer.file_is_indexed(repo="org/repo", path="docs/foo.md") is True
    vector_index.exists_in_collection.assert_awaited_once_with(
        "file_key", "org/repo:docs/foo.md"
    )


async def test_file_is_indexed_false(indexer_parts):
    indexer, vector_index, _ = indexer_parts
    vector_index.exists_in_collection.return_value = False

    assert await indexer.file_is_indexed(repo="org/repo", path="docs/foo.md") is False


async def test_index_file_indexes_new_file(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    vector_index.exists_in_collection.return_value = False
    docs = [{"content": "chunk one", "repo": "org/repo", "path": "docs/foo.md"}]

    count = await indexer.index_file(
        repo="org/repo", path="docs/foo.md", documents=docs
    )

    assert count == 1
    retriever.add_documents.assert_awaited_once_with(
        [
            {
                "content": "chunk one",
                "repo": "org/repo",
                "path": "docs/foo.md",
                "file_key": "org/repo:docs/foo.md",
            }
        ]
    )


async def test_index_file_skips_when_already_indexed(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    vector_index.exists_in_collection.return_value = True
    docs = [{"content": "chunk one", "repo": "org/repo", "path": "docs/foo.md"}]

    count = await indexer.index_file(
        repo="org/repo", path="docs/foo.md", documents=docs
    )

    assert count == 0
    retriever.add_documents.assert_not_awaited()


async def test_index_file_concurrent_calls_index_once(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    docs = [{"content": "chunk one", "repo": "org/repo", "path": "docs/foo.md"}]

    indexed = False

    async def fake_exists_in_collection(_field, _value):
        await asyncio.sleep(0)
        return indexed

    async def fake_add_documents(_stamped):
        nonlocal indexed
        await asyncio.sleep(0)
        indexed = True

    vector_index.exists_in_collection.side_effect = fake_exists_in_collection
    retriever.add_documents.side_effect = fake_add_documents

    results = await asyncio.gather(
        indexer.index_file(repo="org/repo", path="docs/foo.md", documents=docs),
        indexer.index_file(repo="org/repo", path="docs/foo.md", documents=docs),
    )

    assert sorted(results) == [0, 1]
    retriever.add_documents.assert_awaited_once()


async def test_index_file_different_files_index_independently(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    vector_index.exists_in_collection.return_value = False
    docs_a = [{"content": "chunk a", "repo": "org/repo", "path": "docs/a.md"}]
    docs_b = [{"content": "chunk b", "repo": "org/repo", "path": "docs/b.md"}]

    results = await asyncio.gather(
        indexer.index_file(repo="org/repo", path="docs/a.md", documents=docs_a),
        indexer.index_file(repo="org/repo", path="docs/b.md", documents=docs_b),
    )

    assert tuple(results) == (1, 1)
    assert retriever.add_documents.await_count == 2


async def test_clear_repo(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    remaining_docs = [{"content": "other", "repo": "org/repo-b"}]
    vector_index.remove_from_collection.return_value = 65
    vector_index.get_all_documents.return_value = remaining_docs

    removed = await indexer.clear_repo("org/repo-a")

    assert removed == 65
    vector_index.remove_from_collection.assert_awaited_once_with("repo", "org/repo-a")
    retriever.reload_bm25.assert_awaited_once_with(remaining_docs)
    retriever.add_documents.assert_not_awaited()
