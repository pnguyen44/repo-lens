from unittest.mock import MagicMock

import pytest

from repo_lens.rag.document_indexer import DocumentIndexer


@pytest.fixture
def indexer_parts():
    vector_index = MagicMock()
    retriever = MagicMock()
    indexer = DocumentIndexer(vector_index=vector_index, retriever=retriever)
    return indexer, vector_index, retriever


def test_load_from_store_with_docs(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    docs = [
        {"content": "chunk one", "repo": "org/repo"},
        {"content": "chunk two", "repo": "org/repo"},
    ]
    vector_index.get_all_documents.return_value = docs

    count = indexer.load_from_store()

    assert count == 2
    retriever.reload.assert_called_once_with(docs)


def test_load_from_store_empty(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    vector_index.get_all_documents.return_value = []

    count = indexer.load_from_store()

    assert count == 0
    retriever.reload.assert_not_called()


def test_file_is_indexed_true(indexer_parts):
    indexer, vector_index, _ = indexer_parts
    vector_index.exists_in_collection.return_value = True

    assert indexer.file_is_indexed(repo="org/repo", path="docs/foo.md") is True
    vector_index.exists_in_collection.assert_called_once_with(
        "file_key", "org/repo:docs/foo.md"
    )


def test_file_is_indexed_false(indexer_parts):
    indexer, vector_index, _ = indexer_parts
    vector_index.exists_in_collection.return_value = False

    assert indexer.file_is_indexed(repo="org/repo", path="docs/foo.md") is False


def test_index_file_indexes_new_file(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    vector_index.exists_in_collection.return_value = False
    docs = [{"content": "chunk one", "repo": "org/repo", "path": "docs/foo.md"}]

    count = indexer.index_file(repo="org/repo", path="docs/foo.md", documents=docs)

    assert count == 1
    retriever.add_documents.assert_called_once_with(
        [
            {
                "content": "chunk one",
                "repo": "org/repo",
                "path": "docs/foo.md",
                "file_key": "org/repo:docs/foo.md",
            }
        ]
    )


def test_index_file_skips_when_already_indexed(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    vector_index.exists_in_collection.return_value = True
    docs = [{"content": "chunk one", "repo": "org/repo", "path": "docs/foo.md"}]

    count = indexer.index_file(repo="org/repo", path="docs/foo.md", documents=docs)

    assert count == 0
    retriever.add_documents.assert_not_called()


def test_clear_repo(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    remaining_docs = [{"content": "other", "repo": "org/repo-b"}]
    vector_index.remove_from_collection.return_value = 65
    vector_index.get_all_documents.return_value = remaining_docs

    removed = indexer.clear_repo("org/repo-a")

    assert removed == 65
    vector_index.remove_from_collection.assert_called_once_with("repo", "org/repo-a")
    retriever.reload.assert_called_once_with(remaining_docs)
    retriever.add_documents.assert_not_called()
