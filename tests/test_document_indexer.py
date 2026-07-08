import pytest
from unittest.mock import MagicMock
from document_indexer import DocumentIndexer


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


def test_index(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    docs = [
        {"content": "chunk one", "repo": "org/repo"},
        {"content": "chunk two", "repo": "org/repo"},
    ]

    count = indexer.index(docs)

    assert count == 2
    retriever.add_documents.assert_called_once_with(docs)


def test_reindex(indexer_parts):
    indexer, vector_index, retriever = indexer_parts
    existing_docs = [{"content": "old", "repo": "org/repo-a"}]
    new_docs = [{"content": "new chunk", "repo": "org/repo-b"}]
    vector_index.get_all_documents.return_value = existing_docs + new_docs

    count = indexer.reindex(key="repo", value="org/repo-b", documents=new_docs)

    assert count == 1
    vector_index.remove_from_collection.assert_called_once_with("repo", "org/repo-b")
    retriever.add_documents.assert_called_once_with(new_docs)
    retriever.reload.assert_called_once_with(existing_docs + new_docs)
