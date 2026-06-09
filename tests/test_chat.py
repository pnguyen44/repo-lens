from unittest.mock import MagicMock
from chat import Chat
import pytest

from vector_index import VectorIndex


def _make_chat(index: VectorIndex) -> Chat:
    embedder = MagicMock()
    embedder.generate_embeddings.return_value = [[1.0, 0.0, 0.0]]

    return Chat(chat_client=MagicMock(), mcp_clients={}, embedder=embedder, index=index)


CONTEXT_CASES = [
    {
        "name": "includes relevant chunks",
        "vectors": [
            ([1.0, 0.0, 0.0], {"content": "good", "repo": "r", "section": "s"})
        ],
        "expected_in": ["good", "<source"],
        "expected_not_in": [],
    },
    {
        "name": "excludes irrelevant chunks",
        "vectors": [([0.0, 1.0, 0.0], {"content": "bad", "repo": "r", "section": "s"})],
        "expected_in": ["No relevant source found"],
        "expected_not_in": ["bad"],
    },
    {
        "name": "filters mixed relevance",
        "vectors": [
            ([1.0, 0.0, 0.0], {"content": "good", "repo": "r", "section": "s"}),
            ([0.0, 1.0, 0.0], {"content": "bad", "repo": "r", "section": "s"}),
        ],
        "expected_in": ["good", "<source"],
        "expected_not_in": ["bad"],
    },
]


@pytest.mark.parametrize("case", CONTEXT_CASES, ids=lambda c: c["name"])
def test_build_context(case) -> None:
    index = VectorIndex()
    for vec, doc in case["vectors"]:
        index.add_vector(vec, doc)

    chat = _make_chat(index)
    result = chat._build_context("test query")

    print("result", result)

    for item in case["expected_in"]:
        assert item in result

    for item in case["expected_not_in"]:
        assert item not in result


def test_build_context_empty_index() -> None:
    index = VectorIndex()
    chat = _make_chat(index)
    result = chat._build_context("test query")

    assert result == ""
