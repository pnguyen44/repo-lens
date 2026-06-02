import pytest

from typing import Any

from chunker import chunk_by_section, chunk_by_sentence, chunk_by_size

SECTION_CASES = [
    {
        "name": "splits on h2 headers",
        "text": "# Title\n\nIntro text\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B",
        "expected_count": 3,
        "expected_contents": {0: "Title", 1: "Content A", 2: "Content B"},
    },
    {
        "name": "no headers returns single chunk",
        "text": "Just a plain document with no markdown headers.",
        "expected_count": 1,
        "expected_contents": {0: "Just a plain document"},
    },
    {
        "name": "empty string returns single chunk",
        "text": "",
        "expected_count": 1,
        "expected_contents": {0: ""},
    },
    {
        "name": "preserves content within sections",
        "text": "# Repo\n\n## Setup\n\nRun `pip install`\n\n## Usage\n\nCall `main()`",
        "expected_count": 3,
        "expected_contents": {1: "pip install", 2: "main()"},
    },
    {
        "name": "adjacent headers produce empty chunk",
        "text": "# Title\n\n## First\n\n## Second\n\nContent",
        "expected_count": 3,
        "expected_contents": {1: "First"},
    },
]

SIZE_CASES = [
    {
        "name": "splits text into fixed-size chunks",
        "text": "abcdefghij",
        "chunk_size": 4,
        "chunk_overlap": 0,
        "expected_count": 3,
        "expected_contents": {0: "abcd", 1: "efgh", 2: "ij"},
    },
    {
        "name": "overlap produces shared characters between chunks",
        "text": "abcdefghij",
        "chunk_size": 5,
        "chunk_overlap": 2,
        "expected_count": 3,
        "expected_contents": {0: "abcde", 1: "defgh", 2: "ghij"},
    },
    {
        "name": "text shorter than chunk size returns one chunk",
        "text": "short",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "expected_count": 1,
        "expected_contents": {0: "short"},
    },
    {
        "name": "empty string returns single empty chunk",
        "text": "",
        "chunk_size": 50,
        "chunk_overlap": 10,
        "expected_count": 1,
        "expected_contents": {0: ""},
    },
    {
        "name": "no overlap produces non-overlapping chunks",
        "text": "123456789",
        "chunk_size": 3,
        "chunk_overlap": 0,
        "expected_count": 3,
        "expected_contents": {0: "123", 1: "456", 2: "789"},
    },
]


SENTENCE_CASES = [
    {
        "name": "groups sentences with overlap",
        "text": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
        "max_sentences": 3,
        "overlap": 1,
        "expected_count": 3,
        "expected_contents": {
            0: "First sentence. Second sentence. Third sentence.",
            1: "Third sentence. Fourth sentence. Fifth sentence.",
        },
    },
    {
        "name": "single sentence returns one chunk",
        "text": "Only one sentence here.",
        "max_sentences": 3,
        "overlap": 1,
        "expected_count": 1,
        "expected_contents": {0: "Only one sentence here."},
    },
    {
        "name": "fewer sentences than max returns one chunk",
        "text": "Short. Doc.",
        "max_sentences": 5,
        "overlap": 1,
        "expected_count": 1,
        "expected_contents": {0: "Short. Doc."},
    },
    {
        "name": "splits on exclamation and question marks",
        "text": "What happened? It broke! Then we fixed it. All good now.",
        "max_sentences": 2,
        "overlap": 0,
        "expected_count": 2,
        "expected_contents": {
            0: "What happened? It broke!",
            1: "Then we fixed it. All good now.",
        },
    },
    {
        "name": "zero overlap produces non-overlapping chunks",
        "text": "One. Two. Three. Four.",
        "max_sentences": 2,
        "overlap": 0,
        "expected_count": 2,
        "expected_contents": {0: "One. Two.", 1: "Three. Four."},
    },
    {
        "name": "empty string returns single empty chunk",
        "text": "",
        "max_sentences": 3,
        "overlap": 1,
        "expected_count": 1,
        "expected_contents": {0: ""},
    },
]


@pytest.mark.parametrize("case", SENTENCE_CASES, ids=lambda c: c["name"])
def test_chunk_by_sentence(case: dict[str, Any]) -> None:
    chunks = chunk_by_sentence(
        case["text"],
        max_sentences=case["max_sentences"],
        overlap=case["overlap"],
    )

    assert len(chunks) == case["expected_count"]
    for index, substring in case.get("expected_contents", {}).items():
        assert substring in chunks[index], f"chunk[{index}] missing '{substring}'"


@pytest.mark.parametrize("case", SECTION_CASES, ids=lambda c: c["name"])
def test_chunk_by_section(case: dict[str, Any]) -> None:
    chunks = chunk_by_section(case["text"])

    assert len(chunks) == case["expected_count"]
    for index, substring in case.get("expected_contents", {}).items():
        assert substring in chunks[index], f"chunk[{index}] missing '{substring}'"


@pytest.mark.parametrize("case", SIZE_CASES, ids=lambda c: c["name"])
def test_chunk_by_size(case: dict[str, Any]) -> None:
    chunks = chunk_by_size(
        case["text"],
        chunk_size=case["chunk_size"],
        chunk_overlap=case["chunk_overlap"],
    )

    assert len(chunks) == case["expected_count"]
    for index, substring in case.get("expected_contents", {}).items():
        assert substring in chunks[index], f"chunk[{index}] missing '{substring}'"
