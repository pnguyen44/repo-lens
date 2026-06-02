import re


def chunk_by_section(text: str) -> list[str]:
    pattern = r"\n## "
    return re.split(pattern, text)


def chunk_by_size(
    text: str, chunk_size: int = 150, chunk_overlap: int = 20
) -> list[str]:
    if not text:
        return [""]

    chunks = []
    start_idx: int = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))

        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)
        start_idx = end_idx - chunk_overlap if end_idx < len(text) else len(text)

    return chunks


def chunk_by_sentence(text: str, max_sentences: int = 5, overlap: int = 1) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    start_idx = 0
    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences, len(sentences))

        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))

        start_idx += max_sentences - overlap

        if start_idx < 0:
            start_idx = 0

    return chunks
