from typing import NamedTuple, NotRequired, TypedDict


class FetchedFile(NamedTuple):
    text: str
    sha: str


class IndexedDocument(TypedDict):
    content: str
    repo: NotRequired[str]
    path: NotRequired[str]
    file_key: NotRequired[str]
    section: NotRequired[str]
    url: NotRequired[str]
    sha: NotRequired[str]


def validate_document(document: object, index: int | None = None) -> None:
    prefix = f"Document at index {index}: " if index is not None else ""

    if not isinstance(document, dict):
        raise TypeError(f"{prefix}Document must be a dictionary.")

    if "content" not in document:
        raise ValueError(f"{prefix}Document dictionary must contain a 'content' key.")

    if not isinstance(document["content"], str):
        raise TypeError(f"{prefix}Document 'content' must be a string.")
