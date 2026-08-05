from typing import NotRequired, TypedDict


class IndexedDocument(TypedDict):
    content: str
    repo: NotRequired[str]
    path: NotRequired[str]
    file_key: NotRequired[str]
    section: NotRequired[str]
    url: NotRequired[str]
