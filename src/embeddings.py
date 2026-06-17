import logging
from typing import Protocol

from voyageai.client import Client as VoyageClient

from enum import Enum

from voyage import voyage_retry

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"


class Embedder(Protocol):
    def generate_embeddings(
        self, texts: list[str], input_type: InputType = InputType.QUERY
    ) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self, client: VoyageClient, model: str) -> None:
        self.client = client
        self.model = model

    def generate_embeddings(
        self, texts: list[str], input_type: InputType = InputType.QUERY
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")

        if any(t == "" for t in texts):
            raise ValueError("texts must not contain empty strings")

        result = voyage_retry(
            fn=lambda: self.client.embed(
                texts, model=self.model, input_type=input_type
            ),
            retires=2,
        )

        return [list(float(x) for x in vec) for vec in result.embeddings]
