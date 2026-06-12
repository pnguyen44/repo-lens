import logging
from typing import Protocol
import time

from voyageai.client import Client as VoyageClient
from voyageai.error import RateLimitError

from enum import Enum

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"


class Embedder(Protocol):
    def generate_embeddings(
        self, texts: list[str], input_type: InputType = InputType.QUERY
    ) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self, client: VoyageClient, model: str = "voyage-3-large") -> None:
        self.client = client
        self.model = model

    def generate_embeddings(
        self, texts: list[str], input_type: InputType = InputType.QUERY
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")

        if any(t == "" for t in texts):
            raise ValueError("texts must not contain empty strings")

        for attempt in range(2):
            try:
                result = self.client.embed(
                    texts, model=self.model, input_type=input_type
                )
                return [list(float(x) for x in vec) for vec in result.embeddings]
            except RateLimitError:
                if attempt == 0:
                    logger.warning("Rate limited, waiting 60s...")
                    time.sleep(60)
        raise RateLimitError("Still rate limited after retry")
