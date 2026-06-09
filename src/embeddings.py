import logging
from typing import Protocol
import time

from voyageai.client import Client as VoyageClient
from voyageai.error import RateLimitError

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self, client: VoyageClient) -> None:
        self.client = client

    def generate_embeddings(
        self, texts: list[str], model: str = "voyage-3-large", input_type: str = "query"
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")

        if any(t == "" for t in texts):
            raise ValueError("texts must not contain empty strings")

        for attempt in range(2):
            try:
                result = self.client.embed(texts, model=model, input_type=input_type)
                return [list(float(x) for x in vec) for vec in result.embeddings]
            except RateLimitError:
                if attempt == 0:
                    logger.warning("Rate limited, waiting 60s...")
                    time.sleep(60)
        raise RateLimitError("Still rate limited after retry")
