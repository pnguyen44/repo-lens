from typing import Protocol

from voyageai.client import Client as VoyageClient


class Embedder(Protocol):
    def generate_embedding(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    def __init__(self, client: VoyageClient) -> None:
        self.client = client

    def generate_embedding(
        self, text: str, model: str = "voyage-3-large", input_type: str = "query"
    ) -> list[float]:
        result = self.client.embed([text], model=model, input_type=input_type)
        return [float(x) for x in result.embeddings[0]]
