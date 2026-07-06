from typing import Any, Callable, Optional
import math

from enum import Enum
from base_vector_index import BaseVectorIndex


class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class VectorIndex(BaseVectorIndex):
    def __init__(
        self,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        super().__init__(embedding_fn=embedding_fn)
        self.vectors: list[list[float]] = []
        self.documents: list[dict[str, Any]] = []
        self._vector_dim: Optional[int] = None

        self._distance_functions = {
            DistanceMetric.COSINE: self._cosine_distance,
            DistanceMetric.EUCLIDEAN: self._euclidean_distance,
        }

        if distance_metric not in self._distance_functions:
            raise ValueError(
                f"Unknown metric: '{distance_metric}'. Choose from {list(self._distance_functions.keys())}"
            )

        self._distance_metric = distance_metric

    def add_vector(self, vector: list[float], document: dict[str, Any]) -> None:
        if not vector:
            raise ValueError("Vector must not be empty")
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    def _store(self, vector: list[float], document: dict[str, Any]) -> None:
        self.add_vector(vector=vector, document=document)

    def _store_batch(
        self, vectors: list[list[float]], documents: list[dict[str, Any]]
    ) -> None:
        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def _validate_search(self, query_vector: list[float], k: int) -> None:
        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expect {self._vector_dim}, got {len(query_vector)}"
            )
        if k <= 0:
            raise ValueError("k must be a positive integer.")

    def _get_distance_fn(self) -> Callable[[list[float], list[float]], float]:
        return self._distance_functions[self._distance_metric]

    def _find_nearest(
        self, query_vector: list[float], k: int
    ) -> list[tuple[dict[str, Any], float]]:
        dist_func = self._get_distance_fn()

        distances = [
            (dist_func(query_vector, vec), doc)
            for vec, doc in zip(self.vectors, self.documents)
        ]

        distances.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in distances[:k]]

    def search(self, query: Any, k: int = 1) -> list[tuple[dict[str, Any], float]]:
        if not self.vectors or self._vector_dim is None:
            return []

        query_vector = self._resolve_query_vector(query)
        self._validate_search(query_vector, k)

        return self._find_nearest(query_vector, k)

    def _validate_vector_dimensions(self, vec1: list[float], vec2: list[float]) -> None:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

    def _euclidean_distance(self, vec1: list[float], vec2: list[float]) -> float:
        self._validate_vector_dimensions(vec1, vec2)
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _magnitude(self, vec: list[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def _dot_product(self, vec1: list[float], vec2: list[float]) -> float:
        self._validate_vector_dimensions(vec1, vec2)
        return sum(p * q for p, q in zip(vec1, vec2))

    def _cosine_distance(self, vec1: list[float], vec2: list[float]) -> float:
        self._validate_vector_dimensions(vec1, vec2)

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        dot_prod = self._dot_product(vec1, vec2)
        cosine_similarity = dot_prod / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric.value}', has_embedding_fn={self._embedding_fn is not None})"
