import uuid
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from repo_lens.rag.base_vector_index import BaseVectorIndex
from repo_lens.rag.types import IndexedDocument


class QdrantVectorIndex(BaseVectorIndex):
    def __init__(
        self,
        *,
        collection_name: str,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
        url: str,
        api_key: str,
        vector_size: int = 1024,
    ) -> None:
        super().__init__(embedding_fn=embedding_fn)
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        self._client.create_payload_index(
            collection_name=collection_name,
            field_name="repo",
            field_schema="keyword",
        )

    @staticmethod
    def _doc_id(content: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))

    def _extract_payload(self, document: IndexedDocument) -> dict[str, str]:
        return {key: str(value) for key, value in document.items()}

    def _construct_point(
        self, vector: list[float], document: IndexedDocument
    ) -> PointStruct:
        content = document["content"]
        return PointStruct(
            id=self._doc_id(content),
            vector=vector,
            payload=self._extract_payload(document),
        )

    def _build_result_doc(self, payload: dict[str, Any]) -> IndexedDocument:
        doc: IndexedDocument = {"content": payload.get("content", "")}
        for key, value in payload.items():
            if key != "content":
                doc[key] = value  # type: ignore[literal-required]
        return doc

    def _store(self, vector: list[float], document: IndexedDocument) -> None:
        point = self._construct_point(vector=vector, document=document)
        self._client.upsert(collection_name=self._collection_name, points=[point])

    def _store_batch(
        self, vectors: list[list[float]], documents: list[IndexedDocument]
    ) -> None:
        points = []
        for vector, doc in zip(vectors, documents):
            point = self._construct_point(vector=vector, document=doc)
            points.append(point)

        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(
        self, *, query: Any, k: int = 1, repo: str | None = None
    ) -> list[tuple[IndexedDocument, float]]:
        if len(self) == 0:
            return []

        query_vector = self._resolve_query_vector(query)

        query_filter = None
        if repo is not None:
            query_filter = Filter(
                must=[FieldCondition(key="repo", match=MatchValue(value=repo))]
            )

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=k,
        )

        return [
            (self._build_result_doc(point.payload), point.score)
            for point in results.points
        ]

    def get_all_documents(self, limit: int = 100) -> list[IndexedDocument]:
        if len(self) == 0:
            return []

        documents: list[IndexedDocument] = []
        offset = None

        while True:
            results, offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                documents.append(self._build_result_doc(point.payload))

            if offset is None:
                break

        return documents

    def exists_in_collection(self, metadata_key: str, metadata_value: str) -> bool:
        results, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=metadata_key, match=MatchValue(value=metadata_value)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        return len(results) > 0

    def remove_from_collection(self, metadata_key: str, metadata_value: str) -> int:
        count_before = len(self)
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key=metadata_key, match=MatchValue(value=metadata_value)
                        )
                    ]
                )
            ),
        )

        return count_before - len(self)

    def __len__(self) -> int:
        return int(self._client.count(collection_name=self._collection_name).count)
