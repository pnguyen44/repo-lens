import logging

from repo_lens.rag.chroma_index import ChromaVectorIndex
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.types import IndexedDocument

logger = logging.getLogger(__name__)


class DocumentIndexer:
    def __init__(
        self, *, vector_index: ChromaVectorIndex, retriever: HybridRetriever
    ) -> None:
        self._vector_index = vector_index
        self._retriever = retriever

    def sync_bm25_from_store(self) -> int:
        documents = self._vector_index.get_all_documents()
        if documents:
            self._retriever.reload_bm25(documents)

        return len(documents)

    @staticmethod
    def _file_key(repo: str, path: str) -> str:
        return f"{repo}:{path}"

    def _stamp_file_key(self, document: IndexedDocument) -> IndexedDocument:
        file_key = self._file_key(
            repo=document.get("repo", ""), path=document.get("path", "")
        )

        return {**document, "file_key": file_key}

    def file_is_indexed(self, repo: str, path: str) -> bool:
        return self._vector_index.exists_in_collection(
            "file_key", self._file_key(repo=repo, path=path)
        )

    def index_file(self, repo: str, path: str, documents: list[IndexedDocument]) -> int:
        if self.file_is_indexed(repo=repo, path=path):
            logger.debug("index_file: %s already indexed, skipping", path)
            return 0

        stamped: list[IndexedDocument] = [
            self._stamp_file_key(doc) for doc in documents
        ]
        self._retriever.add_documents(stamped)
        logger.info("Indexed %s (%d chunks)", path, len(stamped))
        return len(stamped)

    def _sync_retriever(self) -> None:
        self._retriever.reload_bm25(self._vector_index.get_all_documents())

    def clear_repo(self, repo: str) -> int:
        removed = self._vector_index.remove_from_collection("repo", repo)

        self._sync_retriever()

        logger.info("Cleared index for %s (%d chunks removed)", repo, removed)
        return removed

    def search(
        self, *, query: str, k: int = 5, repo: str | None = None
    ) -> list[tuple[IndexedDocument, float]]:
        return self._retriever.search(query_text=query, k=k, repo=repo)
