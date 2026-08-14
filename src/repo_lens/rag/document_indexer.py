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

    async def sync_bm25_from_store(self) -> int:
        documents = await self._vector_index.get_all_documents()
        if documents:
            await self._retriever.reload_bm25(documents)

        return len(documents)

    @staticmethod
    def _file_key(repo: str, path: str) -> str:
        return f"{repo}:{path}"

    def _stamp_file_key(self, document: IndexedDocument) -> IndexedDocument:
        file_key = self._file_key(
            repo=document.get("repo", ""), path=document.get("path", "")
        )

        return {**document, "file_key": file_key}

    async def file_is_indexed(self, repo: str, path: str) -> bool:
        return await self._vector_index.exists_in_collection(
            "file_key", self._file_key(repo=repo, path=path)
        )

    async def index_file(
        self, repo: str, path: str, documents: list[IndexedDocument]
    ) -> int:
        if await self.file_is_indexed(repo=repo, path=path):
            logger.debug("index_file: %s already indexed, skipping", path)
            return 0

        stamped: list[IndexedDocument] = [
            self._stamp_file_key(doc) for doc in documents
        ]
        await self._retriever.add_documents(stamped)
        logger.info("Indexed %s (%d chunks)", path, len(stamped))
        return len(stamped)

    async def _sync_retriever(self) -> None:
        await self._retriever.reload_bm25(await self._vector_index.get_all_documents())

    async def clear_repo(self, repo: str) -> int:
        removed = await self._vector_index.remove_from_collection("repo", repo)

        await self._sync_retriever()

        logger.info("Cleared index for %s (%d chunks removed)", repo, removed)
        return removed

    async def search(
        self, *, query: str, k: int = 5, repo: str | None = None
    ) -> list[tuple[IndexedDocument, float]]:
        return await self._retriever.search(query_text=query, k=k, repo=repo)
