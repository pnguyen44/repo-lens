import asyncio
import json
import logging

from mcp.types import EmbeddedResource, TextContent, TextResourceContents
from voyageai.client import Client as VoyageClient

from repo_lens.core.config import create_config
from repo_lens.core.mcp_client import MCPClient, create_github_client
from repo_lens.core.repo_context import RepoContext
from repo_lens.rag.bm25_index import BM25Index
from repo_lens.rag.chunker import chunk_by_section
from repo_lens.rag.embeddings import InputType, VoyageEmbedder
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.types import IndexedDocument
from repo_lens.rag.vector_index import VectorIndex

logger = logging.getLogger(__name__)

NUM_RESULTS = 2
PREVIEW_LENGTH = 200

EXCLUDE_FILES = {"CLAUDE.md", "AGENTS.md"}


def file_path_to_chunks(
    repo_context: RepoContext, path: str, text: str
) -> list[IndexedDocument]:
    chunks = [c for c in chunk_by_section(text) if c.strip()]
    documents: list[IndexedDocument] = []

    for chunk in chunks:
        section = chunk.split("\n", 1)[0].lstrip("# ").strip()
        anchor = to_github_anchor(section)
        url = (
            f"https://github.com/{repo_context.owner}/{repo_context.repo}"
            f"/blob/main/{path}#{anchor}"
        )

        documents.append(
            {
                "content": chunk,
                "repo": repo_context.key,
                "path": path,
                "section": section,
                "url": url,
            }
        )
    return documents


class RepoContentFetcher:
    def __init__(self, mcp_client: MCPClient, repo_context: RepoContext) -> None:
        self.mcp_client = mcp_client
        self.repo_context = repo_context

    async def fetch_file(self, path: str) -> str:
        text = ""

        result = await self.mcp_client.call_tool(
            "get_file_contents",
            {
                "owner": self.repo_context.owner,
                "repo": self.repo_context.repo,
                "path": path,
            },
        )

        for item in result.content:
            if isinstance(item, EmbeddedResource) and isinstance(
                item.resource, TextResourceContents
            ):
                text = item.resource.text

        if not text:
            raise ValueError(
                f"No context found for {self.repo_context.owner}/{self.repo_context.repo} on {path}"
            )

        return text

    async def fetch_md_file_list(self) -> list[str]:
        return await self._walk_dir("")

    async def _walk_dir(self, path: str) -> list[str]:
        result = await self.mcp_client.call_tool(
            "get_file_contents",
            {
                "owner": self.repo_context.owner,
                "repo": self.repo_context.repo,
                "path": path,
            },
        )

        entries = []

        for item in result.content:
            if isinstance(item, TextContent):
                entries = json.loads(item.text)

        md_files: list[str] = []

        for entry in entries:
            if entry["type"] == "file" and entry["path"].endswith(".md"):
                if entry["path"].split("/")[-1] not in EXCLUDE_FILES:
                    md_files.append(entry["path"])
            elif entry["type"] == "dir":
                md_files.extend(await self._walk_dir(entry["path"]))

        return md_files

    async def fetch_file_chunks(self, path: str) -> list[IndexedDocument]:
        text = await self.fetch_file(path)
        return file_path_to_chunks(repo_context=self.repo_context, path=path, text=text)

    async def fetch_repo_chunks(self) -> list[IndexedDocument]:
        repo_name = self.repo_context.key
        logger.info("Indexing %s", repo_name)

        file_list = await self.fetch_md_file_list()

        documents: list[IndexedDocument] = []

        for file in file_list:
            documents.extend(await self.fetch_file_chunks(file))

        return documents


def to_github_anchor(section: str) -> str:
    return section.lower().replace(" ", "-").replace(".", "").replace("/", "")


async def main() -> None:
    config = create_config()
    embedder = VoyageEmbedder(VoyageClient(), model=config.voyage_embed_model)
    repo_context = RepoContext(owner="openshift-hyperfleet", repo="hyperfleet-api")

    vector_index = VectorIndex(
        embedding_fn=lambda texts: embedder.generate_embeddings(
            texts=texts, input_type=InputType.DOCUMENT
        )
    )

    bm25 = BM25Index()

    retriever = HybridRetriever(vector_index, bm25)

    async with create_github_client(config.github_token) as mcp_client:
        fetcher = RepoContentFetcher(mcp_client=mcp_client, repo_context=repo_context)
        documents = await fetcher.fetch_repo_chunks()

    count = len(documents)

    print(f"Indexed {count} chunks from {repo_context.key}\n")

    retriever.add_documents(documents)

    query = "How does the API work?"
    print(f'Query: "{query}"\n')

    results = retriever.search(query_text=query, k=NUM_RESULTS)

    for i, (doc, distance) in enumerate(results, 1):
        print(f"Result {i} (distance: {distance:.4f})")
        print(f"Repo: {doc['repo']}")
        print("content:")
        print(doc["content"][:PREVIEW_LENGTH])
        print("──────────────────────────────")
        print()


if __name__ == "__main__":
    asyncio.run(main())
