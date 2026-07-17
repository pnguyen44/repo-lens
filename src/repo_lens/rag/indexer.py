import logging
from typing import Any
from repo_lens.rag.bm25_index import BM25Index
from repo_lens.rag.embeddings import VoyageEmbedder, InputType
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.core.mcp_client import MCPClient, create_github_client
import asyncio
from repo_lens.core.config import create_config
from mcp.types import EmbeddedResource, TextResourceContents
from repo_lens.rag.vector_index import VectorIndex
from voyageai.client import Client as VoyageClient
from repo_lens.rag.chunker import chunk_by_section

logger = logging.getLogger(__name__)

NUM_RESULTS = 2
PREVIEW_LENGTH = 200


def to_github_anchor(section: str) -> str:
    return section.lower().replace(" ", "-").replace(".", "").replace("/", "")


async def fetch_readme(mcp_client: MCPClient, owner: str, repo: str) -> str:
    """Fetch a repo's README.md content via GitHub MCP."""
    result = await mcp_client.call_tool(
        "get_file_contents", {"owner": owner, "repo": repo, "path": "README.md"}
    )
    readme_text = ""
    for item in result.content:
        if isinstance(item, EmbeddedResource) and isinstance(
            item.resource, TextResourceContents
        ):
            readme_text = item.resource.text

    if not readme_text:
        raise ValueError(f"No README context found for {owner}/{repo}")
    return readme_text


async def fetch_repo_chunks(
    github_mcp: MCPClient, owner: str, repo: str
) -> list[dict[str, Any]]:
    readme = await fetch_readme(mcp_client=github_mcp, owner=owner, repo=repo)
    chunks = [c for c in chunk_by_section(readme) if c.strip()]

    if not chunks:
        return []

    repo_name = f"{owner}/{repo}"
    logger.info("Indexing %s", repo_name)

    documents = []
    for chunk in chunks:
        section = chunk.split("\n", 1)[0].lstrip("# ").strip()
        anchor = to_github_anchor(section)
        url = f"https://github.com/{owner}/{repo}/blob/main/README.md#{anchor}"
        documents.append(
            {"content": chunk, "repo": repo_name, "section": section, "url": url}
        )

    return documents


async def main() -> None:
    config = create_config()
    embedder = VoyageEmbedder(VoyageClient(), model=config.voyage_embed_model)
    owner = "openshift-hyperfleet"
    repo = "hyperfleet-api"

    vector_index = VectorIndex(
        embedding_fn=lambda texts: embedder.generate_embeddings(
            texts=texts, input_type=InputType.DOCUMENT
        )
    )

    bm25 = BM25Index()

    retriever = HybridRetriever(vector_index, bm25)

    async with create_github_client(config.github_token) as mcp_client:
        documents = await fetch_repo_chunks(
            github_mcp=mcp_client,
            owner=owner,
            repo=repo,
        )

    count = len(documents)

    print(f"Indexed {count} chunks from {owner}/{repo}\n")

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
