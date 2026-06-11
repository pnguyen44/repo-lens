import logging
from embeddings import Embedder, VoyageEmbedder
from mcp_client import MCPClient, create_github_client
import asyncio
from config import create_config
from mcp.types import EmbeddedResource, TextResourceContents
from vector_index import VectorIndex
from voyageai.client import Client as VoyageClient
from chunker import chunk_by_section

logger = logging.getLogger(__name__)

NUM_RESULTS = 2
PREVIEW_LENGTH = 200


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


async def index_repo(
    mcp_client: MCPClient, embedder: Embedder, index: VectorIndex, owner: str, repo: str
) -> int:
    """Fetch repo README, chunk it, embed each chunk, and store in the index."""
    readme = await fetch_readme(mcp_client=mcp_client, owner=owner, repo=repo)
    chunks = [c for c in chunk_by_section(readme) if c.strip()]

    if not chunks:
        return 0

    repo_name = f"{owner}/{repo}"
    logger.info("Indexing %s", repo_name)

    vectors = embedder.generate_embeddings(chunks)
    for vector, chunk in zip(vectors, chunks):
        section = chunk.split("\n", 1)[0].lstrip("# ").strip()
        index.add_vector(
            vector, {"content": chunk, "repo": repo_name, "section": section}
        )
    return len(index.vectors)


async def main() -> None:
    config = create_config()
    embedder = VoyageEmbedder(VoyageClient())
    index = VectorIndex()
    owner = "openshift-hyperfleet"
    repo = "hyperfleet-api"

    async with create_github_client(config.github_token) as mcp_client:
        count = await index_repo(
            mcp_client=mcp_client,
            embedder=embedder,
            index=index,
            owner=owner,
            repo=repo,
        )

        print(f"Indexed {count} chunks from {owner}/{repo}\n")

        query = "How does the API work?"
        print(f'Query: "{query}"\n')
        query_vector = embedder.generate_embeddings([query])[0]
        results = index.search(query_vector, k=NUM_RESULTS)

        for i, (doc, distance) in enumerate(results, 1):
            print(f"Result {i} (distance: {distance:.4f})")
            print(f"Repo: {doc['repo']}")
            print("content:")
            print(doc["content"][:PREVIEW_LENGTH])
            print("──────────────────────────────")
            print()


if __name__ == "__main__":
    asyncio.run(main())
