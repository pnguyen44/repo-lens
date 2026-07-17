from enum import Enum
from typing import Any

from repo_lens.agents.chat import Chat
from repo_lens.providers.chat_client import ChatClient
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.core.mcp_client import MCPClient
from repo_lens.rag.reranker import Reranker


class AgentName(Enum):
    GITHUB = "github"
    RAG = "rag"


class Agent:
    def __init__(self, *, name: AgentName, chat: Chat, description: str) -> None:
        self.name = name
        self.chat = chat
        self.description = description

    async def run(self, task: str) -> str:
        self.chat.messages = []
        return await self.chat.run(task)


def create_github_agent(chat_client: ChatClient[Any], github_mcp: MCPClient) -> Agent:
    chat = Chat(
        chat_client=chat_client,
        mcp_clients={"github": github_mcp},
        system_prompt="You answer questions about Github repositories using the available tools.",
        web_search=False,
    )

    return Agent(
        name=AgentName.GITHUB,
        chat=chat,
        description="Answers questions about GitHub repositories (PRs, issues, files, commits).",
    )


def create_rag_agent(
    chat_client: ChatClient[Any],
    hybrid_retriever: HybridRetriever,
    reranker: Reranker | None = None,
) -> Agent:
    chat = Chat(
        chat_client=chat_client,
        system_prompt="You answer questions about codebases using the retrieved context.",
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        web_search=False,
    )

    return Agent(
        name=AgentName.RAG,
        chat=chat,
        description="Answers questions about indexed codebases using retrieved context.",
    )
