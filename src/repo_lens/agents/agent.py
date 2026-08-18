from enum import Enum

from repo_lens.agents.chat import (
    Chat,
    OnFileFetchedCallback,
    OnToolInputCallback,
    OnToolStartCallback,
)
from repo_lens.core.config import DEFAULT_MAX_TOOL_ITERATIONS
from repo_lens.core.mcp_client import MCPClient
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.chat_client import ChatClientProtocol
from repo_lens.rag.hybrid_retriever import HybridRetriever
from repo_lens.rag.reranker import Reranker


class AgentName(Enum):
    GITHUB = (
        "github",
        "Answers questions about GitHub repositories (PRs, issues, files, commits).",
    )
    RAG = (
        "rag",
        "Answers questions about indexed codebases using retrieved context.",
    )

    description: str

    def __new__(cls, value: str, description: str = "") -> "AgentName":
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj


WIRED_AGENTS: tuple[AgentName, ...] = (AgentName.GITHUB, AgentName.RAG)


class Agent:
    def __init__(self, *, name: AgentName, chat: Chat) -> None:
        self.name = name
        self.chat = chat
        self.description = name.description

    async def run(
        self,
        task: str,
        *,
        repo_context: RepoContext | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        on_tool_input: OnToolInputCallback | None = None,
        on_file_fetched: OnFileFetchedCallback | None = None,
    ) -> str:
        self.chat.messages = []
        return await self.chat.run(
            query=task,
            repo_context=repo_context,
            on_tool_start=on_tool_start,
            on_tool_input=on_tool_input,
            on_file_fetched=on_file_fetched,
        )


def create_github_agent(
    chat_client: ChatClientProtocol,
    github_mcp: MCPClient,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> Agent:
    chat = Chat(
        chat_client=chat_client,
        mcp_clients={"github": github_mcp},
        system_prompt="You answer questions about Github repositories using the available tools.",
        web_search=False,
        max_tool_iterations=max_tool_iterations,
    )

    return Agent(name=AgentName.GITHUB, chat=chat)


def create_rag_agent(
    chat_client: ChatClientProtocol,
    hybrid_retriever: HybridRetriever,
    reranker: Reranker | None = None,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> Agent:
    chat = Chat(
        chat_client=chat_client,
        system_prompt=(
            "You answer questions about codebases using the retrieved context. "
            "When you reference a section or file, link with the full GitHub URL "
            "from the source title (https://github.com/...). "
            "Do not use relative paths or markdown links without an absolute URL."
        ),
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        web_search=False,
        max_tool_iterations=max_tool_iterations,
    )

    return Agent(name=AgentName.RAG, chat=chat)
