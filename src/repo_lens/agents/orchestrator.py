import copy
import logging
from typing import Any, Protocol

from repo_lens.agents.agent import Agent, AgentName
from repo_lens.agents.chat import OnToolInputCallback, OnToolStartCallback
from repo_lens.core.repo_context import RepoContext
from repo_lens.providers.chat_client import ChatClientProtocol

logger = logging.getLogger(__name__)

delegate_to_agent_schema = {
    "name": "delegate_to_agent",
    "description": "Delegate a task to a specialist agent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "The specialist agent to delegate to.",
            },
            "task": {
                "type": "string",
                "description": "The task for the agent to perform.",
            },
        },
        "required": ["agent_name", "task"],
    },
}

PLANNER_SYSTEM_PROMPT_TEMPLATE = """
<role>
You are a planning agent. Your job is to break down and delegate tasks to specialist agents using the delegate_to_agent tool.
</role>

Available agents:
{agents}

Rules:
- Always delegate to an agent. Do not answer directly.
- After receiving agent results, synthesize a final answer for the user.
- When citing docs or files, use full GitHub URLs only
  (`https://github.com/{{owner}}/{{repo}}/blob/main/...`).
- Prefer source URLs from specialist results / retrieval titles.
- Never use repo-relative paths (`docs/foo.md`) or bare `[Section]` links.
"""


def delegation_label(agent_name: str) -> str:
    return f"Delegating to {agent_name} agent"


class OnDelegateCallback(Protocol):
    async def __call__(self, agent_name: str, task: str) -> None: ...


class OnTextCallback(Protocol):
    def __call__(self, text: str) -> None: ...


class Orchestrator:
    def __init__(
        self,
        *,
        agents: dict[AgentName, Agent],
        chat_client: ChatClientProtocol,
        max_delegations: int = 5,
    ) -> None:
        self.agents = agents
        self.chat_client = chat_client
        self.messages: list[Any] = []
        self.max_delegations = max_delegations

        agent_names = [name.value for name in agents.keys()]
        agent_lines = [
            f"- {name.value}: {agent.description}" for name, agent in agents.items()
        ]

        schema: dict[str, Any] = copy.deepcopy(delegate_to_agent_schema)
        schema["input_schema"]["properties"]["agent_name"]["enum"] = agent_names
        self.tools = [schema]

        self.system_prompt = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
            agents="\n".join(agent_lines)
        )

    def _stream(
        self,
        on_text: OnTextCallback | None = None,
        repo_context: RepoContext | None = None,
    ) -> tuple[Any, str]:
        system = self.system_prompt + (
            repo_context.prompt_suffix() if repo_context else ""
        )
        text = ""
        with self.chat_client.chat_stream(
            messages=self.messages,
            tools=self.tools,
            system=system,
            web_search=False,
        ) as stream:
            for chunk in stream:
                if chunk.type == "text":
                    text += chunk.text
                    if on_text:
                        on_text(chunk.text)
            response = stream.get_final_message()
            if response.usage:
                self.chat_client.record_usage(response.usage)
        self.chat_client.add_assistant_message(messages=self.messages, message=response)

        return response, text

    async def run(
        self,
        query: str,
        repo_context: RepoContext | None = None,
        on_delegate: OnDelegateCallback | None = None,
        on_text: OnTextCallback | None = None,
        on_tool_start: OnToolStartCallback | None = None,
        on_tool_input: OnToolInputCallback | None = None,
    ) -> str:
        self.chat_client.add_user_message(messages=self.messages, content=query)

        delegations = 0

        while True:
            response, text = self._stream(on_text=on_text, repo_context=repo_context)

            if response.stop_reason != "tool_use":
                return text

            for tool in response.tool_calls:
                delegations += 1
                if delegations > self.max_delegations:
                    return text

                agent_name = tool.input["agent_name"]
                task = tool.input["task"]

                try:
                    agent = self.agents.get(AgentName(agent_name))
                except ValueError:
                    agent = None
                if not agent:
                    break

                label = delegation_label(agent_name)
                logger.info("%s: %s", label, task)
                if on_delegate:
                    await on_delegate(agent_name=agent_name, task=task)

                result = await agent.run(
                    task,
                    repo_context=repo_context,
                    on_tool_start=on_tool_start,
                    on_tool_input=on_tool_input,
                )

                tool_result = {
                    "tool_use_id": tool.id,
                    "type": "tool_result",
                    "name": tool.name,
                    "content": result,
                }

                self.chat_client.add_user_message(
                    messages=self.messages, content=[tool_result]
                )
