import copy
import logging
from typing import Any, Protocol

from agent import Agent, AgentName
from chat_client import ChatClient

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
"""


class OnDelegateCallback(Protocol):
    async def __call__(self, agent_name: str, task: str) -> None: ...


class Orchestrator:
    def __init__(
        self,
        *,
        agents: dict[AgentName, Agent],
        chat_client: ChatClient[Any],
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

    async def run(
        self, query: str, on_delegate: OnDelegateCallback | None = None
    ) -> str:
        self.chat_client.add_user_message(messages=self.messages, content=query)

        delegations = 0

        while True:
            response = self.chat_client.chat(
                messages=self.messages,
                tools=self.tools,
                system=self.system_prompt,
                web_search=False,
            )

            self.chat_client.add_assistant_message(
                messages=self.messages, message=response
            )

            if response.stop_reason != "tool_use":
                return response.text

            for tool in response.tool_calls:
                delegations += 1
                if delegations > self.max_delegations:
                    return response.text

                agent_name = tool.input["agent_name"]
                task = tool.input["task"]

                try:
                    agent = self.agents.get(AgentName(agent_name))
                except ValueError:
                    agent = None
                if not agent:
                    break

                logger.info("Delegating to %s: %s", agent_name, task)
                if on_delegate:
                    await on_delegate(agent_name=agent_name, task=task)

                result = await agent.run(task)

                tool_result = {
                    "tool_use_id": tool.id,
                    "type": "tool_result",
                    "name": tool.name,
                    "content": result,
                }

                self.chat_client.add_user_message(
                    messages=self.messages, content=[tool_result]
                )
