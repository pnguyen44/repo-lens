from typing import Any

from repo_lens.agents.agent import WIRED_AGENTS
from repo_lens.agents.orchestrator import (
    PLANNER_SYSTEM_PROMPT_TEMPLATE,
    build_delegate_tool_schema,
)

# Fixed agent list so the eval doesn't need to construct live Agent objects,
# which require MCP clients / a hybrid retriever to build.
MOCK_AGENTS = "\n".join(f"- {name.value}: {name.description}" for name in WIRED_AGENTS)


PLANNER_PROMPT = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(agents=MOCK_AGENTS)

PROMPTS: dict[str, str] = {
    "planner": PLANNER_PROMPT,
}

agent_names = [name.value for name in WIRED_AGENTS]
EVAL_TOOLS: dict[str, list[dict[str, Any]]] = {
    "planner": [build_delegate_tool_schema(agent_names)]
}
