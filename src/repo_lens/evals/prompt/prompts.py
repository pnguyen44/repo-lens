"""Repo-lens system prompts under evaluation.

Imports the live template from orchestrator.py (rather than copying the string)
so this eval stays in sync with the actual planner prompt used in production.
"""

from repo_lens.agents.orchestrator import PLANNER_SYSTEM_PROMPT_TEMPLATE

# Fixed agent list so the eval doesn't need to construct live Agent objects,
# which require MCP clients / a hybrid retriever to build.
MOCK_AGENTS = """- github: Answers questions about GitHub repositories (PRs, issues, files, commits).
- rag: Answers questions about indexed codebases using retrieved context."""

PLANNER_PROMPT = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(agents=MOCK_AGENTS)

PROMPTS: dict[str, str] = {
    "planner": PLANNER_PROMPT,
}
