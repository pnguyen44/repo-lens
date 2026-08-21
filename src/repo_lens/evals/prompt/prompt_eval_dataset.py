"""Test cases for repo-lens's own prompts, keyed by prompt name.

Criteria are text-based: they grade whether the model's response signals the
right delegation intent. Tool-call responses are converted to text before grading.
"""

from repo_lens.evals.prompt.types import PromptTestCase

PROMPT_EVAL_CASES: dict[str, list[PromptTestCase]] = {
    "planner": [
        {
            "input": "How does authentication work in this codebase?",
            "criteria": [
                "States that it will delegate the task to the rag agent",
                "Does not attempt to explain authentication directly",
            ],
            "expected_agent": "rag",
        },
        {
            "input": "Show me the open pull requests.",
            "criteria": [
                "States that it will delegate the task to the github agent",
                "Does not fabricate an answer about pull requests",
            ],
            "expected_agent": "github",
        },
        {
            "input": "What's the database schema?",
            "criteria": [
                "States that it will delegate the task to the rag agent",
                "Does not attempt to describe a schema directly",
            ],
            "expected_agent": "rag",
        },
        {
            "input": "What time is it?",
            "criteria": [
                "Does not delegate to any agent",
                "Explains it cannot answer this (no agent has that capability)",
            ],
            "expected_agent": None,
        },
        {
            "input": "Hello, how are you?",
            "criteria": [
                "Responds conversationally without delegating to any agent",
            ],
            "expected_agent": None,
        },
    ],
}
