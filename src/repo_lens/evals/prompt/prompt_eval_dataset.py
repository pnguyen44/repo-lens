"""Test cases for repo-lens's own prompts, keyed by prompt name.

Criteria are text-based: they grade whether the model's response signals the
right delegation intent. Tool-call responses are converted to text before grading.
"""

PROMPT_EVAL_CASES: dict[str, list[dict[str, object]]] = {
    "planner": [
        {
            "input": "How does authentication work in this codebase?",
            "criteria": [
                "States that it will delegate the task to the rag agent",
                "Does not attempt to explain authentication directly",
            ],
        },
        {
            "input": "Show me the open pull requests.",
            "criteria": [
                "States that it will delegate the task to the github agent",
                "Does not fabricate an answer about pull requests",
            ],
        },
        {
            "input": "What's the database schema?",
            "criteria": [
                "States that it will delegate the task to the rag agent",
                "Does not attempt to describe a schema directly",
            ],
        },
    ],
}
