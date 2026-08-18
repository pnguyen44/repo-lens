"""Test cases for repo-lens's own prompts, keyed by prompt name.

The evaluator runs prompts without tools wired (see PromptEvaluator.run_prompt),
so criteria are text-based: they grade whether the model's response signals the
right *intent* (e.g. delegating), not whether a tool was actually invoked.
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
