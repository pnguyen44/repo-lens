import json
from unittest.mock import MagicMock

from repo_lens.evals.prompt.prompt_eval import PromptEvaluator, _describe_tool_call
from repo_lens.providers.chat_client import ChatResponse, ToolCall


def test_generate_dataset_returns_empty_after_retries() -> None:
    mock_client = MagicMock()
    mock_client.chat_json.return_value = ChatResponse(text="not valid json")

    evaluator = PromptEvaluator(client=mock_client)
    result = evaluator.generate_dataset("test prompt")

    assert result == []
    assert mock_client.chat_json.call_count == 3


def test_generate_dataset_parses_valid_json() -> None:
    mock_client = MagicMock()
    mock_client.chat_json.return_value = ChatResponse(
        text=json.dumps([{"input": "hello", "criteria": ["is polite"]}])
    )

    evaluator = PromptEvaluator(client=mock_client)
    result = evaluator.generate_dataset("test prompt", total_tests=1)

    assert len(result) == 1
    assert result[0]["input"] == "hello"


def test_grade_output_returns_fallback_on_bad_json() -> None:
    mock_client = MagicMock()
    mock_client.chat_json.return_value = ChatResponse(text="not json")

    evaluator = PromptEvaluator(client=mock_client)
    result = evaluator.grade_output(
        {"input": "hi", "criteria": ["is polite"]}, "hello there"
    )

    assert result["score"] == 0
    assert result["reasoning"] == "Failed to parse grade"


def test_run_eval_does_nothing_on_empty_test_cases() -> None:
    mock_client = MagicMock()
    evaluator = PromptEvaluator(client=mock_client)

    evaluator.run_eval("test prompt", [])

    mock_client.chat.assert_not_called()


def test_describe_tool_call_formats_delegation() -> None:
    tool_call = ToolCall(
        id="1",
        name="delegate_to_agent",
        input={"agent_name": "rag", "task": "find auth docs"},
    )
    assert _describe_tool_call(tool_call) == (
        "I will delegate this task to the rag agent: find auth docs"
    )


def test_run_prompt_describes_tool_call_response() -> None:
    mock_client = MagicMock()
    mock_client.chat.return_value = ChatResponse(
        text="",
        stop_reason="tool_use",
        tool_calls=[
            ToolCall(
                id="1",
                name="delegate_to_agent",
                input={"agent_name": "github", "task": "list PRs"},
            )
        ],
    )

    evaluator = PromptEvaluator(
        client=mock_client, tools=[{"name": "delegate_to_agent"}]
    )
    output = evaluator.run_prompt(
        prompt="delegate tasks",
        test_case={"input": "Show me PRs"},
    )

    assert output == "I will delegate this task to the github agent: list PRs"
    mock_client.chat.assert_called_once()
    assert mock_client.chat.call_args.kwargs["tools"] == [{"name": "delegate_to_agent"}]


def test_run_prompt_describes_all_tool_calls() -> None:
    mock_client = MagicMock()
    mock_client.chat.return_value = ChatResponse(
        text="",
        stop_reason="tool_use",
        tool_calls=[
            ToolCall(
                id="1",
                name="delegate_to_agent",
                input={"agent_name": "github", "task": "list PRs"},
            ),
            ToolCall(
                id="2",
                name="delegate_to_agent",
                input={"agent_name": "rag", "task": "find auth docs"},
            ),
        ],
    )

    evaluator = PromptEvaluator(
        client=mock_client, tools=[{"name": "delegate_to_agent"}]
    )
    output = evaluator.run_prompt(
        prompt="delegate tasks",
        test_case={"input": "Show me PRs and explain auth"},
    )

    assert output == (
        "I will delegate this task to the github agent: list PRs\n"
        "I will delegate this task to the rag agent: find auth docs"
    )
