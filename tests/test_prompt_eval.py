import json
from unittest.mock import MagicMock

import pytest

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


def test_run_eval_prints_pass_rate_stats(capsys: pytest.CaptureFixture[str]) -> None:
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        ChatResponse(
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(id="1", name="delegate_to_agent", input={"agent_name": "rag"})
            ],
        ),
        ChatResponse(stop_reason="end_turn", text="no delegation"),
    ]

    evaluator = PromptEvaluator(client=mock_client)
    evaluator.run_eval(
        "test prompt",
        [
            {"input": "q1", "expected_agent": "rag"},
            {"input": "q2", "expected_agent": "rag"},
        ],
    )

    output = capsys.readouterr().out
    stats_start = output.rindex('{\n  "passed"')
    stats_end = output.index("}", stats_start) + 1
    assert json.loads(output[stats_start:stats_end]) == {
        "passed": 1,
        "total": 2,
        "pass_rate": 0.5,
    }
    assert "Pass rate: 50.0%" in output


def test_describe_tool_call_formats_delegation() -> None:
    tool_call = ToolCall(
        id="1",
        name="delegate_to_agent",
        input={"agent_name": "rag", "task": "find auth docs"},
    )
    assert _describe_tool_call(tool_call) == (
        "I will delegate this task to the rag agent: find auth docs"
    )


def test_run_prompt_returns_response_with_tool_call() -> None:
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

    assert output.stop_reason == "tool_use"
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].input["agent_name"] == "github"
    mock_client.chat.assert_called_once()
    assert mock_client.chat.call_args.kwargs["tools"] == [{"name": "delegate_to_agent"}]


def test_run_prompt_returns_response_with_multiple_tool_calls() -> None:
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

    assert [tc.input["agent_name"] for tc in output.tool_calls] == ["github", "rag"]


ROUTING_CASES = [
    {
        "name": "matches expected agent",
        "tool_calls": [
            ToolCall(
                id="1",
                name="delegate_to_agent",
                input={"agent_name": "rag", "task": "find auth docs"},
            )
        ],
        "expected_agent": "rag",
        "expected_score": 10,
    },
    {
        "name": "mismatches expected agent",
        "tool_calls": [
            ToolCall(
                id="1",
                name="delegate_to_agent",
                input={"agent_name": "github", "task": "find auth docs"},
            )
        ],
        "expected_agent": "rag",
        "expected_score": 0,
    },
    {
        "name": "no tool call made",
        "tool_calls": [],
        "expected_agent": "rag",
        "expected_score": 0,
    },
    {
        "name": "no tool call made and none expected",
        "tool_calls": [],
        "expected_agent": None,
        "expected_score": 10,
    },
]


@pytest.mark.parametrize("case", ROUTING_CASES, ids=lambda c: c["name"])
def test_run_test_case_routing(case) -> None:
    mock_client = MagicMock()
    mock_client.chat.return_value = ChatResponse(
        text="" if case["tool_calls"] else "Authentication uses JWT tokens.",
        stop_reason="tool_use" if case["tool_calls"] else "end_turn",
        tool_calls=case["tool_calls"],
    )

    evaluator = PromptEvaluator(client=mock_client)
    grade = evaluator.run_test_case(
        prompt="delegate tasks",
        test_case={
            "input": "How does auth work?",
            "expected_agent": case["expected_agent"],
        },
    )

    assert grade["score"] == case["expected_score"]
    mock_client.chat_json.assert_not_called()
