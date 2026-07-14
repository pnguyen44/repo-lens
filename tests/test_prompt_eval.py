import json
from unittest.mock import MagicMock

from chat_client import ChatResponse
from prompt_eval import PromptEvaluator


def test_generate_dataset_returns_empty_after_retries() -> None:
    mock_client = MagicMock()
    mock_client.chat_json.return_value = ChatResponse(text="not valid json")

    evaluator = PromptEvaluator(mock_client)
    result = evaluator.generate_dataset("test prompt")

    assert result == []
    assert mock_client.chat_json.call_count == 3


def test_generate_dataset_parses_valid_json() -> None:
    mock_client = MagicMock()
    mock_client.chat_json.return_value = ChatResponse(
        text=json.dumps([{"input": "hello", "criteria": ["is polite"]}])
    )

    evaluator = PromptEvaluator(mock_client)
    result = evaluator.generate_dataset("test prompt", total_tests=1)

    assert len(result) == 1
    assert result[0]["input"] == "hello"


def test_grade_output_returns_fallback_on_bad_json() -> None:
    mock_client = MagicMock()
    mock_client.chat_json.return_value = ChatResponse(text="not json")

    evaluator = PromptEvaluator(mock_client)
    result = evaluator.grade_output(
        {"input": "hi", "criteria": ["is polite"]}, "hello there"
    )

    assert result["score"] == 0
    assert result["reasoning"] == "Failed to parse grade"


def test_run_eval_does_nothing_on_empty_test_cases() -> None:
    mock_client = MagicMock()
    evaluator = PromptEvaluator(mock_client)

    evaluator.run_eval("test prompt", [])

    mock_client.chat.assert_not_called()
