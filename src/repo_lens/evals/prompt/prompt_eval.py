import json
import logging
import sys
from typing import Any

from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ClientError as GeminiClientError
from pydantic import TypeAdapter, ValidationError

from repo_lens.agents.agent import WIRED_AGENTS
from repo_lens.core.config import create_config
from repo_lens.core.retry import wait_for_retry_sync
from repo_lens.evals.prompt.models import GradeResult, TestCase
from repo_lens.evals.prompt.prompt_eval_dataset import PROMPT_EVAL_CASES
from repo_lens.evals.prompt.prompts import EVAL_TOOLS, PROMPTS
from repo_lens.evals.prompt.types import GradeOutput, PromptTestCase, TestCaseResult
from repo_lens.evals.structured_output import parse_with_retry
from repo_lens.providers.chat_client import ChatClientProtocol, ChatResponse, ToolCall
from repo_lens.providers.provider import create_chat_client

logger = logging.getLogger(__name__)

adapter = TypeAdapter(list[TestCase])


def _warn_missing_agent_test_coverage(prompt_name: str) -> None:
    """Log when a wired agent has no test case criteria mentioning it."""
    cases = PROMPT_EVAL_CASES.get(prompt_name, [])
    for name in WIRED_AGENTS:
        needle = f"{name.value} agent"
        covered = any(
            needle in str(criterion).lower()
            for case in cases
            for criteria in [case.get("criteria", [])]
            if isinstance(criteria, list)
            for criterion in criteria
        )
        if not covered:
            logger.warning(
                "No %s test cases mention the %s agent in criteria",
                prompt_name,
                name.value,
            )


def _describe_tool_call(tool_call: ToolCall | Any) -> str:
    tool_input = getattr(tool_call, "input", None)
    if not isinstance(tool_input, dict):
        return str(tool_call)

    agent_name = tool_input.get("agent_name", "?")
    task = tool_input.get("task", "")
    return f"I will delegate this task to the {agent_name} agent: {task}"


class PromptEvaluator:
    def __init__(
        self,
        *,
        client: ChatClientProtocol,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.tools = tools

    def _call_with_retry(self, messages: list[Any], *, json_mode: bool) -> ChatResponse:
        retries = 0
        while True:
            try:
                if json_mode:
                    return self.client.chat_json(messages)
                return self.client.chat(messages=messages, tools=self.tools)
            except AnthropicRateLimitError:
                if wait_for_retry_sync(retries=retries):
                    raise
                retries += 1
            except GeminiClientError as e:
                if e.code != 429:
                    raise
                if wait_for_retry_sync(
                    retries=retries, detail=getattr(e, "message", str(e))
                ):
                    raise
                retries += 1

    def generate_dataset(
        self, prompt: str, total_tests: int = 3, max_tries: int = 2
    ) -> list[PromptTestCase]:
        dataset_prompt = f"""
        I have this prompt that I want to evaluate:
        "{prompt}"

        Generate {total_tests} test cases. Each should have:
        - "input": a realistic input that this prompt would receive
        - "criteria": what a good response should include

        Return a JSON array with this structure:
        [
            {{
                "input": "...",
                "criteria": ["...", "..."]
            }}
        ]
        """
        messages: list[Any] = []
        self.client.add_user_message(messages=messages, content=dataset_prompt)
        response = self._call_with_retry(messages, json_mode=True)
        text = response.text

        try:
            result = adapter.validate_json(text)
            return [
                PromptTestCase(input=case.input, criteria=case.criteria)
                for case in result
            ]
        except ValidationError:
            if max_tries > 0:
                logger.warning("Failed to parse generated dataset. Retrying")
                return self.generate_dataset(prompt, total_tests, max_tries - 1)
            logger.error("Failed to generate valid dataset after retries.")
            return []

    def run_prompt(self, *, prompt: str, test_case: PromptTestCase) -> ChatResponse:
        prompt = prompt.strip()
        separator = "" if prompt and prompt[-1] in ".?!:" else ":"
        message = f"""
        {prompt}{separator}
        <input>
        {test_case["input"]}
        </input>
        """
        messages: list[Any] = []
        self.client.add_user_message(messages=messages, content=message)
        response = self._call_with_retry(messages, json_mode=False)

        return response

    def grade_output(self, test_case: PromptTestCase, output: str) -> GradeOutput:
        eval_prompt = f"""
        Evaluate this response.

        Input: {test_case["input"]}
        Response: {output}
        Criteria: {", ".join(test_case["criteria"])}

        Return JSON only: {{"strengths": [], "weaknesses": [], "reasoning": "", "score": "<integer 0-10>"}}
        """
        messages: list[Any] = []
        self.client.add_user_message(messages=messages, content=eval_prompt)
        response = self._call_with_retry(messages, json_mode=True)

        try:
            result = parse_with_retry(
                chat_client=self.client,
                response=response,
                messages=messages,
                model_type=GradeResult,
            )
            return GradeOutput(
                strengths=result.strengths,
                weaknesses=result.weaknesses,
                reasoning=result.reasoning,
                score=result.score,
            )
        except ValueError:
            return GradeOutput(
                strengths=[],
                weaknesses=[],
                reasoning="Failed to parse grade",
                score=0,
            )

    def run_test_case(
        self, *, prompt: str, test_case: PromptTestCase
    ) -> TestCaseResult:
        output = self.run_prompt(prompt=prompt, test_case=test_case)
        called_agent: str | None = None

        if output.tool_calls:
            called_agent = output.tool_calls[0].input.get("agent_name")

        expected_agent = test_case.get("expected_agent")
        result = TestCaseResult(
            input=test_case["input"],
            expected=expected_agent,
            actual=called_agent,
            score=10 if called_agent == expected_agent else 0,
        )

        print(json.dumps(result, indent=2))
        return result

    def run_eval(self, prompt: str, test_cases: list[PromptTestCase]) -> None:
        if len(test_cases) == 0:
            return

        passed = 0
        for test_case in test_cases:
            grade = self.run_test_case(prompt=prompt, test_case=test_case)
            score = int(grade["score"])
            if score == 10:
                passed += 1

        total = len(test_cases)
        pass_rate = round(passed / total, 2)
        stats = {
            "passed": passed,
            "total": total,
            "pass_rate": pass_rate,
        }

        print(json.dumps(stats, indent=2))
        print(f"Pass rate: {pass_rate * 100:.1f}%")


def main() -> None:
    config = create_config()
    client = create_chat_client(config)

    if len(sys.argv) < 2:
        # No prompt given: evaluate repo-lens's own prompts against their
        # curated test cases instead of generating a dataset on the fly.
        for name, prompt in PROMPTS.items():
            _warn_missing_agent_test_coverage(name)
            print(f"\n=== {name} ===")
            evaluator = PromptEvaluator(client=client, tools=EVAL_TOOLS.get(name))
            evaluator.run_eval(prompt, PROMPT_EVAL_CASES.get(name, []))
        return

    evaluator = PromptEvaluator(client=client)
    prompt = sys.argv[1]
    test_cases = evaluator.generate_dataset(prompt)
    evaluator.run_eval(prompt, test_cases)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
