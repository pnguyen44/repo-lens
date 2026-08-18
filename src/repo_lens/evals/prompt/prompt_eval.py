import json
import logging
import sys
from typing import Any

from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ClientError as GeminiClientError
from pydantic import TypeAdapter, ValidationError

from repo_lens.core.config import create_config
from repo_lens.core.retry import wait_for_retry_sync
from repo_lens.evals.prompt.models import GradeResult, TestCase
from repo_lens.evals.prompt.prompt_eval_dataset import PROMPT_EVAL_CASES
from repo_lens.evals.prompt.prompts import PROMPTS
from repo_lens.evals.structured_output import parse_with_retry
from repo_lens.providers.chat_client import ChatClientProtocol, ChatResponse
from repo_lens.providers.provider import create_chat_client

logger = logging.getLogger(__name__)

adapter = TypeAdapter(list[TestCase])


class PromptEvaluator:
    def __init__(self, client: ChatClientProtocol) -> None:
        self.client = client

    def _call_with_retry(self, messages: list[Any], *, json_mode: bool) -> ChatResponse:
        retries = 0
        while True:
            try:
                if json_mode:
                    return self.client.chat_json(messages)
                return self.client.chat(messages=messages)
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
    ) -> list[dict[str, Any]]:
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
            return [case.model_dump() for case in result]
        except ValidationError:
            if max_tries > 0:
                logger.warning("Failed to parse generated dataset. Retrying")
                return self.generate_dataset(prompt, total_tests, max_tries - 1)
            logger.error("Failed to generate valid dataset after retries.")
            return []

    def run_prompt(self, prompt: str, test_case: dict[str, Any]) -> str:
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
        return response.text

    def grade_output(self, test_case: dict[str, Any], output: str) -> dict[str, Any]:
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
            return parse_with_retry(
                chat_client=self.client,
                response=response,
                messages=messages,
                model_type=GradeResult,
            ).model_dump()
        except ValueError:
            return {
                "strengths": [],
                "weaknesses": [],
                "reasoning": "Failed to parse grade",
                "score": 0,
            }

    def run_test_case(self, prompt: str, test_case: dict[str, Any]) -> dict[str, Any]:
        output = self.run_prompt(prompt, test_case)
        grade = self.grade_output(test_case, output)
        print(json.dumps(grade, indent=2))
        return grade

    def run_eval(self, prompt: str, test_cases: list[dict[str, Any]]) -> None:
        if len(test_cases) == 0:
            return

        total = 0
        for test_case in test_cases:
            grade = self.run_test_case(prompt, test_case)
            score = int(grade["score"])
            total += score

        stats = {"average": round(total / len(test_cases), 2)}

        print(json.dumps(stats, indent=2))


def main() -> None:
    config = create_config()
    client = create_chat_client(config)

    evaluator = PromptEvaluator(client)

    if len(sys.argv) < 2:
        # No prompt given: evaluate repo-lens's own prompts against their
        # curated test cases instead of generating a dataset on the fly.
        for name, prompt in PROMPTS.items():
            print(f"\n=== {name} ===")
            evaluator.run_eval(
                prompt=prompt, test_cases=PROMPT_EVAL_CASES.get(name, [])
            )
        return

    prompt = sys.argv[1]
    test_cases = evaluator.generate_dataset(prompt)
    evaluator.run_eval(prompt, test_cases)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
