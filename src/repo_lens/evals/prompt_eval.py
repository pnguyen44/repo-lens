import json
import logging
import sys
from typing import Any
from pydantic import TypeAdapter, ValidationError
from repo_lens.providers.chat_client import ChatClient
from repo_lens.core.config import create_config
from repo_lens.evals.models import GradeResult, TestCase
from repo_lens.providers.provider import create_chat_client
from repo_lens.evals.structured_output import parse_with_retry

logger = logging.getLogger(__name__)

adapter = TypeAdapter(list[TestCase])


class PromptEvaluator:
    def __init__(self, client: ChatClient[Any]) -> None:
        self.client = client

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
        response = self.client.chat_json(messages)
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
        response = self.client.chat(messages=messages)
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
        response = self.client.chat_json(messages)

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
    if len(sys.argv) < 2:
        print('Usage: uv run src/prompt_eval.py "Your prompt here"')
        sys.exit(1)

    config = create_config()
    client = create_chat_client(config)

    evaluator = PromptEvaluator(client)

    prompt = sys.argv[1]
    test_cases = evaluator.generate_dataset(prompt)
    evaluator.run_eval(prompt, test_cases)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
