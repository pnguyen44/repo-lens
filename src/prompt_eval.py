import json
import sys
from typing import Any

from anthropic import Anthropic
from chat_client import ChatClient
from claude import Claude
from config import create_config


class PromptEvaluator:
    def __init__(self, client: ChatClient[Any, Any]) -> None:
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
        self.client.add_assistant_message(messages, "```json")
        response = self.client.chat(messages, stop_sequences=["```"])
        text = self.client.text_from_message(response)

        try:
            result: list[dict[str, Any]] = json.loads(text)
        except json.JSONDecodeError:
            if max_tries > 0:
                print("Failed to parse generated dataset. Retrying")
                return self.generate_dataset(prompt, total_tests, max_tries - 1)
            print("Failed to generate valid dataset after retries.")
            return []

        return result

    def run_prompt(self, prompt: str, test_case: dict[str, Any]) -> str:
        prompt = prompt.strip()
        separator = "" if prompt[-1] in ".?!:" else ":"
        message = f"""
        {prompt}{separator}
        <input>
        {test_case["input"]}
        </input>
        """
        messages: list[Any] = []
        self.client.add_user_message(messages=messages, content=message)
        response = self.client.chat(messages=messages)
        return self.client.text_from_message(response)

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
        self.client.add_assistant_message(messages, "```json")
        response = self.client.chat(messages=messages, stop_sequences=["```"])
        text = self.client.text_from_message(response)

        fallback = {
            "strengths": [],
            "weaknesses": [],
            "reasoning": "Failed to parse grade",
            "score": 0,
        }
        try:
            result: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            return fallback

        if not isinstance(result, dict):
            return fallback

        required_keys = {"score", "strengths", "weaknesses", "reasoning"}

        if not required_keys.issubset(result.keys()):
            return fallback

        try:
            result["score"] = int(result["score"])
        except (TypeError, ValueError):
            return fallback

        return result

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
    client = Anthropic()
    claude = Claude(client=client, model=config.claude_model)

    evaluator = PromptEvaluator(claude)

    prompt = sys.argv[1]
    test_cases = evaluator.generate_dataset(prompt)
    evaluator.run_eval(prompt, test_cases)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
