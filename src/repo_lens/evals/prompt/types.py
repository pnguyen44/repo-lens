from typing import NotRequired, TypedDict


class PromptTestCase(TypedDict):
    input: str
    criteria: NotRequired[list[str]]
    expected_agent: NotRequired[str | None]


class TestCaseResult(TypedDict):
    input: str
    expected: str | None
    actual: str | None
    score: int


class GradeOutput(TypedDict):
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int


class EvalStats(TypedDict):
    passed: int
    total: int
    pass_rate: float
