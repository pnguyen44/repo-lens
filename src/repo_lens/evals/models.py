from typing import Literal

from pydantic import BaseModel, Field


class FaithfulnessVerdict(BaseModel):
    verdict: Literal["grounded", "partial", "hallucinated"]
    reasoning: str


class GradeResult(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int = Field(ge=0, le=10)


class TestCase(BaseModel):
    input: str
    criteria: list[str]
