from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int = Field(ge=0, le=10)


class TestCase(BaseModel):
    input: str
    criteria: list[str]
