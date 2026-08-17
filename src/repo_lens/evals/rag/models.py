from typing import Literal

from pydantic import BaseModel


class FaithfulnessVerdict(BaseModel):
    verdict: Literal["grounded", "partial", "hallucinated"]
    reasoning: str
