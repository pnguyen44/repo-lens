from typing import Literal, NotRequired, TypedDict

Verdict = Literal["grounded", "partial", "hallucinated"]


class EvalCase(TypedDict, total=False):
    question: str
    expected_sections: list[str]
    expected_keywords: list[str]


class FaithfulnessJudgement(TypedDict):
    verdict: Verdict | Literal["unknown"]
    reasoning: str


class RetrievalResult(TypedDict, total=False):
    question: str
    section_precision: float
    section_recall: float
    keyword_recall: float


class FaithfulnessEvalResult(TypedDict):
    question: str
    judgement: FaithfulnessJudgement | None
    error: NotRequired[str]


class EvalResult(TypedDict, total=False):
    question: str
    section_precision: float
    section_recall: float
    keyword_recall: float
    judgement: FaithfulnessJudgement | None
