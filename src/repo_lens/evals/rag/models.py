from pydantic import BaseModel

from repo_lens.evals.rag.types import Verdict


class FaithfulnessVerdict(BaseModel):
    verdict: Verdict
    reasoning: str
