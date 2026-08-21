from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from repo_lens.evals.rag.rag_eval import RAGEvaluator


def _make_evaluator() -> RAGEvaluator:
    return RAGEvaluator(
        embedder=MagicMock(),
        index=MagicMock(),
        eval_cases=[{"question": "q1"}],
        fixture_path=Path("unused.md"),
    )


SWEEP_K_METRICS_CASES = [
    {
        "name": "mixed precision and recall",
        "retrieval_results": [
            {"question": "q1", "section_precision": 0.5, "section_recall": 1.0},
            {"question": "q2", "section_precision": 1.0, "section_recall": 0.5},
        ],
        "expected": {"avg_precision": 0.75, "avg_recall": 0.75, "f1": 0.75},
    },
    {
        "name": "no metrics present",
        "retrieval_results": [{"question": "q1"}],
        "expected": {"avg_precision": 0.0, "avg_recall": 0.0, "f1": 0.0},
    },
]


@pytest.mark.parametrize("case", SWEEP_K_METRICS_CASES, ids=lambda c: c["name"])
async def test_sweep_k_computes_metrics(case) -> None:
    evaluator = _make_evaluator()
    evaluator.evaluate_retrieval = AsyncMock(  # type: ignore[method-assign]
        return_value=case["retrieval_results"]
    )

    results = await evaluator.sweep_k(k_range=range(3, 4))

    assert results == [{"k": 3, **case["expected"]}]
    evaluator.evaluate_retrieval.assert_awaited_once_with(3)


async def test_sweep_k_runs_once_per_k_in_range() -> None:
    evaluator = _make_evaluator()
    evaluator.evaluate_retrieval = AsyncMock(  # type: ignore[method-assign]
        return_value=[]
    )

    results = await evaluator.sweep_k(k_range=range(1, 4))

    assert [r["k"] for r in results] == [1, 2, 3]
    assert evaluator.evaluate_retrieval.await_count == 3
