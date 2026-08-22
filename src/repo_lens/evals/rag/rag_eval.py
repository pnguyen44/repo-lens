import asyncio
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from voyageai.client_async import AsyncClient as VoyageAsyncClient

from repo_lens.agents.chat import MAX_RETRIES
from repo_lens.core.config import create_config
from repo_lens.evals.rag.models import FaithfulnessVerdict
from repo_lens.evals.rag.rag_eval_dataset import EVAL_CASES
from repo_lens.evals.rag.types import (
    EvalCase,
    EvalResult,
    FaithfulnessEvalResult,
    FaithfulnessJudgement,
    RetrievalResult,
    SweepResult,
)
from repo_lens.evals.structured_output import parse_with_retry
from repo_lens.evals.utils import append_jsonl, print_table
from repo_lens.providers.chat_client import ChatClientProtocol
from repo_lens.providers.provider import create_chat_client
from repo_lens.rag.chunker import chunk_by_section
from repo_lens.rag.embeddings import Embedder, InputType, VoyageEmbedder
from repo_lens.rag.vector_index import VectorIndex

logger = logging.getLogger(__name__)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[4] / "tests/fixtures/hyperfleet_api_readme.md"
)

RAG_EVAL_RESULT_PATH = Path(__file__).resolve().parents[4] / "evals/rag_results.jsonl"


class RAGEvaluator:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        eval_cases: list[EvalCase],
        fixture_path: Path,
        chat_client: ChatClientProtocol | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.eval_cases = eval_cases
        self.fixture_path = fixture_path
        self.chat_client = chat_client

    async def load_and_index(self) -> int:
        """Read the fixture README, chunk it, embed, and load into the index."""
        readme_text = self.fixture_path.read_text()
        chunks = [c for c in chunk_by_section(readme_text) if c.strip()]

        if not chunks:
            return 0

        vectors = await self.embedder.generate_embeddings(
            texts=chunks, input_type=InputType.DOCUMENT
        )

        for vector, chunk in zip(vectors, chunks):
            section = self.identify_section(chunk)
            self.index.add_vector(
                vector=vector, document={"content": chunk, "section": section}
            )
        return len(self.index.vectors)

    def identify_section(self, chunk_text: str) -> str:
        """Extract the ## section name from a chunk's text."""
        return chunk_text.split("\n", 1)[0].lstrip("# ").strip()

    def generate_answer(self, question: str, context: str) -> str:
        if self.chat_client is None:
            raise ValueError("chat_client required for answer generation")

        message = textwrap.dedent(f"""\
        {question}
        <context>
        {context}
        </context>
        """)

        messages: list[Any] = []
        self.chat_client.add_user_message(messages=messages, content=message)
        response = self.chat_client.chat(messages=messages)
        return response.text

    def judge_faithfulness(
        self, context: str, question: str, answer: str
    ) -> FaithfulnessJudgement:
        if self.chat_client is None:
            raise ValueError("chat_client required for judge faithfulness")
        eval_prompt = textwrap.dedent(f"""\
        Context: {context}
        Question: {question}
        Answer: {answer}
        Is the answer fully supported by the context?
        Return JSON only: {{"verdict": "grounded | partial | hallucinated", "reasoning": ""}}
        """)

        messages: list[Any] = []
        self.chat_client.add_user_message(messages=messages, content=eval_prompt)
        response = self.chat_client.chat_json(messages=messages)

        try:
            result = parse_with_retry(
                chat_client=self.chat_client,
                response=response,
                messages=messages,
                model_type=FaithfulnessVerdict,
                max_retries=MAX_RETRIES,
            )
            return cast(FaithfulnessJudgement, result.model_dump())
        except ValueError:
            return FaithfulnessJudgement(
                verdict="unknown",
                reasoning="Failed to parse grade",
            )

    async def evaluate_faithfulness(self, k: int = 3) -> list[FaithfulnessEvalResult]:
        if not self.chat_client:
            return []
        results: list[FaithfulnessEvalResult] = []
        for case in self.eval_cases:
            question = str(case["question"])
            try:
                query_vector = (
                    await self.embedder.generate_embeddings(
                        [question], input_type=InputType.QUERY
                    )
                )[0]
                hits = await self.index.search(query=query_vector, k=k)
                context = "\n".join([doc["content"] for doc, _dist in hits])
                answer = self.generate_answer(context=context, question=question)
                judgement = self.judge_faithfulness(
                    context=context, question=question, answer=answer
                )
                results.append({"question": question, "judgement": judgement})
            except Exception as exc:
                logger.warning("Faithfulness eval failed for '%s': %s", question, exc)
                results.append(
                    {"question": question, "judgement": None, "error": str(exc)}
                )
        return results

    def _precision_recall(
        self, retrieved: set[str], expected: set[str]
    ) -> tuple[float, float]:
        true_positions = retrieved & expected
        precision = len(true_positions) / len(retrieved) if retrieved else 0.0
        recall = len(true_positions) / len(expected) if expected else 0.0
        return precision, recall

    @property
    def _questions(self) -> list[str]:
        return [str(case["question"]) for case in self.eval_cases]

    async def evaluate_retrieval(
        self, k: int = 3, query_vectors: list[list[float]] | None = None
    ) -> list[RetrievalResult]:
        """Run each eval case: embed the question, search, compute precision and recall."""

        if query_vectors is None:
            query_vectors = await self.embedder.generate_embeddings(
                texts=self._questions, input_type=InputType.QUERY
            )

        results: list[RetrievalResult] = []

        for case, query_vector in zip(self.eval_cases, query_vectors):
            hits = await self.index.search(query=query_vector, k=k)
            result: RetrievalResult = {"question": case["question"]}

            if "expected_sections" in case:
                retrieved_sections = {
                    self.identify_section(doc["content"]) for doc, _dist in hits
                }
                expected_sections = set(case["expected_sections"])
                result["section_precision"], result["section_recall"] = (
                    self._precision_recall(
                        retrieved=retrieved_sections, expected=expected_sections
                    )
                )

            if "expected_keywords" in case:
                content = " ".join([doc["content"] for doc, _dist in hits]).lower()
                expected_kw = {str(kw).lower() for kw in case["expected_keywords"]}
                found = {kw for kw in expected_kw if kw in content}
                _, result["keyword_recall"] = self._precision_recall(
                    retrieved=found, expected=expected_kw
                )

            results.append(result)
        return results

    async def sweep_k(self, k_range: range = range(1, 11)) -> list[SweepResult]:
        """Run evaluate_retrieval for each k, return aggregate metrics per k."""
        query_vectors = await self.embedder.generate_embeddings(
            texts=self._questions, input_type=InputType.QUERY
        )
        sweep_results: list[SweepResult] = []
        for k in k_range:
            results = await self.evaluate_retrieval(k=k, query_vectors=query_vectors)

            precisions = [
                r["section_precision"] for r in results if "section_precision" in r
            ]
            recalls = [r["section_recall"] for r in results if "section_recall" in r]

            avg_precision = round(
                sum(precisions) / len(precisions) if precisions else 0.0, 2
            )
            avg_recall = round(sum(recalls) / len(recalls) if recalls else 0.0, 2)

            # F1 balances precision and recall into one score, so sweeping k can be
            # judged by a single number instead of comparing two curves.
            f1 = round(
                2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                if (avg_precision + avg_recall) > 0
                else 0.0,
                2,
            )

            sweep_results.append(
                SweepResult(
                    k=k,
                    avg_precision=avg_precision,
                    avg_recall=avg_recall,
                    f1=f1,
                )
            )
        return sweep_results

    def print_results(self, results: list[EvalResult]) -> None:
        """Print per-question scores and aggregate precision/recall."""
        for r in results:
            print(f"\nQ: {r['question']}")
            if "section_precision" in r:
                print(
                    f"  Section Precision: {r['section_precision']:.2f} | Recall: {r['section_recall']:.2f}"
                )
            if "keyword_recall" in r:
                print(f"  Keyword Recall: {r['keyword_recall']:.2f}")

            judgement = r.get("judgement")
            if judgement:
                print(f"  Faithfulness: {judgement['verdict']}")

        print("\n---")
        verdicts = [
            j["verdict"] for r in results if (j := r.get("judgement")) is not None
        ]
        if verdicts:
            grounded = verdicts.count("grounded")
            print(f"Faithfulness: {grounded}/{len(verdicts)} grounded")

        section_precisions = [
            r["section_precision"] for r in results if "section_precision" in r
        ]
        section_recalls = [
            r["section_recall"] for r in results if "section_recall" in r
        ]
        keyword_recalls = [
            r["keyword_recall"] for r in results if "keyword_recall" in r
        ]

        if section_precisions:
            print(
                f"Avg Section Precision: {sum(section_precisions) / len(section_precisions):.2f}"
            )
            print(
                f"Avg Section Recall:    {sum(section_recalls) / len(section_recalls):.2f}"
            )
        if keyword_recalls:
            print(
                f"Avg Keyword Recall:    {sum(keyword_recalls) / len(keyword_recalls):.2f}"
            )


async def main(sweep: bool = False) -> None:
    load_dotenv()
    config = create_config()
    try:
        index = VectorIndex()
        embedder = VoyageEmbedder(VoyageAsyncClient(), model=config.voyage_embed_model)
        chat_client = None if sweep else create_chat_client(config=config)

        rag_evaluator = RAGEvaluator(
            index=index,
            embedder=embedder,
            fixture_path=FIXTURE_PATH,
            eval_cases=EVAL_CASES,
            chat_client=chat_client,
        )
        await rag_evaluator.load_and_index()

        if sweep:
            sweep_results = await rag_evaluator.sweep_k()
            print_table([dict(r) for r in sweep_results])
            append_jsonl(
                results=[dict(r) for r in sweep_results], path=RAG_EVAL_RESULT_PATH
            )
            return

        retrieval_results = await rag_evaluator.evaluate_retrieval()
        faithfulness_results = await rag_evaluator.evaluate_faithfulness()
        combined: list[EvalResult] = [
            {**r, "judgement": f["judgement"]}
            for r, f in zip(retrieval_results, faithfulness_results)
        ]
        rag_evaluator.print_results(combined)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sweep = "--sweep" in sys.argv
    asyncio.run(main(sweep=sweep))
