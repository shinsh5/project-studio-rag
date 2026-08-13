"""Evaluate ROI-RAG outputs with RAGAS's standard Faithfulness metric, judged by Gemini."""

import asyncio
import inspect
import json
import math
import os
import time
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

import config
import llm_client
from roi_rag import get_roi_rag_pipeline


class RagasStructuredOutputError(RuntimeError):
    """Raised when Gemini cannot produce a valid faithfulness evaluation."""


class StatementVerdict(BaseModel):
    """One RAGAS statement joined with its faithfulness verdict."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    statement: str
    reason: str
    verdict: int


class FaithfulnessEvaluation(BaseModel):
    value: float
    supported_claims: int
    total_claims: int
    claims: list[StatementVerdict]
    contexts_evaluated: int


ProgressCallback = Callable[
    [dict[str, Any]],
    Awaitable[None] | None,
]


async def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    progress: int,
    message: str,
) -> None:
    if callback is None:
        return
    result = callback(
        {
            "stage": stage,
            "progress": progress,
            "message": message,
        }
    )
    if inspect.isawaitable(result):
        await result


def _build_faithfulness_metric() -> Faithfulness:
    return Faithfulness(llm=llm_client.get_ragas_llm())


def _build_response_relevancy_metric() -> ResponseRelevancy:
    """
    Answer relevancy embeds with the local model, so only question generation
    costs API calls. Strictness controls that ensemble size -- see
    config.RAGAS_ANSWER_RELEVANCY_STRICTNESS for the measured variance.
    """
    return ResponseRelevancy(
        llm=llm_client.get_ragas_llm(),
        embeddings=llm_client.get_ragas_embeddings(),
        strictness=config.RAGAS_ANSWER_RELEVANCY_STRICTNESS,
    )


def _build_context_recall_metric() -> LLMContextRecall:
    return LLMContextRecall(llm=llm_client.get_ragas_llm())


async def _score_single_turn(metric, sample: SingleTurnSample, label: str) -> float:
    """Run one RAGAS single-turn metric with the shared timeout and error wrapping."""
    try:
        score = await asyncio.wait_for(
            metric.single_turn_ascore(sample),
            timeout=config.GEMINI_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise
    except (RuntimeError, ValidationError, ValueError) as exc:
        raise RagasStructuredOutputError(
            f"Gemini returned invalid {label} output: {exc}"
        ) from exc

    score = float(score)
    if not math.isfinite(score):
        raise RagasStructuredOutputError(
            f"RAGAS returned a non-finite {label} score."
        )
    return score


async def score_response_relevancy(question: str, answer: str) -> float:
    """
    How well the answer addresses the question. Needs no ground truth, so this
    also runs for ad-hoc GUI questions. RAGAS scores non-committal answers
    ("the information is not available") near zero, which makes over-refusal
    visible -- faithfulness alone cannot distinguish that from hallucination.
    """
    if not question.strip():
        raise ValueError("question is missing.")
    if not answer.strip():
        raise ValueError("answer is missing.")

    sample = SingleTurnSample(user_input=question, response=answer)
    return await _score_single_turn(
        _build_response_relevancy_metric(), sample, "answer relevancy"
    )


async def score_context_recall(
    question: str, contexts: list[str], reference: str
) -> float:
    """
    How much of the reference answer is actually covered by the retrieved
    contexts. This is the metric Small-to-Big is expected to improve, since it
    measures retrieval coverage rather than how faithfully the answer copies
    whatever was retrieved. Requires a ground truth, so it only runs in batch
    evaluation over MULTINEWS_QA.
    """
    clean_contexts = [c for c in contexts if c.strip()]
    if not clean_contexts:
        raise ValueError("contexts are missing.")
    if not reference.strip():
        raise ValueError("reference (ground truth) is missing.")

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=clean_contexts,
        reference=reference,
    )
    return await _score_single_turn(
        _build_context_recall_metric(), sample, "context recall"
    )


async def score_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    progress_callback: ProgressCallback | None = None,
) -> FaithfulnessEvaluation:
    """Score faithfulness using RAGAS's own statement generation and NLI judging."""
    await _emit_progress(
        progress_callback,
        "validating_input",
        5,
        "질문·답변 입력을 확인하는 중",
    )
    if not question.strip():
        raise ValueError("question is missing.")
    if not answer.strip():
        raise ValueError("answer is missing.")

    clean_contexts = [context for context in contexts if context.strip()]
    if not clean_contexts:
        raise ValueError("contexts are missing.")

    row = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": clean_contexts,
    }

    metric = _build_faithfulness_metric()

    await _emit_progress(
        progress_callback,
        "generating_statements",
        25,
        "RAGAS가 답변을 atomic statement로 분해하는 중",
    )
    try:
        statements_output = await asyncio.wait_for(
            metric._create_statements(row, callbacks=None),
            timeout=config.GEMINI_TIMEOUT_SECONDS,
        )
        statements = statements_output.statements
    except TimeoutError:
        raise
    except (RuntimeError, ValidationError, ValueError) as exc:
        raise RagasStructuredOutputError(
            f"Gemini returned invalid statement generation output: {exc}"
        ) from exc

    if not statements:
        raise RagasStructuredOutputError(
            "RAGAS produced no statements from the answer."
        )

    await _emit_progress(
        progress_callback,
        "judging_statements",
        60,
        f"Statements: {len(statements)}; contexts: {len(clean_contexts)}",
    )
    try:
        verdicts_output = await asyncio.wait_for(
            metric._create_verdicts(row, statements, callbacks=None),
            timeout=config.GEMINI_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise
    except (RuntimeError, ValidationError, ValueError) as exc:
        raise RagasStructuredOutputError(
            f"Gemini returned invalid structured faithfulness output: {exc}"
        ) from exc

    await _emit_progress(
        progress_callback,
        "calculating_score",
        95,
        f"statement {len(verdicts_output.statements)}개의 최종 점수를 계산하는 중",
    )

    scored_claims = [
        StatementVerdict(
            claim_id=f"c{index + 1}",
            statement=item.statement,
            reason=item.reason,
            verdict=item.verdict,
        )
        for index, item in enumerate(verdicts_output.statements)
    ]

    supported = sum(claim.verdict for claim in scored_claims)
    total = len(scored_claims)
    if total == 0:
        raise RagasStructuredOutputError("RAGAS produced no verdicts.")

    result = FaithfulnessEvaluation(
        value=supported / total,
        supported_claims=supported,
        total_claims=total,
        claims=scored_claims,
        contexts_evaluated=len(clean_contexts),
    )
    await _emit_progress(
        progress_callback,
        "completed",
        100,
        "RAGAS faithfulness 평가 완료",
    )
    return result


async def evaluate_faithfulness(results: list[dict]) -> list[dict]:
    """
    Evaluate generated answers sequentially with fresh Gemini judge calls.

    Scores faithfulness and answer relevancy for every sample, plus context
    recall when the sample carries a ground truth. A failure in one metric is
    recorded and does not abort the remaining metrics or samples -- a long
    batch run should not be lost to a single malformed judge response.
    """
    evaluated = []
    for item in results:
        started_at = time.perf_counter()
        row = {"question": item["question"]}

        metric_result = await score_faithfulness(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
        )
        score = float(metric_result.value)
        if not math.isfinite(score):
            raise ValueError("RAGAS returned a non-finite faithfulness score.")
        row["faithfulness"] = score
        row["supported_claims"] = metric_result.supported_claims
        row["total_claims"] = metric_result.total_claims

        try:
            row["answer_relevancy"] = await score_response_relevancy(
                question=item["question"],
                answer=item["answer"],
            )
        except Exception as exc:
            row["answer_relevancy"] = None
            row["answer_relevancy_error"] = f"{type(exc).__name__}: {exc}"

        reference = (item.get("ground_truth") or "").strip()
        if reference:
            try:
                row["context_recall"] = await score_context_recall(
                    question=item["question"],
                    contexts=item["contexts"],
                    reference=reference,
                )
            except Exception as exc:
                row["context_recall"] = None
                row["context_recall_error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["context_recall"] = None

        row["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        evaluated.append(row)
    return evaluated


def main():
    dataset_path = "eval_dataset.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as stream:
            test_samples = json.load(stream)
    else:
        test_samples = [
            {
                "question": "What is the primary function of ROI-RAG?",
                "ground_truth": (
                    "ROI-RAG evaluates redundancy and diversity to construct "
                    "evidence units for grounded generation."
                ),
            }
        ]

    pipeline = get_roi_rag_pipeline()
    results = []
    for sample in test_samples:
        rag_output = pipeline(sample["question"])
        results.append(
            {
                "question": sample["question"],
                "answer": rag_output["answer"],
                "contexts": rag_output["retrieved_contexts"],
                "ground_truth": sample["ground_truth"],
            }
        )

    evaluated_results = asyncio.run(evaluate_faithfulness(results))
    print(json.dumps(evaluated_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
