"""Evaluate ROI-RAG outputs with RAGAS faithfulness judged by Codex exec."""

import asyncio
import inspect
import json
import math
import os
import re
import time
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import config
import llm_client
from roi_rag import get_roi_rag_pipeline


class RagasStructuredOutputError(RuntimeError):
    """Raised when Codex cannot produce a valid faithfulness evaluation."""


class StrictStatementVerdict(BaseModel):
    """One answer-derived atomic claim and its context-grounding verdict."""

    model_config = ConfigDict(extra="forbid")

    source_text: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    reason: str = Field(min_length=3)
    evidence: str = Field(min_length=1)
    verdict: Literal[0, 1]


class CodexFaithfulnessOutput(BaseModel):
    """Schema-constrained output returned by one Codex exec invocation."""

    model_config = ConfigDict(extra="forbid")

    claims: list[StrictStatementVerdict] = Field(min_length=1)


class FaithfulnessEvaluation(BaseModel):
    value: float
    supported_claims: int
    total_claims: int
    claims: list[StrictStatementVerdict]
    contexts_evaluated: int


_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    """Normalize numeric literals so Codex cannot silently alter answer values."""
    return {match.replace(",", "") for match in _NUMBER_RE.findall(text)}


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).casefold()


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


def _build_faithfulness_prompt(
    question: str,
    answer: str,
    contexts: list[str],
) -> str:
    payload = {
        "question": question,
        "answer": answer,
        "contexts": [
            {"context_id": index + 1, "text": context}
            for index, context in enumerate(contexts)
        ],
    }
    return f"""Evaluate RAGAS faithfulness for the supplied answer.

Perform both stages in this single response:

1. Extract every atomic factual claim explicitly asserted by the answer.
2. Judge each extracted claim using only the supplied contexts.

Claim extraction rules:
- Do not use a fixed number of claims. The number must follow the answer's actual content.
- Do not invent, correct, broaden, or omit factual claims.
- Preserve every name, entity, location, date, number, negation, qualifier, and uncertainty.
- Split a sentence only when it contains multiple independently verifiable factual claims.
- source_text should identify the shortest answer passage that most directly expresses
  the claim. Minor quotation, whitespace, and punctuation normalization is allowed.
- statement may resolve a pronoun using another part of the answer, but it must express
  only an atomic claim asserted by the answer.

Judgment rules:
- verdict is 1 only if the complete statement is semantically entailed by at least one context.
- Exact wording is not required. Accept ordinary paraphrases with the same meaning, such as
  "distress calls" and "distress alerts", while keeping numbers and named entities exact.
- verdict is 0 if any material detail is absent, contradicted, or refers to a different entity,
  place, event, unit, or relationship.
- Distinguish similar locations and roles exactly. For example, Newark is not New York,
  and a shipment destination is not necessarily where an item was discovered.
- Treat metadata about evidence-unit IDs or similarity scores as factual claims when the
  answer asserts them; support them only if that metadata appears in the contexts.
- Do not use outside knowledge.
- evidence must name the context_id and quote or precisely identify the decisive passage.
- reason must briefly explain why that evidence supports or fails to support the whole claim.

Return only the JSON object required by the supplied output schema.

Input data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _validate_claim_provenance(
    answer: str,
    claims: list[StrictStatementVerdict],
) -> None:
    """Reject numeric claim mutation and duplicate claims."""
    seen_statements: set[str] = set()

    for claim in claims:
        # source_text is audit metadata, not a scoring input. Codex may normalize
        # punctuation or resolve references, so textual mismatch must not abort
        # an otherwise valid evaluation.
        # A claim may resolve a pronoun or reference across answer sentences, so its
        # exact source excerpt does not always repeat the number. Only numbers absent
        # from the complete answer are mutations.
        extra_numbers = _numbers(claim.statement) - _numbers(answer)
        if extra_numbers:
            raise RagasStructuredOutputError(
                "Codex changed or introduced a numeric value while extracting a claim: "
                + ", ".join(sorted(extra_numbers))
            )

        normalized_statement = _normalize_whitespace(claim.statement)
        if normalized_statement in seen_statements:
            raise RagasStructuredOutputError(
                "Codex returned a duplicate atomic claim."
            )
        seen_statements.add(normalized_statement)


async def score_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    progress_callback: ProgressCallback | None = None,
) -> FaithfulnessEvaluation:
    """Score faithfulness with one fresh, schema-constrained Codex exec call."""
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

    await _emit_progress(
        progress_callback,
        "preparing_contexts",
        15,
        f"평가 컨텍스트 {len(clean_contexts)}개를 준비하는 중",
    )
    prompt = _build_faithfulness_prompt(question, answer, clean_contexts)
    schema = CodexFaithfulnessOutput.model_json_schema()

    await _emit_progress(
        progress_callback,
        "starting_codex",
        25,
        "Codex 평가 프로세스를 시작하는 중",
    )
    try:
        await _emit_progress(
            progress_callback,
            "running_codex",
            35,
            "Codex exec가 답변의 claim을 분해하고 컨텍스트 근거를 판정하는 중",
        )
        raw_result = await asyncio.wait_for(
            asyncio.to_thread(
                llm_client.codex_generate_structured,
                prompt,
                schema,
            ),
            timeout=config.CODEX_TIMEOUT_SECONDS + 5,
        )
        await _emit_progress(
            progress_callback,
            "validating_output",
            85,
            "Codex 구조화 응답을 검증하는 중",
        )
        structured = CodexFaithfulnessOutput.model_validate(raw_result)
    except TimeoutError:
        raise
    except RagasStructuredOutputError:
        raise
    except (RuntimeError, ValidationError, ValueError) as exc:
        raise RagasStructuredOutputError(
            f"Codex returned invalid structured faithfulness output: {exc}"
        ) from exc

    _validate_claim_provenance(answer, structured.claims)
    await _emit_progress(
        progress_callback,
        "calculating_score",
        95,
        f"claim {len(structured.claims)}개의 최종 점수를 계산하는 중",
    )

    supported = sum(claim.verdict for claim in structured.claims)
    total = len(structured.claims)
    result = FaithfulnessEvaluation(
        value=supported / total,
        supported_claims=supported,
        total_claims=total,
        claims=structured.claims,
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
    """Evaluate generated answers sequentially with a fresh Codex judge call."""
    evaluated = []
    for item in results:
        started_at = time.perf_counter()
        metric_result = await score_faithfulness(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
        )
        score = float(metric_result.value)
        if not math.isfinite(score):
            raise ValueError("RAGAS returned a non-finite faithfulness score.")
        evaluated.append(
            {
                "question": item["question"],
                "faithfulness": score,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            }
        )
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