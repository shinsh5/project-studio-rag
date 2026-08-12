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


class FixedClaim(BaseModel):
    """One deterministically extracted answer claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^c[1-9]\d*$")
    source_text: str = Field(min_length=1)
    statement: str = Field(min_length=1)


class StrictClaimVerdict(BaseModel):
    """One Codex verdict for a fixed claim ID."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^c[1-9]\d*$")
    reason: str = Field(min_length=3)
    evidence: str = Field(min_length=1)
    verdict: Literal[0, 1]


class StrictStatementVerdict(BaseModel):
    """One fixed answer claim joined with its context-grounding verdict."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^c[1-9]\d*$")
    source_text: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    reason: str = Field(min_length=3)
    evidence: str = Field(min_length=1)
    verdict: Literal[0, 1]


class CodexFaithfulnessOutput(BaseModel):
    """Schema-constrained verdicts returned by one Codex exec invocation."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[StrictClaimVerdict] = Field(min_length=1)

class FaithfulnessEvaluation(BaseModel):
    value: float
    supported_claims: int
    total_claims: int
    claims: list[StrictStatementVerdict]
    contexts_evaluated: int


_PROTECTED_PERIOD = "\ue000"
_ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|U\.K)\.",
    re.IGNORECASE,
)
_INITIALISM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_COMMON_FINITE_VERBS = (
    "acquire",
    "aim",
    "announce",
    "arrest",
    "become",
    "begin",
    "build",
    "call",
    "cause",
    "charge",
    "confirm",
    "contain",
    "cost",
    "create",
    "describe",
    "develop",
    "die",
    "discover",
    "estimate",
    "evolve",
    "expect",
    "file",
    "find",
    "happen",
    "identify",
    "include",
    "indicate",
    "lead",
    "license",
    "make",
    "move",
    "plan",
    "provide",
    "receive",
    "recover",
    "remain",
    "report",
    "retain",
    "say",
    "ship",
    "shoot",
    "state",
    "support",
    "use",
    "want",
)
def _third_person_singular(verb: str) -> str:
    """Return the regular English third-person form for a verb lexeme."""
    if verb.endswith("y") and verb[-2:-1] not in "aeiou":
        return f"{verb[:-1]}ies"
    if verb.endswith(("s", "x", "z", "ch", "sh", "o")):
        return f"{verb}es"
    return f"{verb}s"


_COMMON_FINITE_PATTERN = "|".join(
    rf"{verb}|{_third_person_singular(verb)}"
    for verb in _COMMON_FINITE_VERBS
)
_PREDICATE_VERB = (
    rf"(?:[A-Za-z]+ed|is|are|was|were|has|have|had|"
    rf"made|said|did|took|found|built|bought|sold|shot|died|became|began|ran|went|"
    rf"{_COMMON_FINITE_PATTERN})"
)
_COORDINATED_PREDICATE_RE = re.compile(
    rf"^(?P<subject>.+?)\s+(?P<verb1>{_PREDICATE_VERB})\s+"
    rf"(?P<object1>.+?)\s+and\s+(?P<verb2>{_PREDICATE_VERB})\s+"
    rf"(?P<object2>.+)$",
    re.IGNORECASE,
)
_PURPOSE_VERBS = (
    "address",
    "avoid",
    "become",
    "build",
    "create",
    "develop",
    "enable",
    "ensure",
    "establish",
    "expand",
    "form",
    "help",
    "improve",
    "increase",
    "make",
    "move",
    "promote",
    "protect",
    "provide",
    "pursue",
    "raise",
    "reduce",
    "retain",
    "strengthen",
    "support",
)
_PURPOSE_RE = re.compile(
    rf"\s+to\s+(?P<purpose>(?:{'|'.join(_PURPOSE_VERBS)})\b.+)$",
    re.IGNORECASE,
)
_TRAILING_PARTICIPLE_VERBS = (
    "building",
    "creating",
    "developing",
    "enabling",
    "evolving",
    "expanding",
    "helping",
    "improving",
    "increasing",
    "moving",
    "providing",
    "reducing",
    "retaining",
    "shifting",
    "strengthening",
    "supporting",
    "transitioning",
    "using",
)
_TRAILING_PARTICIPLE_RE = re.compile(
    rf",\s*(?P<participle>(?:{'|'.join(_TRAILING_PARTICIPLE_VERBS)})\b.+)$",
    re.IGNORECASE,
)
_INDEPENDENT_CONJUNCTION_RE = re.compile(
    r",\s+(?:and|but|while)\s+",
    re.IGNORECASE,
)
_PASSIVE_ADVERBIAL_RE = re.compile(
    r"\s+(?P<link>after|before|by|while)\s+being\s+"
    r"(?P<participle>[A-Za-z]+ed|built|bought|found|made|shot|sold)\b"
    r"(?P<details>.*)$",
    re.IGNORECASE,
)


def _protect_periods(text: str) -> str:
    """Protect periods that are not sentence boundaries."""
    protected = re.sub(r"(?<=\d)\.(?=\d)", _PROTECTED_PERIOD, text)
    protected = _ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", _PROTECTED_PERIOD),
        protected,
    )
    return _INITIALISM_RE.sub(
        lambda match: match.group(0).replace(".", _PROTECTED_PERIOD),
        protected,
    )


def _finish_claim(text: str, terminal: str) -> str:
    """Normalize a reconstructed atomic claim and preserve terminal punctuation."""
    normalized = " ".join(text.split()).strip(" ,")
    if terminal and normalized and normalized[-1] not in ".!?":
        normalized += terminal
    return normalized


def _leading_subject(statement: str) -> str | None:
    """Return a conservative leading subject for predicate reconstruction."""
    match = re.match(
        rf"^(?P<subject>.+?)\s+{_PREDICATE_VERB}\b",
        statement,
        re.IGNORECASE,
    )
    return match.group("subject").strip() if match else None


def _split_atomic_statement(statement: str) -> list[str]:
    """Deterministically split common English compound predicates into facts."""
    terminal = statement[-1] if statement[-1:] in ".!?" else ""
    core = statement[:-1].strip() if terminal else statement.strip()

    independent_parts = _INDEPENDENT_CONJUNCTION_RE.split(core)
    if len(independent_parts) > 1 and all(
        _leading_subject(part) for part in independent_parts
    ):
        return [
            atomic
            for part in independent_parts
            for atomic in _split_atomic_statement(
                _finish_claim(part, terminal)
            )
        ]

    derived: list[str] = []
    passive_match = _PASSIVE_ADVERBIAL_RE.search(core)
    if passive_match:
        main_clause = core[:passive_match.start()].strip()
        subject = _leading_subject(main_clause)
        if subject:
            participle = passive_match.group("participle")
            details = passive_match.group("details").strip()
            auxiliary = (
                "were"
                if subject.lower() in {"they", "we", "you"}
                else "was"
            )
            derived.append(
                _finish_claim(
                    f"{subject} {auxiliary} {participle} {details}",
                    terminal,
                )
            )
            core = main_clause

    participle_match = _TRAILING_PARTICIPLE_RE.search(core)
    if participle_match:
        participle = participle_match.group("participle")
        core = core[:participle_match.start()].strip()
        subject = _leading_subject(core)
        if subject:
            derived.append(_finish_claim(f"{subject} is {participle}", terminal))

    purpose_match = _PURPOSE_RE.search(core)
    if purpose_match:
        purpose = purpose_match.group("purpose")
        core = core[:purpose_match.start()].strip()
        subject = _leading_subject(core)
        if subject:
            derived.insert(
                0,
                _finish_claim(f"{subject} intended to {purpose}", terminal),
            )

    coordinated = _COORDINATED_PREDICATE_RE.match(core)
    if coordinated:
        subject = coordinated.group("subject").strip()
        verb1 = coordinated.group("verb1")
        verb2 = coordinated.group("verb2")
        if (
            verb1.lower() in {"is", "are", "was", "were", "has", "have", "had"}
            and re.fullmatch(
                r"[A-Za-z]+ed|built|bought|found|made|shot|sold",
                verb2,
                re.IGNORECASE,
            )
        ):
            verb2 = f"{verb1} {verb2}"
        primary = [
            _finish_claim(
                f"{subject} {verb1} {coordinated.group('object1')}",
                terminal,
            ),
            _finish_claim(
                f"{subject} {verb2} {coordinated.group('object2')}",
                terminal,
            ),
        ]
    else:
        primary = [_finish_claim(core, terminal)]

    return [claim for claim in [*primary, *derived] if claim]


def _extract_deterministic_claims(answer: str) -> list[FixedClaim]:
    """Split an answer into fresh deterministic atomic claims without a cache."""
    claims: list[FixedClaim] = []
    blocks = re.split(r"[\r\n]+", answer)

    for block in blocks:
        block = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s+", "", block).strip()
        if not block:
            continue

        protected = _protect_periods(block)
        protected = re.sub(r"([.!?][\"')\]]*)\s+", r"\1\n", protected)
        for sentence in protected.split("\n"):
            for clause in re.split(r"\s*;\s*", sentence):
                statement = " ".join(
                    clause.replace(_PROTECTED_PERIOD, ".").split()
                ).strip()
                if not statement or not re.search(r"\w", statement, re.UNICODE):
                    continue
                for atomic_statement in _split_atomic_statement(statement):
                    claim_id = f"c{len(claims) + 1}"
                    claims.append(
                        FixedClaim(
                            claim_id=claim_id,
                            source_text=statement,
                            statement=atomic_statement,
                        )
                    )

    if claims:
        return claims

    fallback = " ".join(answer.split()).strip()
    return [
        FixedClaim(claim_id="c1", source_text=fallback, statement=fallback)
    ]


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
    claims: list[FixedClaim],
    contexts: list[str],
) -> str:
    payload = {
        "question": question,
        "fixed_claims": [claim.model_dump() for claim in claims],
        "contexts": [
            {"context_id": index + 1, "text": context}
            for index, context in enumerate(contexts)
        ],
    }
    return f"""Evaluate RAGAS faithfulness for the supplied fixed claims.

The claims were extracted deterministically by the application. Judge them using only
supplied contexts. Do not extract, add, remove, merge, split, rewrite, renumber, or
reorder claims.

Judgment rules:
- Treat all supplied contexts as one combined evidence set. Different parts of a claim may
  be supported by different contexts; no single context must entail the complete claim.
- verdict is 1 only if every factual component of the statement is semantically entailed
  somewhere in that combined evidence set.
- Combine evidence across contexts only when the passages clearly concern the same entity
  and event and contain no contradiction or unsupported causal bridge.
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
- Return exactly one verdict for every fixed claim_id and no other claim_id.

Return only the JSON object required by the supplied output schema.

Input data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

def _merge_fixed_claims_with_verdicts(
    claims: list[FixedClaim],
    verdicts: list[StrictClaimVerdict],
) -> list[StrictStatementVerdict]:
    """Require an exact one-to-one verdict mapping for fixed claim IDs."""
    expected_ids = [claim.claim_id for claim in claims]
    actual_ids = [verdict.claim_id for verdict in verdicts]
    if len(set(actual_ids)) != len(actual_ids):
        raise RagasStructuredOutputError("Codex returned a duplicate claim_id.")
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise RagasStructuredOutputError(
            "Codex verdict claim IDs do not exactly match the fixed claim set. "
            f"Expected {expected_ids}, received {actual_ids}."
        )

    verdict_by_id = {verdict.claim_id: verdict for verdict in verdicts}
    return [
        StrictStatementVerdict(
            claim_id=claim.claim_id,
            source_text=claim.source_text,
            statement=claim.statement,
            reason=verdict_by_id[claim.claim_id].reason,
            evidence=verdict_by_id[claim.claim_id].evidence,
            verdict=verdict_by_id[claim.claim_id].verdict,
        )
        for claim in claims
    ]

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

    fixed_claims = _extract_deterministic_claims(answer)

    await _emit_progress(
        progress_callback,
        "preparing_contexts",
        15,
        f"Fixed claims: {len(fixed_claims)}; contexts: {len(clean_contexts)}",
    )
    prompt = _build_faithfulness_prompt(question, fixed_claims, clean_contexts)
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
            "Codex exec is judging only the fixed claims against the contexts.",
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

    scored_claims = _merge_fixed_claims_with_verdicts(
        fixed_claims,
        structured.verdicts,
    )
    await _emit_progress(
        progress_callback,
        "calculating_score",
        95,
        f"claim {len(scored_claims)}개의 최종 점수를 계산하는 중",
    )

    supported = sum(claim.verdict for claim in scored_claims)
    total = len(scored_claims)
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