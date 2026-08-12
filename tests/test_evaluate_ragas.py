import unittest
from unittest.mock import patch

from pydantic import ValidationError

import config
import evaluate_ragas


def claim(
    source_text,
    statement,
    verdict,
    reason="The decisive context passage supports this claim.",
    evidence="context_id 1: matching passage",
):
    return {
        "source_text": source_text,
        "statement": statement,
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
    }


class TestFaithfulnessScoring(unittest.IsolatedAsyncioTestCase):
    async def test_scores_atomic_claims_from_one_codex_call(self):
        answer = (
            "The caller made 28 false distress calls from Annapolis. "
            "The responses cost $500,000."
        )
        output = {
            "claims": [
                claim(
                    "The caller made 28 false distress calls from Annapolis.",
                    "The caller made 28 false distress calls from Annapolis.",
                    1,
                ),
                claim(
                    "The responses cost $500,000.",
                    "The responses cost $500,000.",
                    1,
                ),
            ]
        }

        with (
            patch.object(config, "CODEX_TIMEOUT_SECONDS", 10),
            patch(
                "evaluate_ragas.llm_client.codex_generate_structured",
                return_value=output,
            ) as codex,
        ):
            result = await evaluate_ragas.score_faithfulness(
                question="How many calls and what cost?",
                answer=answer,
                contexts=[
                    "The caller made 28 false distress alerts from Annapolis.",
                    "The responses cost $500,000.",
                    "Unrelated context.",
                ],
            )

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.supported_claims, 2)
        self.assertEqual(result.total_claims, 2)
        self.assertEqual(result.contexts_evaluated, 3)
        codex.assert_called_once()
        prompt, schema = codex.call_args.args
        self.assertIn("28 false distress", prompt)
        self.assertIn("Unrelated context", prompt)
        self.assertFalse(schema["additionalProperties"])

    async def test_claim_count_is_not_fixed(self):
        answer = "Alpha happened. Beta happened. Gamma happened."
        output = {
            "claims": [
                claim("Alpha happened.", "Alpha happened.", 1),
                claim("Beta happened.", "Beta happened.", 0),
                claim("Gamma happened.", "Gamma happened.", 1),
            ]
        }
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value=output,
        ):
            result = await evaluate_ragas.score_faithfulness(
                "What happened?", answer, ["Alpha happened. Gamma happened."]
            )

        self.assertEqual(result.total_claims, 3)
        self.assertAlmostEqual(result.value, 2 / 3)

    async def test_reports_real_progress_stages_in_order(self):
        events = []

        async def collect_progress(event):
            events.append(event)

        output = {"claims": [claim("Alpha happened.", "Alpha happened.", 1)]}
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value=output,
        ):
            result = await evaluate_ragas.score_faithfulness(
                "Question?",
                "Alpha happened.",
                ["Alpha happened."],
                progress_callback=collect_progress,
            )

        self.assertEqual(result.value, 1.0)
        self.assertEqual(
            [event["stage"] for event in events],
            [
                "validating_input",
                "preparing_contexts",
                "starting_codex",
                "running_codex",
                "validating_output",
                "calculating_score",
                "completed",
            ],
        )
        self.assertEqual(events[0]["progress"], 5)
        self.assertEqual(events[-1]["progress"], 100)

    async def test_each_evaluation_runs_codex_again_without_result_cache(self):
        output = {"claims": [claim("Alpha happened.", "Alpha happened.", 1)]}
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value=output,
        ) as codex:
            await evaluate_ragas.score_faithfulness(
                "Question?", "Alpha happened.", ["Alpha happened."]
            )
            await evaluate_ragas.score_faithfulness(
                "Question?", "Alpha happened.", ["Alpha happened."]
            )

        self.assertEqual(codex.call_count, 2)

    async def test_source_text_normalization_does_not_abort_evaluation(self):
        output = {
            "claims": [
                claim(
                    "Caller made twenty-eight false calls",
                    "The caller made 28 false calls.",
                    1,
                )
            ]
        }
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value=output,
        ):
            result = await evaluate_ragas.score_faithfulness(
                "How many?",
                "The caller made 28 false calls.",
                ["The caller made 28 false calls."],
            )

        self.assertEqual(result.value, 1.0)

    async def test_rejects_number_introduced_during_claim_extraction(self):
        output = {
            "claims": [
                claim(
                    "The caller made false calls.",
                    "The caller made 3 false calls.",
                    1,
                )
            ]
        }
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value=output,
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "numeric value",
            ):
                await evaluate_ragas.score_faithfulness(
                    "How many?",
                    "The caller made false calls.",
                    ["The caller made 28 false calls."],
                )

    async def test_allows_answer_number_resolved_outside_source_excerpt(self):
        answer = (
            "The caller made 28 false distress calls. "
            "This information was reported by the Coast Guard."
        )
        output = {
            "claims": [
                claim(
                    "This information was reported by the Coast Guard.",
                    "The Coast Guard reported that the caller made 28 false distress calls.",
                    1,
                )
            ]
        }
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value=output,
        ):
            result = await evaluate_ragas.score_faithfulness(
                "How many?",
                answer,
                ["The Coast Guard reported 28 false distress calls."],
            )

        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.claims[0].verdict, 1)

    async def test_rejects_duplicate_atomic_claims(self):
        repeated = claim("Alpha happened.", "Alpha happened.", 1)
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value={"claims": [repeated, repeated]},
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "duplicate",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?", "Alpha happened.", ["Alpha happened."]
                )

    async def test_wraps_schema_validation_error(self):
        with patch(
            "evaluate_ragas.llm_client.codex_generate_structured",
            return_value={"claims": [{"statement": "missing fields"}]},
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "invalid structured",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?", "Alpha happened.", ["Alpha happened."]
                )

    def test_prompt_accepts_semantically_equivalent_paraphrases(self):
        prompt = evaluate_ragas._build_faithfulness_prompt(
            "How many calls?",
            "The caller made 28 distress calls.",
            ["The caller made 28 distress alerts."],
        )
        self.assertIn("Exact wording is not required", prompt)
        self.assertIn('"distress calls" and "distress alerts"', prompt)

    def test_verdict_rejects_values_other_than_zero_or_one(self):
        with self.assertRaises(ValidationError):
            evaluate_ragas.StrictStatementVerdict(
                source_text="Alpha happened.",
                statement="Alpha happened.",
                reason="Reason.",
                evidence="context_id 1",
                verdict=2,
            )

    def test_output_schema_requires_all_claim_fields(self):
        schema = evaluate_ragas.CodexFaithfulnessOutput.model_json_schema()
        verdict_schema = schema["$defs"]["StrictStatementVerdict"]
        self.assertEqual(
            set(verdict_schema["required"]),
            {"source_text", "statement", "reason", "evidence", "verdict"},
        )
        self.assertEqual(verdict_schema["properties"]["verdict"]["enum"], [0, 1])


if __name__ == "__main__":
    unittest.main()