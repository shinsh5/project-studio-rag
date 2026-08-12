import unittest
from unittest.mock import patch

from pydantic import ValidationError

import config
import evaluate_ragas


def verdict(
    claim_id,
    value,
    reason="The decisive context passage supports this claim.",
    evidence="context_id 1: matching passage",
):
    return {
        "claim_id": claim_id,
        "verdict": value,
        "reason": reason,
        "evidence": evidence,
    }


class TestDeterministicClaimExtraction(unittest.TestCase):
    def test_same_answer_always_produces_same_claims_and_ids(self):
        answer = "Alpha happened. Beta happened; Gamma happened."

        first = evaluate_ragas._extract_deterministic_claims(answer)
        second = evaluate_ragas._extract_deterministic_claims(answer)

        self.assertEqual(
            [claim.model_dump() for claim in first],
            [claim.model_dump() for claim in second],
        )
        self.assertEqual([claim.claim_id for claim in first], ["c1", "c2", "c3"])
        self.assertEqual(
            [claim.statement for claim in first],
            ["Alpha happened.", "Beta happened", "Gamma happened."],
        )

    def test_preserves_abbreviations_decimals_and_currency(self):
        answer = (
            "The U.S. Coast Guard recorded 3.14 incidents. "
            "The responses cost $500,000."
        )

        claims = evaluate_ragas._extract_deterministic_claims(answer)

        self.assertEqual(len(claims), 2)
        self.assertEqual(
            claims[0].statement,
            "The U.S. Coast Guard recorded 3.14 incidents.",
        )
        self.assertEqual(claims[1].statement, "The responses cost $500,000.")

    def test_splits_numbered_lines_without_making_number_claims(self):
        claims = evaluate_ragas._extract_deterministic_claims(
            "1. Alpha happened.\n2. Beta happened."
        )

        self.assertEqual(
            [claim.statement for claim in claims],
            ["Alpha happened.", "Beta happened."],
        )

    def test_splits_compound_microsoft_answer_into_atomic_claims(self):
        answer = (
            "Microsoft acquired Nokia's cell phone business and licensed its "
            "patent portfolio to build a devices and services strategy, evolving "
            "away from its traditional focus on Windows and Office."
        )

        claims = evaluate_ragas._extract_deterministic_claims(answer)

        self.assertEqual(
            [claim.claim_id for claim in claims],
            ["c1", "c2", "c3", "c4"],
        )
        self.assertEqual(
            [claim.statement for claim in claims],
            [
                "Microsoft acquired Nokia's cell phone business.",
                "Microsoft licensed its patent portfolio.",
                "Microsoft intended to build a devices and services strategy.",
                "Microsoft is evolving away from its traditional focus on "
                "Windows and Office.",
            ],
        )
        self.assertTrue(all(claim.source_text == answer for claim in claims))

    def test_does_not_split_object_conjunction(self):
        answers = [
            "Microsoft retained Windows and Office.",
            "Microsoft acquired a software and services business.",
        ]

        statements = [
            [
                claim.statement
                for claim in evaluate_ragas._extract_deterministic_claims(answer)
            ]
            for answer in answers
        ]

        self.assertEqual(
            statements,
            [
                ["Microsoft retained Windows and Office."],
                ["Microsoft acquired a software and services business."],
            ],
        )

    def test_varied_sentences_produce_their_natural_claim_counts(self):
        cases = [
            (
                "The painting was shipped under false pretenses and "
                "discovered in Newark.",
                [
                    "The painting was shipped under false pretenses.",
                    "The painting was discovered in Newark.",
                ],
            ),
            (
                "Federal prosecutors filed papers in Brooklyn, and officials "
                "identified the shipper as Robert.",
                [
                    "Federal prosecutors filed papers in Brooklyn.",
                    "officials identified the shipper as Robert.",
                ],
            ),
            (
                "Officer Rogelio Santander died after being shot at a Home "
                "Depot store.",
                [
                    "Officer Rogelio Santander died.",
                    "Officer Rogelio Santander was shot at a Home Depot store.",
                ],
            ),
            (
                "The responses cost $500,000.",
                ["The responses cost $500,000."],
            ),
        ]

        for answer, expected in cases:
            with self.subTest(answer=answer):
                first = evaluate_ragas._extract_deterministic_claims(answer)
                second = evaluate_ragas._extract_deterministic_claims(answer)
                self.assertEqual(
                    [claim.statement for claim in first],
                    expected,
                )
                self.assertEqual(
                    [claim.model_dump() for claim in first],
                    [claim.model_dump() for claim in second],
                )

    def test_present_tense_independent_clauses_are_split(self):
        claims = evaluate_ragas._extract_deterministic_claims(
            "The report identifies the shipper, and the filing describes "
            "the destination."
        )

        self.assertEqual(len(claims), 2)



class TestFaithfulnessScoring(unittest.IsolatedAsyncioTestCase):
    async def test_scores_fixed_claims_from_one_gemini_call(self):
        answer = (
            "The caller made 28 false distress calls from Annapolis. "
            "The responses cost $500,000."
        )
        output = {"verdicts": [verdict("c1", 1), verdict("c2", 1)]}

        with (
            patch.object(config, "GEMINI_TIMEOUT_SECONDS", 10),
            patch(
                "evaluate_ragas.llm_client.gemini_generate_structured",
                return_value=output,
            ) as gemini,
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
        self.assertEqual(
            [claim.statement for claim in result.claims],
            [
                "The caller made 28 false distress calls from Annapolis.",
                "The responses cost $500,000.",
            ],
        )
        gemini.assert_called_once()
        prompt, schema = gemini.call_args.args
        self.assertIn('"claim_id": "c1"', prompt)
        self.assertIn("Unrelated context", prompt)
        self.assertIn("Do not extract, add, remove", prompt)
        self.assertNotIn("Extract every atomic factual claim", prompt)
        self.assertFalse(schema["additionalProperties"])

    async def test_reports_real_progress_stages_in_order(self):
        events = []

        async def collect_progress(event):
            events.append(event)

        output = {"verdicts": [verdict("c1", 1)]}
        with patch(
            "evaluate_ragas.llm_client.gemini_generate_structured",
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
                "starting_gemini",
                "running_gemini",
                "validating_output",
                "calculating_score",
                "completed",
            ],
        )
        self.assertIn("Fixed claims: 1", events[1]["message"])
        self.assertEqual(events[0]["progress"], 5)
        self.assertEqual(events[-1]["progress"], 100)

    async def test_each_evaluation_rejudges_same_fixed_claims_without_cache(self):
        output = {"verdicts": [verdict("c1", 1), verdict("c2", 0)]}
        with patch(
            "evaluate_ragas.llm_client.gemini_generate_structured",
            return_value=output,
        ) as gemini:
            first = await evaluate_ragas.score_faithfulness(
                "Question?",
                "Alpha happened. Beta happened.",
                ["Alpha happened."],
            )
            second = await evaluate_ragas.score_faithfulness(
                "Question?",
                "Alpha happened. Beta happened.",
                ["Alpha happened."],
            )

        self.assertEqual(gemini.call_count, 2)
        self.assertEqual(gemini.call_args_list[0].args[0], gemini.call_args_list[1].args[0])
        self.assertEqual(first.total_claims, second.total_claims)
        self.assertEqual(
            [claim.statement for claim in first.claims],
            [claim.statement for claim in second.claims],
        )

    async def test_reorders_gemini_verdicts_to_fixed_claim_order(self):
        output = {"verdicts": [verdict("c2", 0), verdict("c1", 1)]}
        with patch(
            "evaluate_ragas.llm_client.gemini_generate_structured",
            return_value=output,
        ):
            result = await evaluate_ragas.score_faithfulness(
                "Question?",
                "Alpha happened. Beta happened.",
                ["Alpha happened."],
            )

        self.assertEqual(
            [claim.statement for claim in result.claims],
            ["Alpha happened.", "Beta happened."],
        )
        self.assertEqual([claim.verdict for claim in result.claims], [1, 0])

    async def test_rejects_missing_or_unknown_claim_ids(self):
        with patch(
            "evaluate_ragas.llm_client.gemini_generate_structured",
            return_value={"verdicts": [verdict("c1", 1), verdict("c3", 0)]},
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "do not exactly match",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?",
                    "Alpha happened. Beta happened.",
                    ["Alpha happened."],
                )

    async def test_rejects_duplicate_claim_ids(self):
        repeated = verdict("c1", 1)
        with patch(
            "evaluate_ragas.llm_client.gemini_generate_structured",
            return_value={"verdicts": [repeated, repeated]},
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "duplicate claim_id",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?", "Alpha happened.", ["Alpha happened."]
                )

    async def test_gemini_cannot_rewrite_or_mutate_a_fixed_claim(self):
        output = {
            "verdicts": [
                {
                    **verdict("c1", 1),
                    "statement": "The caller made 3 false calls.",
                }
            ]
        }
        with patch(
            "evaluate_ragas.llm_client.gemini_generate_structured",
            return_value=output,
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "invalid structured",
            ):
                await evaluate_ragas.score_faithfulness(
                    "How many?",
                    "The caller made 28 false calls.",
                    ["The caller made 28 false calls."],
                )

    async def test_wraps_schema_validation_error(self):
        with patch(
            "evaluate_ragas.llm_client.gemini_generate_structured",
            return_value={"verdicts": [{"claim_id": "c1"}]},
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "invalid structured",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?", "Alpha happened.", ["Alpha happened."]
                )

    def test_prompt_accepts_semantically_equivalent_paraphrases(self):
        claims = evaluate_ragas._extract_deterministic_claims(
            "The caller made 28 distress calls."
        )
        prompt = evaluate_ragas._build_faithfulness_prompt(
            "How many calls?",
            claims,
            ["The caller made 28 distress alerts."],
        )
        self.assertIn("Exact wording is not required", prompt)
        self.assertIn('"distress calls" and "distress alerts"', prompt)

    def test_prompt_combines_support_across_contexts(self):
        claims = evaluate_ragas._extract_deterministic_claims(
            "Microsoft acquired Nokia and changed its strategy."
        )
        prompt = evaluate_ragas._build_faithfulness_prompt(
            "Why?",
            claims,
            ["Microsoft acquired Nokia.", "Microsoft changed its strategy."],
        )

        self.assertIn("one combined evidence set", prompt)
        self.assertIn("no single context must entail the complete claim", prompt)
        self.assertIn("same entity", prompt)

    def test_verdict_rejects_values_other_than_zero_or_one(self):
        with self.assertRaises(ValidationError):
            evaluate_ragas.StrictClaimVerdict(
                claim_id="c1",
                reason="Reason.",
                evidence="context_id 1",
                verdict=2,
            )

    def test_output_schema_contains_only_fixed_id_verdict_fields(self):
        schema = evaluate_ragas.GeminiFaithfulnessOutput.model_json_schema()
        verdict_schema = schema["$defs"]["StrictClaimVerdict"]
        self.assertEqual(
            set(verdict_schema["required"]),
            {"claim_id", "reason", "evidence", "verdict"},
        )
        self.assertNotIn("statement", verdict_schema["properties"])
        self.assertEqual(verdict_schema["properties"]["verdict"]["enum"], [0, 1])
        self.assertEqual(schema["required"], ["verdicts"])


if __name__ == "__main__":
    unittest.main()