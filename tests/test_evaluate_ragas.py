import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ragas.metrics._faithfulness import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)

import config
import evaluate_ragas


def statements_output(*texts: str) -> StatementGeneratorOutput:
    return StatementGeneratorOutput(statements=list(texts))


def verdicts_output(*items: tuple[str, int, str]) -> NLIStatementOutput:
    return NLIStatementOutput(
        statements=[
            StatementFaithfulnessAnswer(statement=text, verdict=verdict, reason=reason)
            for text, verdict, reason in items
        ]
    )


class TestFaithfulnessScoring(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch(
            "evaluate_ragas.llm_client.get_ragas_llm",
            return_value=MagicMock(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_scores_statements_using_ragas_statement_generation_and_nli(self):
        with (
            patch.object(config, "GEMINI_TIMEOUT_SECONDS", 10),
            patch(
                "evaluate_ragas.Faithfulness._create_statements",
                new_callable=AsyncMock,
                return_value=statements_output(
                    "The caller made 28 false distress calls from Annapolis.",
                    "The responses cost $500,000.",
                ),
            ) as create_statements,
            patch(
                "evaluate_ragas.Faithfulness._create_verdicts",
                new_callable=AsyncMock,
                return_value=verdicts_output(
                    ("The caller made 28 false distress calls from Annapolis.", 1, "Supported."),
                    ("The responses cost $500,000.", 1, "Supported."),
                ),
            ) as create_verdicts,
        ):
            result = await evaluate_ragas.score_faithfulness(
                question="How many calls and what cost?",
                answer=(
                    "The caller made 28 false distress calls from Annapolis. "
                    "The responses cost $500,000."
                ),
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
        self.assertEqual([claim.claim_id for claim in result.claims], ["c1", "c2"])

        create_statements.assert_called_once()
        row = create_statements.call_args.args[0]
        self.assertEqual(row["user_input"], "How many calls and what cost?")
        self.assertEqual(row["retrieved_contexts"], [
            "The caller made 28 false distress alerts from Annapolis.",
            "The responses cost $500,000.",
            "Unrelated context.",
        ])

        create_verdicts.assert_called_once()
        verdict_row, statements = create_verdicts.call_args.args
        self.assertIs(verdict_row, row)
        self.assertEqual(
            statements,
            [
                "The caller made 28 false distress calls from Annapolis.",
                "The responses cost $500,000.",
            ],
        )

    async def test_reports_real_progress_stages_in_order(self):
        events = []

        async def collect_progress(event):
            events.append(event)

        with (
            patch(
                "evaluate_ragas.Faithfulness._create_statements",
                new_callable=AsyncMock,
                return_value=statements_output("Alpha happened."),
            ),
            patch(
                "evaluate_ragas.Faithfulness._create_verdicts",
                new_callable=AsyncMock,
                return_value=verdicts_output(("Alpha happened.", 1, "Supported.")),
            ),
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
                "generating_statements",
                "judging_statements",
                "calculating_score",
                "completed",
            ],
        )
        self.assertEqual(events[0]["progress"], 5)
        self.assertEqual(events[-1]["progress"], 100)

    async def test_rejects_empty_question(self):
        with self.assertRaisesRegex(ValueError, "question is missing"):
            await evaluate_ragas.score_faithfulness("  ", "Answer.", ["Context."])

    async def test_rejects_empty_answer(self):
        with self.assertRaisesRegex(ValueError, "answer is missing"):
            await evaluate_ragas.score_faithfulness("Question?", "  ", ["Context."])

    async def test_rejects_empty_contexts(self):
        with self.assertRaisesRegex(ValueError, "contexts are missing"):
            await evaluate_ragas.score_faithfulness("Question?", "Answer.", ["  "])

    async def test_raises_when_ragas_produces_no_statements(self):
        with patch(
            "evaluate_ragas.Faithfulness._create_statements",
            new_callable=AsyncMock,
            return_value=statements_output(),
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "no statements",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?", "Alpha happened.", ["Alpha happened."]
                )

    async def test_wraps_llm_errors_from_statement_generation(self):
        with patch(
            "evaluate_ragas.Faithfulness._create_statements",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Gemini exploded"),
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "invalid statement generation output",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?", "Alpha happened.", ["Alpha happened."]
                )

    async def test_wraps_llm_errors_from_verdict_generation(self):
        with (
            patch(
                "evaluate_ragas.Faithfulness._create_statements",
                new_callable=AsyncMock,
                return_value=statements_output("Alpha happened."),
            ),
            patch(
                "evaluate_ragas.Faithfulness._create_verdicts",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Gemini exploded"),
            ),
        ):
            with self.assertRaisesRegex(
                evaluate_ragas.RagasStructuredOutputError,
                "invalid structured faithfulness output",
            ):
                await evaluate_ragas.score_faithfulness(
                    "Question?", "Alpha happened.", ["Alpha happened."]
                )

    async def test_computes_partial_score_from_mixed_verdicts(self):
        with (
            patch(
                "evaluate_ragas.Faithfulness._create_statements",
                new_callable=AsyncMock,
                return_value=statements_output("Alpha happened.", "Beta happened."),
            ),
            patch(
                "evaluate_ragas.Faithfulness._create_verdicts",
                new_callable=AsyncMock,
                return_value=verdicts_output(
                    ("Alpha happened.", 1, "Supported."),
                    ("Beta happened.", 0, "Not supported."),
                ),
            ),
        ):
            result = await evaluate_ragas.score_faithfulness(
                "Question?",
                "Alpha happened. Beta happened.",
                ["Alpha happened."],
            )

        self.assertEqual(result.value, 0.5)
        self.assertEqual(result.supported_claims, 1)
        self.assertEqual(result.total_claims, 2)
        self.assertEqual([claim.verdict for claim in result.claims], [1, 0])


if __name__ == "__main__":
    unittest.main()
