import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from evaluate_ragas import FaithfulnessEvaluation, StrictStatementVerdict


class TestRagasProgressStream(unittest.TestCase):
    def test_streams_progress_events_before_result(self):
        async def fake_score(
            question,
            answer,
            contexts,
            progress_callback=None,
        ):
            for stage, progress in (
                ("validating_input", 5),
                ("running_codex", 35),
                ("validating_output", 85),
                ("completed", 100),
            ):
                await progress_callback(
                    {
                        "stage": stage,
                        "progress": progress,
                        "message": stage,
                    }
                )
            return FaithfulnessEvaluation(
                value=1.0,
                supported_claims=1,
                total_claims=1,
                claims=[
                    StrictStatementVerdict(
                        source_text="Alpha happened.",
                        statement="Alpha happened.",
                        reason="Supported by context 1.",
                        evidence="context_id 1: Alpha happened.",
                        verdict=1,
                    )
                ],
                contexts_evaluated=1,
            )

        client = TestClient(app)
        with patch("app.main.score_faithfulness", side_effect=fake_score):
            with client.stream(
                "POST",
                "/api/evaluate-single-stream",
                json={
                    "question": "What happened?",
                    "answer": "Alpha happened.",
                    "contexts": ["Alpha happened."],
                },
            ) as response:
                self.assertEqual(response.status_code, 200)
                self.assertTrue(
                    response.headers["content-type"].startswith(
                        "application/x-ndjson"
                    )
                )
                events = [
                    json.loads(line)
                    for line in response.iter_lines()
                    if line.strip()
                ]

        self.assertEqual(
            [event["type"] for event in events],
            ["progress", "progress", "progress", "progress", "result"],
        )
        self.assertEqual(events[0]["stage"], "validating_input")
        self.assertEqual(events[-1]["scores"]["faithfulness"], 1.0)


if __name__ == "__main__":
    unittest.main()