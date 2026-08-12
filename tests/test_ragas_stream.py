import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import QueryRequest, app
from evaluate_ragas import FaithfulnessEvaluation, StrictStatementVerdict


class TestQueryResponseCacheOption(unittest.TestCase):
    def test_query_request_uses_cache_by_default(self):
        self.assertTrue(QueryRequest(query="Question?").use_cache)

    def test_api_forwards_cache_choice_and_reports_hit(self):
        pipeline_result = {
            "answer": "Cached answer.",
            "retrieved_contexts": ["Context."],
            "raw_contexts": [],
            "latency_ms": 1,
            "api_calls": 0,
            "tokens_used": 0,
            "prompt": "Prompt.",
            "cache_hit": True,
        }
        client = TestClient(app)
        with patch("app.main.roi_pipeline", return_value=pipeline_result) as pipeline:
            response = client.post(
                "/api/query",
                json={"query": "Question?", "use_cache": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["roi_rag"]["cache_hit"])
        pipeline.assert_called_once_with("Question?", use_cache=False)

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
                ("running_gemini", 35),
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
                        claim_id="c1",
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