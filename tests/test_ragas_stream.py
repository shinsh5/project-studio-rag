import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import config
from app.main import BuildIndexRequest, QueryRequest, app
from evaluate_ragas import FaithfulnessEvaluation, StatementVerdict


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
        pipeline.assert_called_once_with(
            "Question?", use_cache=False, stb_retrieval_mode=None
        )

class TestAutoMergeThresholdScope(unittest.TestCase):
    """The threshold is a query-time knob: config is the default and stays intact."""

    PIPELINE_RESULT = {
        "answer": "A.", "retrieved_contexts": [], "raw_contexts": [],
        "latency_ms": 1, "api_calls": 0, "tokens_used": 0,
        "prompt": "P.", "cache_hit": False,
    }

    def _query(self, payload):
        client = TestClient(app)
        with patch("app.main.roi_pipeline", return_value=self.PIPELINE_RESULT):
            return client.post("/api/query", json=payload)

    def test_build_request_has_no_threshold_field(self):
        # Building must not be able to pin a retrieval-time value.
        self.assertNotIn("automerge_threshold", BuildIndexRequest.model_fields)

    def test_omitted_threshold_leaves_config_untouched(self):
        with patch.object(config, "AUTOMERGE_THRESHOLD", 0.3):
            self._query({"query": "Q?"})
            self.assertEqual(config.AUTOMERGE_THRESHOLD, 0.3)

    def test_supplied_threshold_is_restored_after_the_query(self):
        with patch.object(config, "AUTOMERGE_THRESHOLD", 0.0):
            self._query({"query": "Q?", "automerge_threshold": 0.7})
            # Without the restore, the next query would silently inherit 0.7.
            self.assertEqual(config.AUTOMERGE_THRESHOLD, 0.0)

    def test_threshold_is_restored_even_when_the_pipeline_fails(self):
        client = TestClient(app)
        with patch.object(config, "AUTOMERGE_THRESHOLD", 0.0):
            with patch("app.main.roi_pipeline", side_effect=RuntimeError("boom")):
                client.post(
                    "/api/query",
                    json={"query": "Q?", "automerge_threshold": 0.7},
                )
            self.assertEqual(config.AUTOMERGE_THRESHOLD, 0.0)


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
                ("generating_statements", 25),
                ("judging_statements", 60),
                ("calculating_score", 95),
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
                    StatementVerdict(
                        claim_id="c1",
                        statement="Alpha happened.",
                        reason="Supported by context 1.",
                        verdict=1,
                    )
                ],
                contexts_evaluated=1,
            )

        async def fake_relevancy(question, answer):
            return 0.87

        client = TestClient(app)
        with patch("app.main.score_faithfulness", side_effect=fake_score), patch(
            "app.main.score_response_relevancy", side_effect=fake_relevancy
        ):
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
            [
                "progress",
                "progress",
                "progress",
                "progress",
                "progress",
                "progress",
                "result",
            ],
        )
        self.assertEqual(events[0]["stage"], "validating_input")
        self.assertEqual(events[-2]["stage"], "answer_relevancy")
        self.assertEqual(events[-1]["scores"]["faithfulness"], 1.0)
        self.assertEqual(events[-1]["scores"]["answer_relevancy"], 0.87)


if __name__ == "__main__":
    unittest.main()