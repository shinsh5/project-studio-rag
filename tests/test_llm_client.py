import json
import unittest
from unittest.mock import MagicMock, patch

import config
import llm_client


class TestLLMClient(unittest.TestCase):
    @patch("llm_client._ollama_generate", return_value="local answer")
    def test_internal_generation_uses_ollama(self, mock_generate):
        with patch.object(config, "LLM_BACKEND", "ollama"):
            self.assertEqual(llm_client.generate("question"), "local answer")
        mock_generate.assert_called_once_with("question")

    def test_internal_generation_rejects_non_ollama_backend(self):
        with patch.object(config, "LLM_BACKEND", "unsupported"):
            with self.assertRaisesRegex(ValueError, "Set LLM_BACKEND=ollama"):
                llm_client.generate("question")

    @patch("llm_client._ollama_generate", return_value='{"score": 1}')
    def test_local_json_generation_still_uses_ollama(self, mock_generate):
        with patch.object(config, "LLM_BACKEND", "ollama"):
            self.assertEqual(llm_client.generate_json("local task"), '{"score": 1}')
        mock_generate.assert_called_once_with("local task", json_mode=True, stop=None)

    @patch("llm_client._ollama_client")
    def test_local_generation_uses_deterministic_ollama_settings(self, mock_factory):
        mock_factory.return_value.chat.return_value = {
            "message": {"content": '{"score": 1}'}
        }

        with (
            patch.object(config, "OLLAMA_TEMPERATURE", 0.0),
            patch.object(config, "OLLAMA_SEED", 42),
            patch.object(config, "OLLAMA_TOP_K", 1),
            patch.object(config, "OLLAMA_TOP_P", 1.0),
            patch.object(config, "OLLAMA_NUM_CTX", 8192),
            patch.object(config, "OLLAMA_NUM_PREDICT", 160),
            patch.object(config, "OLLAMA_NUM_BATCH", 32),
            patch.object(config, "OLLAMA_KEEP_ALIVE", 0),
            patch.object(config, "OLLAMA_FRESH_RUNNER", True),
        ):
            result = llm_client._ollama_generate(
                "local task", json_mode=True, stop=["END"]
            )

        self.assertEqual(result, '{"score": 1}')
        request = mock_factory.return_value.chat.call_args.kwargs
        mock_factory.return_value.generate.assert_called_once_with(
            model=config.OLLAMA_MODEL,
            prompt="",
            keep_alive=0,
        )
        self.assertEqual(request["format"], "json")
        self.assertEqual(request["keep_alive"], 0)
        self.assertEqual(
            request["options"],
            {
                "temperature": 0.0,
                "seed": 42,
                "top_k": 1,
                "top_p": 1.0,
                "num_ctx": 8192,
                "num_predict": 160,
                "num_batch": 32,
                "stop": ["END"],
            },
        )

    def test_gemini_client_requires_api_key(self):
        with patch.object(llm_client, "_GEMINI_CLIENT", None):
            with patch.object(config, "GEMINI_API_KEY", ""):
                with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                    llm_client._gemini_client()

    def test_schema_conversion_inlines_refs_and_drops_unsupported_keywords(self):
        schema = {
            "$defs": {
                "Verdict": {
                    "type": "object",
                    "title": "Verdict",
                    "additionalProperties": False,
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "pattern": "^c[1-9]\\d*$",
                            "title": "Claim Id",
                        },
                        "verdict": {
                            "type": "integer",
                            "enum": [0, 1],
                            "title": "Verdict",
                        },
                    },
                    "required": ["claim_id", "verdict"],
                },
            },
            "type": "object",
            "title": "Output",
            "additionalProperties": False,
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Verdict"},
                },
            },
            "required": ["verdicts"],
        }

        prepared = llm_client._prepare_response_schema(schema)

        self.assertNotIn("$defs", prepared)
        self.assertNotIn("title", prepared)
        self.assertNotIn("additionalProperties", prepared)
        verdict_item = prepared["properties"]["verdicts"]["items"]
        self.assertNotIn("$ref", verdict_item)
        self.assertNotIn("pattern", verdict_item["properties"]["claim_id"])
        self.assertEqual(verdict_item["required"], ["claim_id", "verdict"])
        self.assertNotIn("enum", verdict_item["properties"]["verdict"])
        self.assertEqual(verdict_item["properties"]["verdict"]["type"], "integer")

    @patch.object(llm_client, "_GEMINI_CLIENT", MagicMock())
    def test_gemini_structured_output_parses_json_response(self):
        llm_client._GEMINI_CLIENT.models.generate_content.return_value = MagicMock(
            text='{"verdicts": [{"claim_id": "c1", "verdict": 1}]}'
        )

        with patch.object(config, "GEMINI_MODEL", "gemini-2.5-flash-lite"):
            result = llm_client.gemini_generate_structured(
                "judge",
                {
                    "type": "object",
                    "properties": {"verdicts": {"type": "array"}},
                    "required": ["verdicts"],
                    "additionalProperties": False,
                },
            )

        self.assertEqual(result["verdicts"][0]["claim_id"], "c1")
        call_kwargs = llm_client._GEMINI_CLIENT.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-2.5-flash-lite")
        self.assertIn("judge", call_kwargs["contents"])
        self.assertEqual(
            call_kwargs["config"].response_mime_type, "application/json"
        )

    @patch.object(llm_client, "_GEMINI_CLIENT", MagicMock())
    def test_gemini_structured_output_rejects_empty_response(self):
        llm_client._GEMINI_CLIENT.models.generate_content.return_value = MagicMock(
            text=""
        )

        with self.assertRaisesRegex(RuntimeError, "empty structured response"):
            llm_client.gemini_generate_structured(
                "judge",
                {"type": "object", "properties": {}},
            )

    @patch.object(llm_client, "_GEMINI_CLIENT", MagicMock())
    def test_gemini_structured_output_rejects_invalid_json(self):
        llm_client._GEMINI_CLIENT.models.generate_content.return_value = MagicMock(
            text="not json"
        )

        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            llm_client.gemini_generate_structured(
                "judge",
                {"type": "object", "properties": {}},
            )


if __name__ == "__main__":
    unittest.main()
