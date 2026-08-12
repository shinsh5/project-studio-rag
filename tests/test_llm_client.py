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

    def test_get_ragas_llm_requires_api_key(self):
        with patch.object(llm_client, "_RAGAS_LLM", None):
            with patch.object(config, "GEMINI_API_KEY", ""):
                with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                    llm_client.get_ragas_llm()

    def test_get_ragas_llm_wraps_gemini_chat_model(self):
        mock_chat_model = MagicMock()
        with (
            patch.object(llm_client, "_RAGAS_LLM", None),
            patch.object(config, "GEMINI_API_KEY", "test-key"),
            patch.object(config, "GEMINI_MODEL", "gemini-2.5-flash-lite"),
            patch.object(config, "GEMINI_TEMPERATURE", 0.0),
            patch(
                "langchain_google_genai.ChatGoogleGenerativeAI",
                return_value=mock_chat_model,
            ) as mock_chat_cls,
        ):
            wrapper = llm_client.get_ragas_llm()

        mock_chat_cls.assert_called_once_with(
            model="gemini-2.5-flash-lite",
            google_api_key="test-key",
            temperature=0.0,
        )
        self.assertIs(wrapper.langchain_llm, mock_chat_model)

    def test_get_ragas_llm_is_cached(self):
        with (
            patch.object(llm_client, "_RAGAS_LLM", None),
            patch.object(config, "GEMINI_API_KEY", "test-key"),
            patch("langchain_google_genai.ChatGoogleGenerativeAI"),
        ):
            first = llm_client.get_ragas_llm()
            second = llm_client.get_ragas_llm()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
