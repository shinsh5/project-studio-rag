import subprocess
import unittest
from unittest.mock import patch

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

    @patch("llm_client.shutil.which", return_value=r"C:\tools\codex.cmd")
    @patch("llm_client.subprocess.run")
    def test_codex_evaluator_is_ephemeral_and_read_only(self, mock_run, _mock_which):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="evaluation result\n", stderr=""
        )

        with patch.object(config, "CODEX_MODEL", "gpt-5.6-luna"):
            result = llm_client.codex_generate("evaluate this")

        self.assertEqual(result, "evaluation result")
        command = mock_run.call_args.args[0]
        self.assertIn("--ephemeral", command)
        self.assertIn("read-only", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--model", command)
        self.assertIn("gpt-5.6-luna", command)
        self.assertEqual(command[-1], "-")
        self.assertIn("evaluate this", mock_run.call_args.kwargs["input"])


if __name__ == "__main__":
    unittest.main()
