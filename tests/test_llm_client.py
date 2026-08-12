import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import llm_client


class TestLLMClient(unittest.TestCase):
    def test_bare_windows_codex_path_resolves_through_appdata_npm(self):
        expected = r"C:\Users\runner\AppData\Roaming\npm\codex.cmd"
        with patch("config.os.path.isfile", return_value=True):
            resolved = config._resolve_codex_cli_path(
                "codex.cmd",
                platform_name="nt",
                appdata=r"C:\Users\runner\AppData\Roaming",
            )

        self.assertEqual(resolved, expected)

    def test_explicit_codex_path_is_preserved(self):
        explicit = r"D:\tools\codex.cmd"
        self.assertEqual(
            config._resolve_codex_cli_path(
                explicit,
                platform_name="nt",
                appdata=r"C:\Users\runner\AppData\Roaming",
            ),
            explicit,
        )

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
        self.assertIn("--ignore-rules", command)
        self.assertIn("--model", command)
        self.assertIn("gpt-5.6-luna", command)
        self.assertEqual(command[-1], "-")
        self.assertIn("evaluate this", mock_run.call_args.kwargs["input"])

    @patch("llm_client.shutil.which", return_value=r"C:\tools\codex.cmd")
    @patch("llm_client.subprocess.run")
    def test_codex_structured_output_uses_schema_and_cleans_temp_files(
        self, mock_run, _mock_which
    ):
        captured_paths = {}

        def fake_run(command, **kwargs):
            schema_path = command[command.index("--output-schema") + 1]
            output_path = command[command.index("-o") + 1]
            captured_paths.update(schema=schema_path, output=output_path)
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            self.assertEqual(schema["required"], ["claims"])
            Path(output_path).write_text(
                '{"claims":[{"statement":"claim"}]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )

        mock_run.side_effect = fake_run
        result = llm_client.codex_generate_structured(
            "judge",
            {
                "type": "object",
                "properties": {"claims": {"type": "array"}},
                "required": ["claims"],
                "additionalProperties": False,
            },
        )

        self.assertEqual(result["claims"][0]["statement"], "claim")
        self.assertFalse(Path(captured_paths["schema"]).exists())
        self.assertFalse(Path(captured_paths["output"]).exists())
        command = mock_run.call_args.args[0]
        self.assertIn("--output-schema", command)
        self.assertIn("-o", command)
        self.assertEqual(command[-1], "-")


if __name__ == "__main__":
    unittest.main()