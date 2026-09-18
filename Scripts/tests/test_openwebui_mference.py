"""Standard-library tests; do not import Open WebUI or open its database."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "openwebui-mference.py"
spec = importlib.util.spec_from_file_location("adapter", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ReasoningHistoryTests(unittest.TestCase):
    def test_check_mode_needs_no_secret_or_application_import(self):
        with patch("sys.argv", [str(path), "check"]), \
             patch.object(adapter, "version", return_value=adapter.SUPPORTED_VERSION), \
             patch.dict(adapter.os.environ, {}, clear=True):
            adapter.main()

    def test_check_mode_rejects_unsupported_package(self):
        with patch("sys.argv", [str(path), "check"]), \
             patch.object(adapter, "version", return_value="0.0.1"):
            with self.assertRaises(SystemExit) as failure:
                adapter.main()
            self.assertEqual(failure.exception.code, 2)

    def test_swift_and_library_duplicates_preserve_reasoning(self):
        for identifier in (adapter.SWIFT_ID, adapter.SWIFT_ID + "@copy#2"):
            self.assertEqual(adapter.reasoning_format(lambda _: None,
                {"id": identifier, "owned_by": "mference"}), "reasoning_content")

    def test_other_models_and_providers_keep_original_policy(self):
        for model in ({}, {"id": adapter.SWIFT_ID, "owned_by": "other"},
                      {"id": "qwen3.8-27b-4bit", "owned_by": "mference"},
                      {"id": adapter.SWIFT_ID + "-other", "owned_by": "mference"}):
            seen = []
            def original(value):
                seen.append(value)
                return "original-policy"
            self.assertEqual(adapter.reasoning_format(original, model), "original-policy")
            self.assertEqual(seen, [model])

    def test_openwebui_merged_identity(self):
        model = {"id": adapter.SWIFT_ID, "owned_by": "openai",
                 "openai": {"id": adapter.SWIFT_ID, "owned_by": "mference"}}
        self.assertEqual(adapter.reasoning_format(lambda _: None, model), "reasoning_content")


if __name__ == "__main__":
    unittest.main()
