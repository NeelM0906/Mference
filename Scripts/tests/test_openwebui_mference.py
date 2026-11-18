"""Standard-library tests; do not import Open WebUI or open its database."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "openwebui-mference.py"
spec = importlib.util.spec_from_file_location("adapter", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class LauncherTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
