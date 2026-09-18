"""Launcher checks without building, downloading, starting services or touching user data."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "mference-ui.sh"


class LauncherTests(unittest.TestCase):
    def run_launcher(self, *args, env=None):
        return subprocess.run(["/bin/bash", str(LAUNCHER), *args], cwd=ROOT,
                              env=env, text=True, capture_output=True, timeout=15)

    def test_help_needs_no_dependencies(self):
        result = self.run_launcher("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("doctor", result.stdout)

    def test_missing_values_are_actionable(self):
        for flag in ("--library", "--model", "--server-port", "--webui-port",
                     "--max-context", "--prompt-cache-mode", "--build-path", "--data-dir"):
            for suffix in ([], ["--dry-run"]):
                with self.subTest(flag=flag, suffix=suffix):
                    result = self.run_launcher(flag, *suffix)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("needs a value", result.stderr)
                    self.assertNotIn("unbound variable", result.stderr)

    def test_invalid_options_fail_before_side_effects(self):
        for args in [("--server-port", "0"), ("--server-port", "65536"),
                     ("--webui-port", "text"), ("--max-context", "-1"),
                     ("--max-context", "999999999999999999999"),
                     ("--prompt-cache-mode", "unknown"),
                     ("--server-port", "3000")]:
            with self.subTest(args=args):
                result = self.run_launcher(*args, "--dry-run")
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("would run", result.stdout)

    def test_decimal_ports_and_build_path(self):
        result = self.run_launcher("--server-port", "08080", "--build-path", "build with spaces", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--port 8080", result.stdout)
        self.assertIn("build with spaces/release/MferenceServer", result.stdout)
        self.assertIn("(incremental)", result.stdout)
        self.assertNotIn("if missing", result.stdout)

    def test_install_dry_run_does_not_download(self):
        result = self.run_launcher("--build-path", "/tmp/mference-test-build", "install", "qwen36", "--resume", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--scratch-path /tmp/mference-test-build", result.stdout)
        self.assertIn("--resume", result.stdout)

    def test_isolated_data_and_doctor_dry_run(self):
        result = self.run_launcher("--data-dir", "/tmp/mference-test-data", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DATA_DIR=/tmp/mference-test-data", result.stdout)
        self.assertIn("/tmp/mference-test-data/webui-secret-key", result.stdout)
        result = self.run_launcher("doctor", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("would check", result.stdout)

    def test_browser_origins_are_limited_to_the_selected_loopback_port(self):
        result = self.run_launcher("--webui-port", "18490", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CORS_ALLOW_ORIGIN=http://127.0.0.1:18490;http://localhost:18490", result.stdout)
        self.assertNotIn("CORS_ALLOW_ORIGIN=*", result.stdout)

    def test_platform_and_occupied_port_checks(self):
        with tempfile.TemporaryDirectory(prefix="mference-launcher-test-") as directory:
            mock_bin = Path(directory)
            def mock(name, body):
                script = mock_bin / name
                script.write_text("#!/bin/sh\n" + body + "\n")
                script.chmod(0o700)
            env = dict(os.environ, PATH=str(mock_bin) + ":/usr/bin:/bin:/usr/sbin:/sbin")
            mock("uname", 'case "$1" in -s) echo Linux;; -m) echo x86_64;; esac')
            result = self.run_launcher("doctor", env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Apple Silicon", result.stderr)
            mock("uname", 'case "$1" in -s) echo Darwin;; -m) echo arm64;; esac')
            mock("sw_vers", "echo 26.3")
            mock("swift", "echo 'Apple Swift version 6.3.3'")
            mock("pgrep", "exit 1")
            mock("lsof", "echo 12345")
            result = self.run_launcher("doctor", env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("port 8080 is already in use", result.stderr)
            self.assertIn("No existing process was stopped", result.stderr)


if __name__ == "__main__":
    unittest.main()
