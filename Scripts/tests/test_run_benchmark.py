"""Exercise benchmark safety without loading any model or requiring macOS."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BenchmarkSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy(ROOT / "run-benchmark.sh", self.root)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.script("sw_vers", "echo 26.3")
        self.script("swift", "echo 'Apple Swift version 6.3.3'")
        self.script("df", "printf 'header\\na b c 1000\\n'")
        self.script("memory_pressure", "echo 'System-wide memory free percentage: 98%'")
        self.script("pgrep", "exit 1")
        for name in ["system_profiler", "pmset", "git"]:
            self.script(name, "exit 0")
        self.script("model-cli", "echo answer; echo '[stop=endOfTurn prefill=5tok/1.00s new=1tok decode=0.10s tok/s=10.000]' >&2")
        summary = self.root / "summarize-benchmarks.sh"
        summary.write_text("#!/bin/bash\nexit 0\n")
        summary.chmod(0o755)
        model = self.root / "model"
        model.mkdir()
        for name in ["manifest.json", "verified-install.json"]:
            (model / name).write_text("{}")
        prompts = self.root / "docs/benchmark-prompts/real-generation-v1"
        prompts.mkdir(parents=True)
        for name in ["short-explanation", "medium-review", "long-synthesis"]:
            (prompts / (name + ".json")).write_text("[]")
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        BENCH_CLI=str(self.bin / "model-cli"))
        for key in ["BENCH_CASES", "WARMUP_CASES", "MIN_FREE_GB", "MIN_FREE_PCT", "BENCH_SETTLE_SECONDS"]:
            self.env.pop(key, None)

    def script(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)

    def run_benchmark(self):
        return subprocess.run(["bash", str(self.root / "run-benchmark.sh"), "test", "model", "1"],
                              cwd=self.root, env=self.env, capture_output=True, text=True)

    def test_success_records_every_launch_and_refuses_overwrite(self):
        result = self.run_benchmark()
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = self.root / "benchmark-results/test"
        self.assertEqual(len(list(evidence.rglob("*.exit"))), 6)
        self.assertEqual(len(list(evidence.rglob("*.command"))), 6)
        self.assertEqual(len(list(evidence.rglob("*.preflight"))), 6)
        self.assertEqual((evidence / "exit-status").read_text().strip(), "0")
        result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)
        self.assertEqual((evidence / "exit-status").read_text().strip(), "0")

    def test_memory_rechecked_after_initial_preflight(self):
        self.script("memory_pressure", "if [ -e pressure-read ]; then echo 'System-wide memory free percentage: 1%'; else touch pressure-read; echo 'System-wide memory free percentage: 98%'; fi")
        result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("memory preflight failed", result.stderr)
        self.assertFalse(list(self.root.rglob("*.exit")))
        evidence = self.root / "benchmark-results/test"
        self.assertEqual((evidence / "exit-status").read_text().strip(), "1")
        self.assertIn("memory preflight failed", (evidence / "failure.txt").read_text())
        self.assertIn("memory_free_percent=1", next(evidence.rglob("*.preflight")).read_text())

    def test_owner_rechecked_after_initial_preflight(self):
        self.script("pgrep", "if [ -e owner-read ]; then echo '123 /test/MferenceServer'; else touch owner-read; exit 1; fi")
        result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another model owner", result.stderr)
        self.assertFalse(list(self.root.rglob("*.exit")))

    def test_empty_visible_answer_is_rejected(self):
        self.script("model-cli", "echo '   '; echo '[stop=endOfTurn prefill=5tok/1.00s new=1tok decode=0.10s tok/s=10.000]' >&2")
        result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no visible answer", result.stderr)

    def test_fixed_settling_between_runs_is_recorded(self):
        self.script("sleep", 'echo "$1" >> slept')
        self.env["BENCH_SETTLE_SECONDS"] = "10"
        result = self.run_benchmark()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "slept").read_text().splitlines(), ["10"] * 5)
        system = (self.root / "benchmark-results/test/system/system.txt").read_text()
        self.assertIn("fixed inter-process idle seconds: 10", system)

    def test_settling_never_retries_a_failed_check(self):
        self.script("sleep", 'echo "$1" >> slept')
        self.env["BENCH_SETTLE_SECONDS"] = "10"
        self.script("model-cli", "touch model-finished; echo answer; echo '[stop=endOfTurn prefill=5tok/1.00s new=1tok decode=0.10s tok/s=10.000]' >&2")
        self.script("memory_pressure", "if [ -e model-finished ]; then echo 'System-wide memory free percentage: 1%'; else echo 'System-wide memory free percentage: 98%'; fi")
        result = self.run_benchmark()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("memory preflight failed", result.stderr)
        self.assertEqual((self.root / "slept").read_text().splitlines(), ["10"])
        self.assertEqual(len(list(self.root.rglob("*.exit"))), 1)

    def test_invalid_settling_rejected_before_launch(self):
        for value in ["-1", "1.5", "61", "100", "oops"]:
            self.env["BENCH_SETTLE_SECONDS"] = value
            result = self.run_benchmark()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("BENCH_SETTLE_SECONDS", result.stderr)
            self.assertFalse(list(self.root.rglob("*.exit")))

    def test_truncation_and_process_failure_preserve_exit(self):
        for status in [0, 7]:
            with self.subTest(status=status):
                evidence = self.root / "benchmark-results"
                if evidence.exists():
                    shutil.rmtree(evidence)
                self.script("model-cli", "echo partial; echo '[stop=maxTokens prefill=5tok/1.00s new=1tok decode=0.10s tok/s=10.000]' >&2; exit " + str(status))
                result = self.run_benchmark()
                self.assertNotEqual(result.returncode, 0)
                exits = list(evidence.rglob("*.exit"))
                self.assertEqual(len(exits), 1)
                self.assertEqual(exits[0].read_text().strip(), str(status))


if __name__ == "__main__":
    unittest.main()
