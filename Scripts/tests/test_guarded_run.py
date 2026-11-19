"""Tests for Scripts/native/guarded-run.sh: the power and swap-guard policy.

No model, no lock: LOCK is /usr/bin/true, pmset and the process check's pgrep
are stubs on PATH, and the guarded command is /usr/bin/true.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "native/guarded-run.sh"


class GuardedRunTests(unittest.TestCase):
    def run_script(self, args, power, extra_env=None):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "pmset"
        stub.write_text(f"#!/bin/sh\necho \"Now drawing from '{power}'\"\n", encoding="utf-8")
        stub.chmod(0o755)
        # the "other model process" checks (pgrep -lx, -lf) see none, whatever
        # runs on this machine; the wrapper's child lookup uses the real pgrep
        pgrep = bin_dir / "pgrep"
        pgrep.write_text("#!/bin/sh\ncase \"$1\" in -lx|-lf) exit 1 ;; esac\nexec /usr/bin/pgrep \"$@\"\n",
                         encoding="utf-8")
        pgrep.chmod(0o755)
        env = dict(os.environ, LOCK="/usr/bin/true", PATH=f"{bin_dir}:{os.environ['PATH']}")
        env.pop("SWAP_GUARD_MB", None)
        env.update(extra_env or {})
        log = tmp / "log"
        gguf = tmp / "model.gguf"
        gguf.write_bytes(b"GGUF")
        proc = subprocess.run(["bash", str(SCRIPT), *args, str(log), str(gguf), "--", "/usr/bin/true"],
                              env=env, capture_output=True, text=True, timeout=120)
        return proc, log

    def test_timing_run_refuses_on_battery(self):
        proc, log = self.run_script(["--timing"], "Battery Power")
        self.assertEqual(proc.returncode, 71, proc.stderr)
        self.assertIn("REFUSED: timing run not on AC power", (log / "run.log").read_text())
        self.assertFalse((log / "run.json").exists())

    def test_correctness_run_records_battery(self):
        proc, log = self.run_script([], "Battery Power")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        run = json.loads((log / "run.json").read_text())
        self.assertEqual(run["power_source"], "Battery Power")
        self.assertFalse(run["timing_run"])
        self.assertEqual(run["exit"], 0)

    def test_timing_run_on_ac(self):
        proc, log = self.run_script(["--timing"], "AC Power")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        run = json.loads((log / "run.json").read_text())
        self.assertEqual(run["power_source"], "AC Power")
        self.assertTrue(run["timing_run"])

    def test_looser_swap_guard_is_refused(self):
        proc, _ = self.run_script([], "AC Power", {"SWAP_GUARD_MB": "64"})
        self.assertEqual(proc.returncode, 71)
        self.assertIn("SWAP_GUARD_MB must be 0", proc.stderr)


if __name__ == "__main__":
    unittest.main()
