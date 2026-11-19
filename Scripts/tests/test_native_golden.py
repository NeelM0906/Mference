"""Tests for the native engine's golden scripts (Scripts/native/golden).

No model, no network, no machine lock.
"""
import array
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompt_set = load("make_prompt_set", SCRIPTS / "native/golden/make_prompt_set.py")
make_slice = load("make_slice", SCRIPTS / "native/golden/make_slice.py")


class PromptSetTests(unittest.TestCase):
    def test_committed_set_is_current(self):
        self.assertEqual(prompt_set.main(["--check"]), 0)

    def test_set_shape(self):
        s = prompt_set.build()
        ids = [p["id"] for p in s["prompts"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(s["name"], "quick-v1")
        for p in s["prompts"]:
            self.assertTrue(p["messages"], p["id"])
            self.assertGreaterEqual(p["max_new"], 128, p["id"])
            self.assertIn("capture", p)
        # the long prompts of the bar: about 512, 2K, 4K and 32K tokens
        self.assertTrue({"review-512", "doc-2k", "doc-4k", "long-32k"} <= set(ids))
        # deterministic
        self.assertEqual(prompt_set.render(s), prompt_set.render(prompt_set.build()))


class SliceTests(unittest.TestCase):
    def write_run(self, root):
        d = root / "p"
        (d / "tensors").mkdir(parents=True)
        vocab = 12
        rows = [[float(i == r + 3) * 5 + i * 0.01 for i in range(vocab)] for r in range(2)]
        a = array.array("f", [v for row in rows for v in row])
        (d / "logits.f32").write_bytes(a.tobytes())
        (d / "tensors/l_out-0.f32").write_bytes(array.array("f", [1, 2, 3, 4]).tobytes())
        (d / "prompt.txt").write_text("hi", encoding="utf-8")
        m = {"format": "mfe-golden-v1", "prompt_id": "p", "n_vocab": vocab,
             "prompt_tokens": [1, 2, 3], "generated_tokens": [3, 4], "stopped_on_eog": False,
             "rendered_prompt_file": "prompt.txt", "source": {"model_path": "/x"},
             "logits": {"file": "logits.f32", "rows": 2, "cols": vocab, "position_ranges": [[2, 4]]},
             "tensors": [{"file": "tensors/l_out-0.f32", "key": "l_out-0"}]}
        (d / "golden.json").write_text(json.dumps(m), encoding="utf-8")
        return d, rows

    def test_slice_prompt(self):
        with tempfile.TemporaryDirectory() as t:
            d, rows = self.write_run(Path(t))
            s, _ = make_slice.slice_prompt(d)
            self.assertEqual(s["prompt_token_count"], 3)
            self.assertEqual(s["generated_tokens"], [3, 4])
            self.assertEqual([top["ids"][0] for top in s["logits_top"]], [3, 4])
            self.assertEqual(len(s["logits_top"][0]["ids"]), make_slice.TOP)
            row = [array.array("f", [v])[0] for v in rows[0]]
            lse = max(row) + math.log(sum(math.exp(v - max(row)) for v in row))
            self.assertAlmostEqual(s["logits_top"][0]["log_sum_exp"], lse, places=9)
            tokens = array.array("i", [1, 2, 3]).tobytes()
            self.assertEqual(s["prompt_tokens_sha256"], hashlib.sha256(tokens).hexdigest())
            self.assertEqual(s["tensor_count"], 1)
            tsha = hashlib.sha256(array.array("f", [1, 2, 3, 4]).tobytes()).hexdigest()
            line = f"tensors/l_out-0.f32 {tsha}\n"
            self.assertEqual(s["tensors_digest"], hashlib.sha256(line.encode()).hexdigest())
            self.assertEqual(set(s["files"]), {"logits.f32", "prompt.txt"})


if __name__ == "__main__":
    unittest.main()
