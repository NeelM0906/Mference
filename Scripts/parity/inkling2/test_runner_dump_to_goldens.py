#!/usr/bin/env python3
"""Self-test for runner_dump_to_goldens.py (stdlib unittest, no model, no
program files): python3 -I Scripts/parity/inkling2/test_runner_dump_to_goldens.py"""

import array
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner_dump_to_goldens as rd  # noqa: E402

V = rd.VOCAB


def write_f32(path, values):
    a = array.array("f", values)
    if sys.byteorder != "little":
        a.byteswap()
    open(path, "wb").write(a.tobytes())


def row(peaks, base=-20.0, length=V):
    r = [base] * length
    for i, v in peaks.items():
        r[i] = v
    return r


class Convert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dump = os.path.join(self.tmp.name, "dump")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(self.dump)

    def tearDown(self):
        self.tmp.cleanup()

    def prompt(self, pid, rows, positions, ids, row_len=V, softcapped=True, **extra):
        write_f32(os.path.join(self.dump, pid + ".logits.f32"), [v for r in rows for v in r])
        d = {"prompt_ids": ids, "positions": positions, "logits_file": pid + ".logits.f32",
             "logits_row_length": row_len, "logits_softcapped": softcapped}
        d.update(extra)
        json.dump(d, open(os.path.join(self.dump, pid + ".json"), "w"))

    def test_logprobs_match_a_direct_log_softmax(self):
        r0 = row({5: 3.0, 9: 1.0, 2: 1.0})  # ids 2 and 9 tie: lower id first
        r1 = row({7: 2.5})
        self.prompt("p", [r0, r1], [0, 1], [11, 22])
        rd.convert(self.dump, self.out, None)
        rows = json.load(open(os.path.join(self.out, "p", "logprobs.json")))
        lse = math.log(math.exp(3) + 2 * math.exp(1) + (V - 3) * math.exp(-20))
        self.assertEqual([t[0] for t in rows[0]["top"][:3]], [5, 2, 9])
        self.assertAlmostEqual(rows[0]["top"][0][1], 3 - lse, places=5)
        self.assertEqual(len(rows[0]["top"]), 20)
        self.assertEqual(rows[0]["next"], 22)
        self.assertAlmostEqual(rows[0]["next_logprob"], -20 - lse, places=4)
        g = json.load(open(os.path.join(self.out, "p", "golden.json")))
        self.assertEqual(g["generated_ids"], [7])  # argmax of the last position
        self.assertEqual(g["logprobs_file"], "logprobs.json")
        self.assertEqual(json.load(open(os.path.join(self.out, "index.json")))["prompts"], [{"id": "p"}])

    def test_softcap_and_padded_rows(self):
        pre = row({4: 45.0, 6: 30.0}, base=-50.0, length=V + 966)
        pre[V + 3] = 1e9  # padding beyond the real vocabulary is ignored
        self.prompt("q", [pre], [0], [1], row_len=V + 966, softcapped=False)
        rd.convert(self.dump, self.out, None)
        top = json.load(open(os.path.join(self.out, "q", "logprobs.json")))[0]["top"]
        c = lambda x: 30 * math.tanh(x / 30)
        lse = math.log(math.exp(c(45)) + math.exp(c(30)) + (V - 2) * math.exp(c(-50)))
        self.assertEqual(top[0][0], 4)
        self.assertAlmostEqual(top[0][1], c(45) - lse, places=4)
        self.assertAlmostEqual(top[1][1], c(30) - lse, places=4)

    def test_refusals(self):
        self.prompt("bad", [row({})], [0], [1])
        os.truncate(os.path.join(self.dump, "bad.logits.f32"), 4 * V - 4)
        with self.assertRaisesRegex(ValueError, "bytes"):
            rd.convert(self.dump, self.out, None)
        self.prompt("bad", [row({3: float("nan")})], [0], [1])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rd.convert(self.dump, self.out, None)
        self.prompt("bad", [row({})], [1], [1])
        with self.assertRaisesRegex(ValueError, "positions"):
            rd.convert(self.dump, self.out, None)

    def test_trace(self):
        n = 2
        write_f32(os.path.join(self.dump, "t.hidden.f32"), [0.5] * (rd.VISITS * n * rd.HIDDEN))
        ex = [[[1, 2, 3, 4, 5, 6, 7, 8]] * n for _ in range(rd.VISITS)]
        json.dump({"prompt_ids": [3, 4], "hidden_file": "t.hidden.f32", "experts": ex},
                  open(os.path.join(self.dump, "t.json"), "w"))
        rd.convert(self.dump, self.out, None)
        t = json.load(open(os.path.join(self.out, "t", "trace.json")))
        self.assertEqual(t["hidden_shape"], [rd.VISITS, n, rd.HIDDEN])
        self.assertEqual(len(t["visits"]), rd.VISITS)
        self.assertEqual(os.path.getsize(os.path.join(self.out, "t", "hidden_after_visit.f32")),
                         4 * rd.VISITS * n * rd.HIDDEN)
        json.dump({"prompt_ids": [3, 4, 5], "hidden_file": "t.hidden.f32", "experts": ex},
                  open(os.path.join(self.dump, "t.json"), "w"))
        with self.assertRaisesRegex(ValueError, "hidden file"):
            rd.convert(self.dump, self.out, None)


if __name__ == "__main__":
    unittest.main()
