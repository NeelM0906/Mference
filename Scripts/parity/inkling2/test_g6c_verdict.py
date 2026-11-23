#!/usr/bin/env python3
"""Self-test for g6c_verdict.py's gating rule 2, re-based (spec v1.12): stdlib
unittest on a synthetic world (no model, no program files):
python3 -I Scripts/parity/inkling2/test_g6c_verdict.py"""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_goldens as cg  # noqa: E402
import g6c_verdict as gv  # noqa: E402
import next_token_quality as ntq  # noqa: E402

VOCAB = 25


def logprobs(p, shift):
    """Log-softmax over VOCAB ids at position p: id p % VOCAB leads by about 3
    nats; `shift` perturbs the logits (0 = the base distribution)."""
    lead = p % VOCAB
    logits = [-(abs(i - lead) % 7) * 0.5 + (3.0 if i == lead else 0.0) + shift * ((i * 7 + p) % 5 - 2)
              for i in range(VOCAB)]
    z = math.log(sum(math.exp(x) for x in logits))
    return [x - z for x in logits]


def write_run(d, pid, ids, shift=0.0, next_shift=0.0):
    """ink-golden's layout: <d>/<pid>/golden.json and logprobs.json, one row per
    position, top 20, the true next token's log-probability (offset by
    next_shift, for the quality rule)."""
    os.makedirs(os.path.join(d, pid), exist_ok=True)
    rows = []
    for p in range(len(ids)):
        lp = logprobs(p, shift)
        top = sorted(range(VOCAB), key=lambda i: (-lp[i], i))[:20]
        row = {"pos": p, "top": [[i, lp[i]] for i in top]}
        if p + 1 < len(ids):
            row["next"] = ids[p + 1]
            row["next_logprob"] = lp[ids[p + 1]] + next_shift
        rows.append(row)
    json.dump({"prompt_ids": ids, "generated_ids": [], "logprobs_file": "logprobs.json"},
              open(os.path.join(d, pid, "golden.json"), "w"))
    json.dump(rows, open(os.path.join(d, pid, "logprobs.json"), "w"))


def write_trace(d, pid, experts, margins=None):
    """trace.json with per-visit expert ids ([tokens][8]) and an all-ones hidden file."""
    os.makedirs(os.path.join(d, pid), exist_ok=True)
    nv, nt = len(experts), len(experts[0])
    visits = []
    for v, e in enumerate(experts):
        rec = {"slot": 0, "layer": v, "visit": v, "experts": e}
        if margins is not None:
            rec["margins"] = margins[v]
        visits.append(rec)
    json.dump({"hidden_shape": [nv, nt, 2], "hidden_file": "h.f32", "visits": visits},
              open(os.path.join(d, pid, "trace.json"), "w"))
    open(os.path.join(d, pid, "h.f32"), "wb").write(b"\x00\x00\x80\x3f" * (nv * nt * 2))
    json.dump({"prompt_ids": list(range(nt)), "generated_ids": []}, open(os.path.join(d, pid, "golden.json"), "w"))


def experts_with(flips, visits=4, tokens=25):
    """visits x tokens expert sets, the first `flips` token-visits with one expert changed."""
    out, k = [], 0
    for _ in range(visits):
        row = []
        for _ in range(tokens):
            s = list(range(8))
            if k < flips:
                s[-1] = 99
            row.append(s)
            k += 1
        out.append(row)
    return out


class World:
    """goldens (generate-tf/{metal,cpu}, generate/metal), K4's precision tree and a
    candidate. Prompts: SHORT (20 tokens, a trace prompt), MID (60), LONG (400,
    beyond (a)); quality on MID and LONG."""
    SHORT, MID, LONG = "t-short", "p-mid", "p-long"

    def __init__(self, tmp, cand_shift=0.0, cand_next=0.0, fp32=True, modes=True, traces=True):
        self.goldens = os.path.join(tmp, "goldens")
        self.root = os.path.join(tmp, "precision")
        self.cand = os.path.join(tmp, "cand")
        self.tdir = os.path.join(tmp, "cand-traces")
        self.ids = {self.SHORT: [(3 * i) % VOCAB for i in range(20)],
                    self.MID: [(5 * i + 1) % VOCAB for i in range(60)],
                    self.LONG: [(7 * i + 2) % VOCAB for i in range(400)]}
        for pid, ids in self.ids.items():
            for gd in ("generate-tf/metal", "generate-tf/cpu", "generate/metal"):
                write_run(os.path.join(self.goldens, gd), pid, ids)
            write_run(self.cand, pid, ids, shift=cand_shift, next_shift=cand_next)
            if fp32 and pid != self.LONG:
                write_run(os.path.join(self.root, pid, "fp32", "tf"), pid, ids)
            if modes:
                mode = "llamaCPU" if len(ids) < gv.B_SHORT else "llamaExperts"
                write_run(os.path.join(self.root, pid, mode, "tf"), pid, ids)
        if traces:
            write_trace(os.path.join(self.root, self.SHORT, "fp32", "traces"), self.SHORT, experts_with(0))
            write_trace(self.tdir, self.SHORT, experts_with(0))
        metal, cpu = (os.path.join(self.goldens, "generate-tf", m) for m in ("metal", "cpu"))
        self.meas = {pid: gv.metrics(cg.lp_metrics(cg.load(metal, pid, "logprobs.json"),
                                                  cg.load(cpu, pid, "logprobs.json"), len(ids)))
                     for pid, ids in self.ids.items()}

    def run(self):
        return gv.rebased(self.goldens, self.cand, self.tdir, self.root, self.meas, list(self.ids))


class RebasedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.saved = (gv.TRACE_GOLDENS, gv.B_REQUIRED, ntq.DEFAULT)
        gv.TRACE_GOLDENS = {World.SHORT: "generate/metal"}
        gv.B_REQUIRED = (World.SHORT, World.MID)
        ntq.DEFAULT = [World.MID, World.LONG]

    def tearDown(self):
        gv.TRACE_GOLDENS, gv.B_REQUIRED, ntq.DEFAULT = self.saved
        self._tmp.cleanup()

    def test_identical_runs_pass_with_full_coverage(self):
        rb = World(self.tmp).run()
        state, failed, missing = gv.rebased_state(rb)
        self.assertEqual((state, failed, missing), ("pass", [], []))
        # (a) covers prompts of at most 347 tokens and the trace prompts, not LONG
        self.assertEqual(sorted(rb["a"]["prompts"]), sorted([World.SHORT, World.MID]))
        self.assertNotIn(World.LONG, rb["c2"]["prompts"])
        self.assertEqual(sorted(rb["c1"]["prompts"]), sorted([World.SHORT, World.MID, World.LONG]))
        self.assertEqual(rb["a"]["prompts"][World.SHORT]["flip_margins"], "not recorded")

    def test_b_picks_the_mode_by_length(self):
        w = World(self.tmp)
        # a wrong-mode run beside each must be ignored
        write_run(os.path.join(w.root, World.SHORT, "llamaExperts", "tf"), World.SHORT, w.ids[World.SHORT], shift=0.5)
        write_run(os.path.join(w.root, World.MID, "llamaCPU", "tf"), World.MID, w.ids[World.MID], shift=0.5)
        rb = w.run()
        self.assertEqual(rb["b"]["prompts"][World.SHORT]["mode"], "llamaCPU")
        self.assertEqual(rb["b"]["prompts"][World.MID]["mode"], "llamaExperts")
        self.assertTrue(all(v["pass"] for v in rb["b"]["prompts"].values()))

    def test_runner_far_from_fp32_fails_a(self):
        rb = World(self.tmp, cand_shift=0.08).run()
        self.assertFalse(rb["a"]["prompts"][World.MID]["pass"])
        self.assertIn("mean_kl", rb["a"]["prompts"][World.MID]["fails"])
        self.assertEqual(gv.rebased_state(rb)[0], "fail")

    def test_missing_fp32_is_incomplete_not_pass(self):
        rb = World(self.tmp, fp32=False).run()
        state, failed, missing = gv.rebased_state(rb)
        self.assertEqual(state, "incomplete")
        self.assertEqual(failed, [])
        self.assertIn("a", missing)
        self.assertIn("c2", missing)

    def test_missing_required_b_is_incomplete(self):
        rb = World(self.tmp, modes=False).run()
        self.assertEqual(gv.rebased_state(rb)[:2], ("incomplete", []))
        self.assertEqual(len(rb["b"]["missing"]), 2)

    def test_a_failure_wins_over_missing_inputs(self):
        rb = World(self.tmp, cand_shift=0.08, modes=False).run()
        self.assertEqual(gv.rebased_state(rb)[0], "fail")

    def test_quality_rule(self):
        # Metal and CPU are identical (their own lower bound is 0): a runner that
        # predicts the text 0.01 nats worse fails the mean, one 0.01 better passes
        worse = World(self.tmp, cand_next=-0.01).run()
        self.assertFalse(worse["d"]["pass"])
        self.assertIn("d", gv.rebased_state(worse)[1])
        better = World(tempfile.mkdtemp(dir=self.tmp), cand_next=0.01).run()
        self.assertTrue(better["d"]["pass"])
        self.assertEqual(better["d"]["measure"], "exact")

    def test_missing_quality_prompt_is_incomplete(self):
        w = World(self.tmp)
        ntq.DEFAULT = [World.MID, World.LONG, "absent"]
        state, _, missing = gv.rebased_state(w.run())
        self.assertEqual(state, "incomplete")
        self.assertIn("d", missing)


class ExpertRuleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, "precision")
        self.tdir = os.path.join(self._tmp.name, "cand")

    def tearDown(self):
        self._tmp.cleanup()

    def judge(self, flips, margins=None):
        write_trace(os.path.join(self.root, "t", "fp32", "traces"), "t", experts_with(0), margins)
        write_trace(self.tdir, "t", experts_with(flips))
        return gv.expert_rule(self.root, "t", self.tdir)

    def test_two_percent_of_token_visits(self):
        # 4 visits x 25 tokens = 100 token-visits
        fails, ex, fm = self.judge(2)
        self.assertEqual(fails, {})
        self.assertAlmostEqual(ex, 0.98)
        self.assertIsNone(fm)
        fails, ex, _ = self.judge(3)
        self.assertIn("expert_sets", fails)

    def test_flip_margin_clause_where_recorded(self):
        small = [[0.001] * 25 for _ in range(4)]
        fails, _, fm = self.judge(1, small)
        self.assertEqual((fails, fm), ({}, [0.001]))
        large = [[0.03] + [0.001] * 24] + [[0.001] * 25 for _ in range(3)]
        fails, _, fm = self.judge(1, large)
        self.assertIn("flip_margin", fails)

    def test_missing_trace_is_none(self):
        self.assertIsNone(gv.expert_rule(self.root, "t", self.tdir))


if __name__ == "__main__":
    unittest.main()
