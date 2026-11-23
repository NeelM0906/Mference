#!/usr/bin/env python3
"""Self-test for g6c_verdict.py's gating rule 2, re-based (spec v1.12): stdlib
unittest on a synthetic world (no model, no program files):
python3 -I Scripts/parity/inkling2/test_g6c_verdict.py"""

import contextlib
import io
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


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


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
    dump({"prompt_ids": ids, "generated_ids": [], "logprobs_file": "logprobs.json"}, os.path.join(d, pid, "golden.json"))
    dump(rows, os.path.join(d, pid, "logprobs.json"))


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
    dump({"hidden_shape": [nv, nt, 2], "hidden_file": "h.f32", "visits": visits}, os.path.join(d, pid, "trace.json"))
    with open(os.path.join(d, pid, "h.f32"), "wb") as f:
        f.write(b"\x00\x00\x80\x3f" * (nv * nt * 2))
    dump({"prompt_ids": list(range(nt)), "generated_ids": []}, os.path.join(d, pid, "golden.json"))


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
        self.assertEqual(rb["a"]["prompts"][World.SHORT]["flip_margins"], {"flips": 0, "source": "no flips", "max": None})
        self.assertEqual(rb["a"]["rederivation_loosens"], [])

    def test_flips_without_margins_make_a_incomplete(self):
        w = World(self.tmp)
        write_trace(w.tdir, World.SHORT, experts_with(1))
        rb = w.run()
        self.assertEqual(rb["a"]["prompts"][World.SHORT]["flip_margins"], "not checkable")
        self.assertEqual(gv.rebased_state(rb)[:2], ("incomplete", []))
        # with the reference's margins recorded, the same flip is judged
        write_trace(os.path.join(w.root, World.SHORT, "fp32", "traces"), World.SHORT, experts_with(0),
                    [[0.001] * 25 for _ in range(4)])
        rb = w.run()
        self.assertEqual(gv.rebased_state(rb)[0], "pass")
        self.assertEqual(rb["a"]["prompts"][World.SHORT]["flip_margins"]["source"], "fp32")

    def test_expert_sets_checked_on_covered_prompts_with_traces(self):
        w = World(self.tmp)
        write_trace(os.path.join(w.root, World.MID, "fp32", "traces"), World.MID, experts_with(0))
        write_trace(w.tdir, World.MID, experts_with(5))
        rb = w.run()
        self.assertIn("expert_sets", rb["a"]["prompts"][World.MID]["fails"])
        self.assertEqual(gv.rebased_state(rb)[0], "fail")

    def test_rederivation_is_reported_never_gated(self):
        rb = World(self.tmp, cand_shift=0.01).run()
        worst = rb["a"]["measured_worst"]
        expect = [k for k, b in gv.EXACT_BOUNDS.items() if 3 * worst[k] > b]
        self.assertEqual(rb["a"]["rederivation_loosens"], expect)
        for k, b in gv.EXACT_BOUNDS.items():
            self.assertEqual(rb["a"]["rederived"][k], max(3 * worst[k], b))

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
        self.assertTrue(rb["c1"]["prompts"][World.LONG]["pass"])  # within 2x the addendum bounds
        self.assertFalse(rb["c2"]["prompts"][World.MID]["pass"])   # but farther from tf CPU than fp32

    def test_runner_far_from_tf_cpu_fails_c1(self):
        rb = World(self.tmp, cand_shift=0.5).run()
        self.assertFalse(rb["c1"]["prompts"][World.LONG]["pass"])
        self.assertIn("c1", gv.rebased_state(rb)[1])
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
        rb = w.run()
        state, _, missing = gv.rebased_state(rb)
        self.assertEqual(state, "incomplete")
        self.assertIn("d", missing)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            gv.print_rebased(rb)
        line = next(l for l in out.getvalue().splitlines() if l.startswith("(d)"))
        self.assertIn("incomplete (missing absent", line)
        self.assertNotIn("FAIL", line)


    def test_truncated_runner_output_fails(self):
        w = World(self.tmp)
        for pid in w.ids:
            path = os.path.join(w.cand, pid, "logprobs.json")
            with open(path) as f:
                rows = json.load(f)
            dump(rows[:2], path)
        rb = w.run()
        self.assertIn("positions", rb["a"]["prompts"][World.MID]["fails"])
        self.assertIn("positions", rb["c1"]["prompts"][World.LONG]["fails"])
        self.assertFalse(rb["d"]["pass"])
        self.assertEqual(gv.rebased_state(rb)[0], "fail")

    def test_b_fails_when_the_rounded_reference_drifts(self):
        w = World(self.tmp)
        write_run(os.path.join(w.root, World.MID, "llamaExperts", "tf"), World.MID, w.ids[World.MID], shift=0.3)
        rb = w.run()
        self.assertFalse(rb["b"]["prompts"][World.MID]["pass"])
        self.assertIn("b", gv.rebased_state(rb)[1])

    def test_a_covers_by_length(self):
        gv.A_MAX_TOKENS, saved = 59, gv.A_MAX_TOKENS
        try:
            rb = World(self.tmp).run()
        finally:
            gv.A_MAX_TOKENS = saved
        self.assertNotIn(World.MID, rb["a"]["prompts"])  # 60 tokens > 59
        self.assertIn(World.SHORT, rb["a"]["prompts"])

    def test_mismatched_trace_shape_fails_without_crashing(self):
        w = World(self.tmp)
        write_trace(w.tdir, World.SHORT, experts_with(0, visits=3))
        rb = w.run()
        self.assertIn("trace_shape", rb["a"]["prompts"][World.SHORT]["fails"])
        self.assertEqual(gv.rebased_state(rb)[0], "fail")


    def test_runner_without_next_logprob_fails_d(self):
        w = World(self.tmp)
        for pid in (World.MID, World.LONG):
            path = os.path.join(w.cand, pid, "logprobs.json")
            with open(path) as f:
                rows = json.load(f)
            for r in rows:
                r.pop("next_logprob", None)
            dump(rows, path)
        rb = w.run()
        self.assertEqual(rb["d"]["measure"], "top20")
        self.assertFalse(rb["d"]["pass"])
        self.assertTrue(any("runner gives next_logprob" in x for x in rb["d"]["incomplete_positions"]))

    def test_truncated_fp32_fails_a(self):
        w = World(self.tmp)
        path = os.path.join(w.root, World.MID, "fp32", "tf", World.MID, "logprobs.json")
        with open(path) as f:
            rows = json.load(f)
        dump(rows[:10], path)
        rb = w.run()
        self.assertIn("reference_positions", rb["a"]["prompts"][World.MID]["fails"])
        self.assertIn("fp32_positions", rb["c2"]["prompts"][World.MID]["fails"])

    def test_trace_without_shape_fails_without_crashing(self):
        w = World(self.tmp)
        for d in (os.path.join(w.root, World.SHORT, "fp32", "traces"), w.tdir):
            path = os.path.join(d, World.SHORT, "trace.json")
            with open(path) as f:
                t = json.load(f)
            del t["hidden_shape"]
            dump(t, path)
        rb = w.run()
        self.assertIn("trace_shape", rb["a"]["prompts"][World.SHORT]["fails"])


class VerdictTests(unittest.TestCase):
    def test_exit_codes(self):
        self.assertEqual(gv.verdict(False, "incomplete", ["a"], True)[1], 1)  # a failure wins
        self.assertEqual(gv.verdict(True, "incomplete", ["a"], True)[1], 4)
        self.assertEqual(gv.verdict(True, "pass", [], False)[1], 4)          # no traces: never pass
        self.assertEqual(gv.verdict(True, "pass", [], True), ("PASS", 0))

    def test_rule3_needs_every_reference_position(self):
        tmp = tempfile.mkdtemp()
        try:
            w = World(tmp)
            metal = os.path.join(w.goldens, "generate-tf", "metal")
            recs = {pid: {"logprobs": gv.pair(metal, w.cand, pid)} for pid in w.ids}
            ok = gv.rule3(recs, w.meas, list(w.ids))
            self.assertTrue(ok["pass"])
            self.assertNotIn("incomplete", ok)
            # one prompt with fewer positions compared than the reference has
            path = os.path.join(w.cand, World.MID, "logprobs.json")
            with open(path) as f:
                rows = json.load(f)
            dump(rows[:-5], path)
            recs[World.MID] = {"logprobs": gv.pair(metal, w.cand, World.MID)}
            short = gv.rule3(recs, w.meas, list(w.ids))
            self.assertFalse(short["pass"])
            self.assertEqual(short["incomplete"], [World.MID])
            # and one missing altogether
            del recs[World.LONG]
            self.assertEqual(sorted(gv.rule3(recs, w.meas, list(w.ids))["incomplete"]), [World.LONG, World.MID])
        finally:
            import shutil
            shutil.rmtree(tmp)

    def test_pooled_skips_prompts_without_positions(self):
        empty = {"logprobs": {"positions": 0, "top1_agreement": None, "top1_at_margin": {"2.0": {"positions": 0, "agree": 0}}}}
        self.assertFalse(gv.pooled([empty])["pass"])


class ForcedReportTests(unittest.TestCase):
    """Option (i), report only: K4's inkling2-forced-routing-v1 files."""
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, "precision")
        self.tdir = os.path.join(self._tmp.name, "cand-x", "traces")
        write_trace(self.tdir, "t", experts_with(0, visits=2, tokens=3))

    def tearDown(self):
        self._tmp.cleanup()

    def forced(self, **override):
        import hashlib
        with open(os.path.join(self.tdir, "t", "trace.json"), "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        visits = [{"visit": v, "layer": v, "slot": 0, "own_experts": [list(range(8))] * 3, "fp32_margins": [0.01] * 3,
                   "shortfalls": [0.0, 0.004, 0.0] if v == 1 else [0.0] * 3, "differs": [False, v == 1, False],
                   "increment_rel_l2": [0.001] * 3} for v in range(2)]
        d = {"schema": "inkling2-forced-routing-v1", "id": "t", "candidate": "cand-x", "runner_tip": "abc",
             "runner_trace_sha256": sha, "reference_commit": "def", "n_tokens": 3, "top_k": 8,
             "near_tie_bound": 0.0258, "token_visits": 6, "disagreement_count": 1, "disagreement_rate": 1 / 6,
             "not_near_tie_count": 0, "visits": visits,
             "disagreements": [{"visit": 1, "layer": 1, "pos": 1, "shortfall": 0.004, "fp32_margin": 0.002,
                                "runner_margin": None, "runner_only": [9], "fp32_only": [7], "near_tie": True}]}
        d.update(override)
        os.makedirs(os.path.join(self.root, "t", "forced"), exist_ok=True)
        dump(d, os.path.join(self.root, "t", "forced", "t.json"))

    def test_ok(self):
        self.forced()
        r = gv.forced_report(self.root, "t", self.tdir)
        self.assertEqual((r["status"], r["disagreements"], r["not_near_tie"], r["per_layer"]), ("ok", 1, 0, {1: 1}))
        self.assertAlmostEqual(r["max_shortfall"], 0.004)

    def test_stale_and_malformed(self):
        self.forced(candidate="cand-other")  # the forced-on candidate must be on disk to verify the hidden file (v1.18)
        self.assertEqual(gv.forced_report(self.root, "t", self.tdir)["status"], "stale")
        self.forced(runner_trace_sha256="0" * 64)
        self.assertEqual(gv.forced_report(self.root, "t", self.tdir)["status"], "stale")
        self.forced(disagreement_count=2)
        self.assertEqual(gv.forced_report(self.root, "t", self.tdir)["status"], "malformed")
        self.forced(not_near_tie_count=1)
        self.assertEqual(gv.forced_report(self.root, "t", self.tdir)["status"], "malformed")
        self.forced(token_visits=7)
        self.assertEqual(gv.forced_report(self.root, "t", self.tdir)["status"], "malformed")
        self.forced(precision="llamaExperts")
        r = gv.forced_report(self.root, "t", self.tdir)
        self.assertEqual(r["status"], "malformed")
        self.assertIn("not fp32", r["why"])
        self.forced(precision="fp32")
        self.assertEqual(gv.forced_report(self.root, "t", self.tdir)["status"], "ok")

    def test_absent_is_none(self):
        self.assertIsNone(gv.forced_report(self.root, "t", self.tdir))


class V116Tests(unittest.TestCase):
    """The closing form: (a) as the forced per-visit view on four prompts, the reference against llama.cpp."""
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, "precision")
        self.tdir = os.path.join(self._tmp.name, "cand-x", "traces")
        self.saved = (gv.FORCED_PROMPTS, gv.FORCED_TOTAL)
        gv.FORCED_PROMPTS = ("t",)

    def tearDown(self):
        gv.FORCED_PROMPTS, gv.FORCED_TOTAL = self.saved
        self._tmp.cleanup()

    def forced(self, pid="t", where=None, shortfall=0.0004, inc=0.001, precision=None, positions=3, extra=None,
               runner_margin=None):
        """A forced file over a runner trace of 2 visits x `positions` tokens (written to match)."""
        import hashlib
        n = positions
        write_trace(self.tdir, "t", experts_with(0, visits=2, tokens=n))
        gv.FORCED_TOTAL = 2 * n
        with open(os.path.join(self.tdir, "t", "trace.json"), "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        visits = [{"visit": v, "layer": v, "slot": 0, "own_experts": [list(range(8))] * n, "fp32_margins": [0.01] * n,
                   "shortfalls": [0.0] * n, "differs": [v == 1 and t == 1 for t in range(n)],
                   "increment_rel_l2": [inc if not callable(inc) else inc(t) for t in range(n)]} for v in range(2)]
        d = {"schema": "inkling2-forced-routing-v1", "id": pid, "candidate": "cand-x", "runner_tip": "abc",
             "runner_trace_sha256": sha, "reference_commit": "def", "precision": precision, "n_tokens": n,
             "top_k": 8, "near_tie_bound": 0.0258, "token_visits": 2 * n, "disagreement_count": 1,
             "disagreement_rate": 1 / (2 * n), "not_near_tie_count": 0, "visits": visits,
             "disagreements": [{"visit": 1, "layer": 1, "pos": 1, "shortfall": shortfall, "fp32_margin": shortfall / 2,
                                "runner_margin": runner_margin, "runner_only": [9], "fp32_only": [7], "near_tie": True}]}
        d.update(extra or {})
        path = where or os.path.join(self.root, pid, "forced", f"{pid}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(d, path)

    def test_a_forced_bounds(self):
        self.forced()  # 1 of 6 token-visits is far above 0.5%
        a = gv.forced_gate(self.root, self.tdir)
        self.assertIn("rate", a["prompts"]["t"]["fails"])
        self.forced(positions=300)  # 1 of 600
        self.assertTrue(gv.forced_gate(self.root, self.tdir)["prompts"]["t"]["pass"])
        self.forced(positions=300, shortfall=0.002)
        self.assertIn("shortfall", gv.forced_gate(self.root, self.tdir)["prompts"]["t"]["fails"])
        self.forced(positions=300, inc=0.02)
        self.assertIn("increment", gv.forced_gate(self.root, self.tdir)["prompts"]["t"]["fails"])

    def test_a_stale_wrong_or_short_files_are_incomplete(self):
        for extra in ({"runner_trace_sha256": "0" * 64}, {"runner_trace_sha256": None}, {"id": "other"}):
            self.forced(positions=300, extra=extra)
            a = gv.forced_gate(self.root, self.tdir)
            self.assertIsNone(a["prompts"]["t"]["pass"], extra)
            self.assertTrue(a["missing"], extra)
        self.forced(positions=300)
        write_trace(self.tdir, "t", experts_with(0, visits=2, tokens=5))  # the runner's trace is not the forced one
        self.assertIsNone(gv.forced_gate(self.root, self.tdir)["prompts"]["t"]["pass"])
        self.forced(positions=300)
        gv.FORCED_TOTAL = 18816  # the four files must cover the measured total
        self.assertTrue(any("not 18816" in m for m in gv.forced_gate(self.root, self.tdir)["missing"]))

    def test_trace_by_content(self):
        """v1.18: the forced-on trace sits beside the candidate; key order only passes, content differences are stale."""
        import shutil
        self.forced(positions=300)  # forced on cand-x's trace as written; now move that trace to cand-orig
        orig = os.path.join(self._tmp.name, "cand-orig", "traces")
        shutil.copytree(self.tdir, orig)
        path = os.path.join(self.root, "t", "forced", "t.json")
        with open(path) as f:
            d = json.load(f)
        d["candidate"] = "cand-orig"
        dump(d, path)
        tj = os.path.join(self.tdir, "t", "trace.json")
        with open(tj) as f:
            t = json.load(f)
        def write(obj):
            with open(tj, "w") as f:
                f.write(json.dumps(obj)); f.flush()
        # 1. key order only: passes, with the note
        write({k: t[k] for k in reversed(list(t))})
        r = gv.forced_report(self.root, "t", self.tdir)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["trace_note"], "raw bytes differ, key order only")
        # 2. one expert id differs: stale
        t2 = json.loads(json.dumps(t))
        t2["visits"][0]["experts"][0][0] = 99
        write(t2)
        r = gv.forced_report(self.root, "t", self.tdir)
        self.assertEqual(r["status"], "stale")
        self.assertIn("content differs", r["why"])
        # 3. the hidden-state file differs: stale
        write(t)
        with open(os.path.join(self.tdir, "t", "h.f32"), "r+b") as f:
            f.write(b"\x00\x00\x00\x40")
        r = gv.forced_report(self.root, "t", self.tdir)
        self.assertEqual(r["status"], "stale")
        self.assertIn("hidden-state file differs", r["why"])
        # 4. a margin's sign flips to -0.0 from 0.0: content, not formatting
        with open(os.path.join(self.tdir, "t", "h.f32"), "r+b") as f:
            f.write(b"\x00\x00\x80\x3f")
        t3 = json.loads(json.dumps(t))
        t3["visits"][0]["slot"] = -0.0 if t3["visits"][0]["slot"] == 0 else t3["visits"][0]["slot"]
        write(t3)
        self.assertEqual(gv.forced_report(self.root, "t", self.tdir)["status"], "stale")
        # 5. the forced-on candidate is gone: the hidden file cannot be verified, even with trace.json identical
        write(t)
        shutil.rmtree(orig)
        r = gv.forced_report(self.root, "t", self.tdir)
        self.assertEqual(r["status"], "stale")
        self.assertIn("not on disk", r["why"])

    def ref(self, **kw):
        kw.setdefault("positions", 300)
        self.forced(where=os.path.join(self.root, gv.REF_LLAMA), precision=kw.pop("precision", "llamaExperts"), **kw)
        return gv.reference_check(self.root)

    def test_reference_check(self):
        self.assertEqual(gv.reference_check(self.root)["status"], "pending")
        self.assertTrue(self.ref(shortfall=0.01)["pass"])
        self.assertIn("margin", self.ref(shortfall=0.06)["fails"])  # .llamaExperts margin 0.03
        self.assertIn("margin", self.ref(shortfall=0.01, runner_margin=0.5)["fails"])  # llama.cpp's own margin
        self.assertFalse(self.ref(shortfall=0.01, runner_margin=0.001)["fails"])
        grow = self.ref(shortfall=0.01, inc=lambda t: 0.0002 * t)
        self.assertFalse(grow["pass"])
        self.assertFalse(grow["flat_miss"])
        # one spike at position 0 must not hide growth after it
        spike = self.ref(shortfall=0.01, inc=lambda t: 0.05 if t == 0 else 0.001 + 0.0002 * t)
        self.assertFalse(spike["flat_miss"])
        self.assertFalse(spike["pass"])
        flat = self.ref(shortfall=0.01, inc=0.03)
        self.assertTrue(flat["flat_miss"])
        # past position 400 still counts
        far = self.ref(shortfall=0.01, positions=450, inc=lambda t: 0.5 if t == 420 else 0.001)
        self.assertFalse(far["pass"])
        # one visit grows while the others stay flat above 0.02: growth, not flat
        self.forced(where=os.path.join(self.root, gv.REF_LLAMA), precision="llamaExperts", positions=300, shortfall=0.01,
                    inc=0.05, extra={})
        import json as _j
        refp = os.path.join(self.root, gv.REF_LLAMA)
        with open(refp) as f:
            d = _j.load(f)
        d["visits"][1]["increment_rel_l2"] = [0.05 + 0.0015 * t for t in range(300)]
        dump(d, refp)
        one = gv.reference_check(self.root)
        self.assertFalse(one["flat_miss"])
        self.assertFalse(one["pass"])
        # a NaN shortfall row and scalar extras
        d["visits"][1]["increment_rel_l2"] = [0.001] * 300
        d["visits"][0]["element_rel_max"] = 0.001
        dump(d, refp)
        self.assertEqual(gv.reference_check(self.root)["status"], "ok")
        with open(refp) as f:
            raw = f.read()
        open(refp, "w").write(raw.replace('"shortfall": 0.01', '"shortfall": NaN', 1))
        self.assertEqual(gv.reference_check(self.root)["status"], "malformed")
        self.assertEqual(self.ref(precision="llamaCPU")["status"], "malformed")
        self.assertEqual(self.ref(positions=20)["status"], "malformed")
        self.assertEqual(self.ref(extra={"visits": [], "token_visits": 0})["status"], "malformed")

    def test_state(self):
        b_ok = {"prompts": {"x": {"pass": True, "fails": {}}}, "missing": []}
        d_ok = {"missing": [], "positions": 10, "pass": True}
        a_ok = {"prompts": {"t": {"pass": True}}, "missing": []}
        ref_ok = {"status": "ok", "pass": True, "flat_miss": False}
        v = {"a": a_ok, "reference": ref_ok, "b": b_ok, "d": d_ok, "unmeasured": []}
        self.assertEqual(gv.v116_state(v)[0], "pass")
        self.assertEqual(gv.v116_state(dict(v, unmeasured=["g06-chat-none"]))[0], "incomplete")
        self.assertEqual(gv.v116_state(dict(v, reference={"status": "pending"}))[0], "incomplete")
        self.assertEqual(gv.v116_state(dict(v, reference={"status": "ok", "pass": False, "flat_miss": True}))[0], "incomplete")
        self.assertEqual(gv.v116_state(dict(v, reference={"status": "ok", "pass": False, "flat_miss": False}))[0], "fail")
        self.assertEqual(gv.v116_state(dict(v, a={"prompts": {"t": {"pass": False}}, "missing": []}))[0], "fail")
        self.assertEqual(gv.v116_state(dict(v, d=dict(d_ok, missing=["lp06"])))[0], "incomplete")


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
        # 4 visits x 25 tokens = 100 token-visits; no margins anywhere
        fails, ex, fm, src = self.judge(2)
        self.assertEqual(fails, {})
        self.assertAlmostEqual(ex, 0.98)
        self.assertEqual((fm, src), (None, None))  # flips, nothing to judge them by
        fails, ex, _, _ = self.judge(3)
        self.assertIn("expert_sets", fails)
        self.assertEqual(self.judge(0)[2:], ([], "no flips"))

    def test_flip_margin_clause(self):
        small = [[0.001] * 25 for _ in range(4)]
        fails, _, fm, src = self.judge(1, small)
        self.assertEqual((fails, fm, src), ({}, [0.001], "fp32"))
        large = [[0.03] + [0.001] * 24] + [[0.001] * 25 for _ in range(3)]
        fails, _, fm, _ = self.judge(1, large)
        self.assertIn("flip_margin", fails)

    def test_runner_margins_when_fp32_has_none(self):
        write_trace(os.path.join(self.root, "t", "fp32", "traces"), "t", experts_with(0))
        write_trace(self.tdir, "t", experts_with(1), [[0.002] * 25 for _ in range(4)])
        fails, _, fm, src = gv.expert_rule(self.root, "t", self.tdir)
        self.assertEqual((fails, fm, src), ({}, [0.002], "runner"))

    def test_missing_trace_is_none(self):
        self.assertIsNone(gv.expert_rule(self.root, "t", self.tdir))


if __name__ == "__main__":
    unittest.main()
