#!/usr/bin/env python3
"""Convert a native runner's raw dump into the ink-golden layout that
g6c_verdict.py reads (stdlib only).

    python3 -I Scripts/parity/inkling2/runner_dump_to_goldens.py <dump-dir> <out-dir> [--goldens <dir>]

The dump is one JSON file per prompt, `<dump-dir>/<id>.json`, next to its
binary files:

    {
      "prompt_ids": [int, ...],          the ids the runner prefilled (no BOS)
      "positions": [int, ...],           the positions whose logits are dumped,
                                         ascending; position p is the
                                         distribution of token p + 1
      "logits_file": "<id>.logits.f32",  F32 little-endian, len(positions) rows
                                         of logits_row_length values each
      "logits_row_length": 201024,       200,058 (real vocabulary) or the
                                         padded 201,024; only the first 200,058
                                         are used
      "logits_softcapped": false,        true if the final softcap 30*tanh(x/30)
                                         is already applied; false applies it
      "hidden_file": "<id>.hidden.f32",  traces only: F32 LE [48 x n x 2304], the
                                         running block sum after each visit in
                                         visit order (as l_out-<slot>)
      "experts": [[[int x 8] x n] x 48]  traces only: routed expert ids per visit
                                         and position (any order)
    }

For each prompt it writes `<out>/<id>/golden.json` and, when logits were
dumped, `<out>/<id>/logprobs.json` with ink-golden's semantics: log-softmax
over the first 200,058 logits after the softcap, the top 20 by logit (ties:
lower id first), each row {"pos", "top": [[id, logprob], ...], "next",
"next_logprob"} where "next" is the prompt's next id (-1 at the last prompt
position, where ink-golden records its own generated token instead;
g6c_verdict.py reads neither field); `generated_ids` is the argmax of the
last dumped row when that row is the last prompt position. For a
trace it writes `trace.json` (`hidden_file`, `hidden_shape`, `visits[v]`
with `visit` and `experts`) and copies the hidden states. It also writes
`index.json`. With `--goldens` (default: the program's goldens directory)
it checks every prompt against the reference it will be judged on: ids equal
to `generate-tf/metal`'s (log-prob prompts) or the Metal trace golden's
(traces), and positions covering the reference's, and refuses otherwise.
"""

import array
import heapq
import json
import math
import os
import shutil
import subprocess
import sys

VOCAB = 200058
TOP = 20
SOFTCAP = 30.0
HIDDEN = 2304
VISITS = 48
TRACE_GOLDENS = {"trace01": "generate/metal", "trace02": "generate2/metal", "trace03-tool-result": "generate3/metal"}


def ink_dir():
    if os.environ.get("INK"):
        return os.environ["INK"]
    here = os.path.dirname(os.path.abspath(__file__))
    common = subprocess.run(["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return os.path.join(os.path.dirname(common), "scratch", "inkling-program")


def logprob_rows(path, positions, row_len, softcapped, ids):
    """ink-golden's record_lp for each dumped row: the top 20 and the
    log-probability of the prompt's next token (-1 and 0.0 at the last)."""
    if row_len < VOCAB:
        raise ValueError(f"logits_row_length {row_len} < {VOCAB}")
    size = os.path.getsize(path)
    if size != 4 * row_len * len(positions):
        raise ValueError(f"{path}: {size} bytes, expected {4 * row_len * len(positions)}")
    rows = []
    with open(path, "rb") as f:
        for p in positions:
            a = array.array("f")
            a.frombytes(f.read(4 * row_len))
            if sys.byteorder != "little":
                a.byteswap()
            x = a[:VOCAB] if softcapped else [SOFTCAP * math.tanh(v / SOFTCAP) for v in a[:VOCAB]]
            if not all(map(math.isfinite, x)):
                raise ValueError(f"{path}: non-finite logit at position {p}")
            mx = max(x)
            lse = mx + math.log(math.fsum(math.exp(v - mx) for v in x))
            top = heapq.nsmallest(TOP, range(VOCAB), key=lambda i: (-x[i], i))
            nxt = ids[p + 1] if p + 1 < len(ids) else -1
            if nxt >= VOCAB:
                raise ValueError(f"{path}: prompt id {nxt} outside the {VOCAB}-token vocabulary")
            rows.append({"pos": p, "top": [[i, x[i] - lse] for i in top], "next": nxt,
                         "next_logprob": x[nxt] - lse if nxt >= 0 else 0.0})
    return rows


def check_against_reference(goldens, pid, d, out_rows):
    if pid in TRACE_GOLDENS:
        g = json.load(open(os.path.join(goldens, TRACE_GOLDENS[pid], pid, "golden.json")))
        if d["prompt_ids"] != g["prompt_ids"]:
            raise ValueError(f"{pid}: prompt ids differ from the Metal trace golden's")
    ref = os.path.join(goldens, "generate-tf", "metal", pid)
    if os.path.isdir(ref):
        g = json.load(open(os.path.join(ref, "golden.json")))
        if d["prompt_ids"] != g["prompt_ids"]:
            raise ValueError(f"{pid}: prompt ids differ from generate-tf/metal's (the teacher-forced ids)")
        want = [r["pos"] for r in json.load(open(os.path.join(ref, "logprobs.json")))]
        got = {r["pos"] for r in out_rows}
        missing = [p for p in want if p not in got]
        if missing:
            raise ValueError(f"{pid}: {len(missing)} reference positions not dumped (first {missing[0]})")
    elif pid not in TRACE_GOLDENS:
        raise ValueError(f"{pid}: not a G6c prompt (no generate-tf/metal/{pid}, not a trace)")


def convert(dump, out, goldens):
    names = sorted(f[:-5] for f in os.listdir(dump) if f.endswith(".json"))
    if not names:
        raise ValueError(f"no <id>.json in {dump}")
    os.makedirs(out, exist_ok=True)
    index = {"format": "ink-golden-v1", "source": "runner_dump_to_goldens.py", "prompts": []}
    for pid in names:
        d = json.load(open(os.path.join(dump, pid + ".json")))
        ids = d["prompt_ids"]
        if not ids or not all(isinstance(t, int) for t in ids):
            raise ValueError(f"{pid}: prompt_ids must be a non-empty list of ints")
        od = os.path.join(out, pid)
        os.makedirs(od, exist_ok=True)
        golden = {"id": pid, "kind": "ids", "prompt_ids": ids, "generated_ids": [], "source": "runner"}
        rows = []
        if d.get("logits_file"):
            pos = d["positions"]
            if pos != sorted(set(pos)) or pos[0] < 0 or pos[-1] > len(ids) - 1:
                raise ValueError(f"{pid}: positions must be ascending, unique, within 0..{len(ids) - 1}")
            rows = logprob_rows(os.path.join(dump, d["logits_file"]), pos, int(d["logits_row_length"]),
                                bool(d["logits_softcapped"]), ids)
            golden["generated_ids"] = [rows[-1]["top"][0][0]] if rows[-1]["pos"] == len(ids) - 1 else []
            golden["logprobs_file"] = "logprobs.json"
            golden["logprobs_first_pos"] = pos[0]
            json.dump(rows, open(os.path.join(od, "logprobs.json"), "w"))
        if d.get("hidden_file"):
            n = len(ids)
            src = os.path.join(dump, d["hidden_file"])
            if os.path.getsize(src) != 4 * VISITS * n * HIDDEN:
                raise ValueError(f"{pid}: hidden file is not F32 [{VISITS} x {n} x {HIDDEN}]")
            ex = d["experts"]
            if len(ex) != VISITS or any(len(v) != n or any(len(e) != 8 for e in v) for v in ex):
                raise ValueError(f"{pid}: experts must be [{VISITS}][{n}][8]")
            shutil.copyfile(src, os.path.join(od, "hidden_after_visit.f32"))
            json.dump({"hidden_file": "hidden_after_visit.f32", "hidden_shape": [VISITS, n, HIDDEN],
                       "visits": [{"visit": v, "experts": ex[v]} for v in range(VISITS)]},
                      open(os.path.join(od, "trace.json"), "w"))
            golden["trace_file"] = "trace.json"
        if goldens:
            check_against_reference(goldens, pid, d, rows)
        json.dump(golden, open(os.path.join(od, "golden.json"), "w"), indent=1)
        index["prompts"].append({"id": pid})
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    return names


def main():
    args = sys.argv[1:]
    goldens = None
    if "--goldens" in args:
        i = args.index("--goldens")
        if i + 1 >= len(args):
            sys.exit("--goldens needs a directory")
        goldens = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit("usage: runner_dump_to_goldens.py <dump-dir> <out-dir> [--goldens <dir>]")
    goldens = goldens or os.path.join(ink_dir(), "goldens")
    try:
        names = convert(args[0], args[1], goldens)
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"runner_dump_to_goldens.py: {e}")
    print(f"{len(names)} prompts written to {args[1]}: {', '.join(names)}")


if __name__ == "__main__":
    main()
