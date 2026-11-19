#!/usr/bin/env python3
"""Write the committed digest of a golden run (docs/NATIVE_PARITY.md).

    python3 Scripts/native/golden/make_slice.py <run-dir> [--run metal]

<run-dir> is a capture directory such as
$PROGRAM/goldens/quick-v1/metal (one subdirectory per prompt, each with
golden.json). Writes Tests/Mference/Fixtures/native/goldens/quick-v1-<run>-slice.json:
per prompt the token counts and hashes, the greedy tokens, the top-8 logits
and log-sum-exp of every logits row, the SHA-256 of logits.f32 and prompt.txt,
and one digest over every tensor file (sha256 of the sorted "<file> <sha256>"
lines). The
slice is small enough to commit; it pins the multi-gigabyte goldens (which are
not committed) and lets model-free tests check the sampler and parity metrics
against real llama.cpp numbers. Stdlib only, no model.
"""
import argparse
import array
import hashlib
import heapq
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "Tests/Mference/Fixtures/native"
TOP = 8


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def top_and_lse(row):
    best = heapq.nlargest(TOP, range(len(row)), key=row.__getitem__)
    mx = row[best[0]]
    lse = mx + math.log(math.fsum(math.exp(v - mx) for v in row))
    return {"ids": best, "values": [float(row[i]) for i in best], "log_sum_exp": lse}


def slice_prompt(d):
    m = json.loads((d / "golden.json").read_text(encoding="utf-8"))
    tokens = array.array("i", m["prompt_tokens"])
    if sys.byteorder != "little":
        tokens.byteswap()
    n_vocab = m["n_vocab"]
    logits = array.array("f")
    with open(d / m["logits"]["file"], "rb") as f:
        logits.frombytes(f.read())
    rows = m["logits"]["rows"]
    assert len(logits) == rows * n_vocab, f"{d}: logits size"
    tops = [top_and_lse(logits[r * n_vocab:(r + 1) * n_vocab]) for r in range(rows)]
    files = {m["logits"]["file"]: sha256_file(d / m["logits"]["file"]),
             m["rendered_prompt_file"]: sha256_file(d / m["rendered_prompt_file"])}
    # one digest over every tensor file: sha256 of "<file> <sha256>\n" lines, sorted by file
    lines = sorted(f"{t['file']} {sha256_file(d / t['file'])}\n" for t in m["tensors"])
    tensors_digest = hashlib.sha256("".join(lines).encode()).hexdigest()
    return {
        "prompt_token_count": len(m["prompt_tokens"]),
        "prompt_tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "rendered_prompt_sha256": files[m["rendered_prompt_file"]],
        "generated_tokens": m["generated_tokens"],
        "stopped_on_eog": m["stopped_on_eog"],
        "logits_top": tops,
        "files": dict(sorted(files.items())),
        "tensor_count": len(m["tensors"]),
        "tensors_digest": tensors_digest,
    }, m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--run", default="metal")
    ap.add_argument("--set", default="quick-v1")
    args = ap.parse_args(argv)
    prompt_set = FIXTURES / f"prompts/{args.set}.json"
    ids = [p["id"] for p in json.loads(prompt_set.read_text(encoding="utf-8"))["prompts"]]
    out = {"format": "mfe-golden-slice-v1", "prompt_set": args.set,
           "prompt_set_sha256": sha256_file(prompt_set), "run": args.run, "source": None, "prompts": {}}
    for pid in ids:
        d = args.run_dir / pid
        if not (d / "golden.json").exists():
            print(f"missing {d}/golden.json", file=sys.stderr)
            return 1
        s, m = slice_prompt(d)
        out["prompts"][pid] = s
        src = dict(m["source"])
        src.pop("model_path", None)
        src.pop("n_ctx", None)
        out["source"] = out["source"] or src
        print(f"{pid}: {s['prompt_token_count']} prompt tokens, {len(s['generated_tokens'])} generated, "
              f"{len(s['files'])} files", file=sys.stderr)
    dest = FIXTURES / f"goldens/{args.set}-{args.run}-slice.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # compact: one prompt per line, values with float32 round-trip precision
    for s in out["prompts"].values():
        for top in s["logits_top"]:
            top["values"] = [float(f"{v:.9g}") for v in top["values"]]
            top["log_sum_exp"] = float(f"{top['log_sum_exp']:.12g}")
    head = {k: v for k, v in out.items() if k != "prompts"}
    lines = [json.dumps(head, separators=(",", ":"))[:-1] + ',"prompts":{']
    items = list(out["prompts"].items())
    for i, (pid, s) in enumerate(items):
        lines.append(json.dumps(pid) + ":" + json.dumps(s, separators=(",", ":")) + ("," if i + 1 < len(items) else ""))
    lines.append("}}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
