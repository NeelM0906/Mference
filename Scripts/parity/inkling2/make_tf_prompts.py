#!/usr/bin/env python3
"""Teacher-forced prompts from a set of ink-golden goldens (stdlib only).

    python3 -I Scripts/parity/inkling2/make_tf_prompts.py <out.json> <generate-dir> [<generate-dir> ...]

For every prompt with log-probs in a golden directory (e.g. generate/metal),
writes an "ids" prompt with the same id: its prompt ids followed by all but
the last generated id, n_gen 1, log-probs on. Run through `ink-golden
generate`, that gives another backend's (or the same backend's prefill-only)
log-probabilities on exactly the golden's context at every recorded position:
positions from the golden's `logprobs_first_pos` to n_prompt + n_gen - 2.
compare_goldens.py --teacher-forced pairs them with the golden.
"""

import json
import os
import sys


def main():
    out = []
    for d in sys.argv[2:]:
        for p in json.load(open(os.path.join(d, "index.json")))["prompts"]:
            g = json.load(open(os.path.join(d, p["id"], "golden.json")))
            if not g.get("logprobs_file"):
                continue
            ids = g["prompt_ids"] + g["generated_ids"][:-1]
            rec = {"id": p["id"], "kind": "ids", "ids": ids, "n_gen": 1, "logprobs": True}
            first = g.get("logprobs_first_pos", 0)
            if first > 0:
                # same first position: L - 1 - N = first, with L = len(ids)
                rec["logprobs_last"] = len(ids) - 1 - first
            out.append(rec)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(len(out), "teacher-forced prompts")


if __name__ == "__main__":
    main()
