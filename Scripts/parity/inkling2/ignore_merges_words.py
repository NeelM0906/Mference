#!/usr/bin/env python3
"""The vocabulary entries that plain BPE (merges by rank) does not rebuild as
one token: the words where llama.cpp's `ignore_merges` (inkling2 pre-tokenizer,
#32799) changes the encoding. Reads only the GGUF header (tokens, merges),
never tensor data. Stdlib only; run with -I.

    python3 -I Scripts/parity/inkling2/ignore_merges_words.py <gguf> <out.json>

Writes [{"id", "token" (byte-level form), "text" (decoded UTF-8), "bpe" (the
pieces plain merges produce)}], sorted by id.
"""

import json
import struct
import sys


def byte_decoder():
    # GPT-2 byte-level alphabet (bytes_to_unicode), inverted
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def read_header(path):
    f = open(path, "rb")
    rd = lambda n: f.read(n)
    u32 = lambda: struct.unpack("<I", rd(4))[0]
    u64 = lambda: struct.unpack("<Q", rd(8))[0]
    s = lambda: rd(u64()).decode("utf-8", errors="surrogateescape")
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    assert rd(4) == b"GGUF"
    u32(); u64(); n_kv = u64()
    out = {}
    for _ in range(n_kv):
        k = s()
        t = u32()
        if t == 8:
            v = s()
        elif t == 9:
            et = u32(); n = u64()
            if et == 8:
                v = [s() for _ in range(n)]
            elif et in sizes:
                rd(sizes[et] * n); v = None
            else:
                raise SystemExit(f"unsupported GGUF array element type {et} in {k}")
        else:
            rd(sizes[t]); v = None
        if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
            out[k] = v
        if len(out) == 2:
            break
    return out["tokenizer.ggml.tokens"], out["tokenizer.ggml.merges"]


def main():
    tokens, merges = read_header(sys.argv[1])
    ranks = {tuple(m.split(" ", 1)): i for i, m in enumerate(merges)}
    dec = byte_decoder()

    def bpe(word):
        parts = list(word)
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            i = best[1]
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return parts

    odd = []
    for i, tok in enumerate(tokens):
        if len(tok) > 1 and not (tok.startswith("<|") and tok.endswith("|>")):
            parts = bpe(tok)
            if parts != [tok]:
                try:
                    text = bytes(dec[c] for c in tok).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    text = None
                odd.append({"id": i, "token": tok, "text": text, "bpe": parts})
    json.dump(odd, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("entries not rebuilt by merges:", len(odd), "decodable:", sum(1 for o in odd if o["text"] is not None))


if __name__ == "__main__":
    main()
