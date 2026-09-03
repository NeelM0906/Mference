#!/usr/bin/env python3
"""Compare the per-tensor quantization *mixture* of two .gturbo installs.

The W2.1b model-level gate rests on the two installs being comparable on every
tensor. That is a claim about width, per tensor, and it is cheap to check
exactly: a resident index records each tensor's logical shape alongside the
byte count of its packed payload, so the stored width is
`8 * sizeBytes / prod(shape)` — no config file, no trust in either installer's
own manifest.

    Scripts/quantizer-mixture-compare.py CONTROL.gturbo OURS.gturbo

Exit status is 0 only when the two installs quantize exactly the same tensors
at exactly the same widths. Anything else is reported by name, because a
silent width difference on even one router would make a KLD comparison measure
routing divergence instead of weight fidelity (docs/QUANTIZER_QUALITY.md §6).

Names are normalized because the two sides come from differently-named
sources: an mlx-lm conversion writes `language_model.model.<...>` while the
vendor's original repo writes `model.language_model.<...>`.
"""
import os
import struct
import sys

INDEX_HEADER_BYTES = 24
INDEX_ENTRY_BYTES = 72


def read_index(path):
    """Every resident entry as (name, dims, size_bytes, scale_size)."""
    with open(path, "rb") as f:
        head = f.read(INDEX_HEADER_BYTES)
        index_size, _resident_size, entry_count = struct.unpack("<QQQ", head)
        f.seek(0)
        blob = f.read(index_size)
    out = []
    for i in range(entry_count):
        base = INDEX_HEADER_BYTES + i * INDEX_ENTRY_BYTES
        rec = blob[base:base + INDEX_ENTRY_BYTES]
        name_off, name_len = struct.unpack("<IH", rec[0:6])
        size_bytes = struct.unpack("<Q", rec[16:24])[0]
        dims = [d for d in struct.unpack("<IIII", rec[24:40]) if d > 0]
        scale_size = struct.unpack("<Q", rec[48:56])[0]
        name = blob[name_off:name_off + name_len].decode()
        out.append((name, dims, size_bytes, scale_size))
    return out


def normalize(name):
    """Canonical tensor name, independent of which repo the install came from."""
    for prefix in ("language_model.model.", "model.language_model.",
                   "language_model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def width_of(dims, size_bytes, scale_size):
    """Stored bits per weight, or None for a tensor that is not quantized."""
    if scale_size == 0 or not dims:
        return None
    count = 1
    for d in dims:
        count *= d
    if count == 0:
        return None
    return round(8 * size_bytes / count)


def mixture(path):
    weights = os.path.join(path, "model_weights.bin")
    if not os.path.exists(weights):
        raise SystemExit(f"no resident index at {weights}")
    table = {}
    for name, dims, size_bytes, scale_size in read_index(weights):
        table[normalize(name)] = width_of(dims, size_bytes, scale_size)
    return table


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    control_path, ours_path = sys.argv[1], sys.argv[2]
    control, ours = mixture(control_path), mixture(ours_path)

    only_control = sorted(set(control) - set(ours))
    only_ours = sorted(set(ours) - set(control))
    shared = sorted(set(control) & set(ours))
    differing = [(n, control[n], ours[n]) for n in shared if control[n] != ours[n]]

    def census(table):
        counts = {}
        for width in table.values():
            key = "unquantized" if width is None else f"INT{width}"
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{k} {counts[k]}" for k in sorted(counts))

    print(f"control {control_path}")
    print(f"  {len(control)} resident tensors: {census(control)}")
    print(f"ours    {ours_path}")
    print(f"  {len(ours)} resident tensors: {census(ours)}")

    print(f"\nshared tensors: {len(shared)}")
    print(f"width mismatches: {len(differing)}")
    for name, c, o in differing[:20]:
        print(f"  {name}: control INT{c} vs ours INT{o}")
    if len(differing) > 20:
        print(f"  ... and {len(differing) - 20} more")
    if only_control:
        print(f"\nonly in the control ({len(only_control)}):")
        for name in only_control[:20]:
            print(f"  {name}")
    if only_ours:
        print(f"\nonly in ours ({len(only_ours)}):")
        for name in only_ours[:20]:
            print(f"  {name}")

    # The set that matters most for this gate, called out by name.
    control_int8 = {n for n, w in control.items() if w == 8}
    ours_int8 = {n for n, w in ours.items() if w == 8}
    print(f"\nINT8 override set: control {len(control_int8)}, ours {len(ours_int8)}")
    if control_int8 == ours_int8:
        print("  identical")
    else:
        for name in sorted(control_int8 ^ ours_int8)[:20]:
            side = "control only" if name in control_int8 else "ours only"
            print(f"  {name} ({side})")

    ok = not differing and not only_control and not only_ours
    print("\nMIXTURES MATCH" if ok else "\nMIXTURES DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
