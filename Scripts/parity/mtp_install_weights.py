"""Read only the installed MTP sidecar and shared head; never copy a checkpoint."""
import json
import mmap
import struct

import numpy as np
import torch


def load_install(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["arch"]["family"] != "qwen38flashnext":
        raise ValueError("expected a Flash-Next install")
    receipt = json.loads((root / "verified-install.json").read_text())
    for name, info in receipt["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != info["size"]:
            raise ValueError(f"invalid receipt entry: {name}")

    def bf16(data, offset, count):
        return (np.frombuffer(data, dtype="<u2", count=count, offset=offset)
                .astype(np.uint32) << 16).view(np.float32)

    def packed(data, offset, count, bits, scale_offset, bias_offset):
        raw = np.frombuffer(data, dtype=np.uint8, count=count * bits // 8, offset=offset)
        if bits == 4:
            values = np.empty(count, dtype=np.float32)
            values[0::2], values[1::2] = raw & 15, raw >> 4
        elif bits == 8:
            values = raw.astype(np.float32)
        else:
            raise ValueError(f"unsupported packed width {bits}")
        groups = values.reshape(-1, 64)
        groups *= bf16(data, scale_offset, count // 64)[:, None]
        groups += bf16(data, bias_offset, count // 64)[:, None]
        return values

    weights = {}
    with (root / "model_weights.bin").open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        index_size, _, entries = struct.unpack_from("<QQQ", data)
        for index in range(entries):
            entry = struct.unpack_from("<IHBBQQ4IQQQQ", data, 24 + 72 * index)
            no, nl, dtype, _, offset, size, *rest = entry
            shape, so, ss, bo, bs = rest[:4], *rest[4:]
            if no + nl > index_size:
                raise ValueError("invalid tensor name range")
            name = data[no:no + nl].decode()
            if not (name.startswith("mtp.") or name == "lm_head.weight"):
                continue
            shape = [n for n in shape if n]
            count = int(np.prod(shape))
            if dtype == 1:
                values = bf16(data, offset, count)
            elif dtype == 0:
                bits = size * 8 // count
                if ss != count // 64 * 2 or bs != ss or size * 8 != count * bits:
                    raise ValueError(f"invalid quantization companions: {name}")
                values = packed(data, offset, count, bits, so, bo)
            else:
                raise ValueError(f"unsupported dtype: {name}")
            weights[name] = torch.from_numpy(values.reshape(shape))
            if manifest.get("zeroCenteredNormsBakedAtInstall") and name.endswith((
                "hc_norm.weight", "q_norm.weight", "k_norm.weight", "q_layernorm.weight", "k_layernorm.weight",
                "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
            )):
                # Upstream adds one at runtime; undo only an explicitly
                # declared install-time fold so it is not applied twice.
                weights[name] = weights[name] - 1
    arch = manifest["arch"]
    pool = next(p for p in manifest["auxiliaryExpertPools"] if p["name"] == "mtp")
    if len(pool["layers"]) != 1 or pool["layers"][0]["layer"] != 0:
        raise ValueError("expected exactly one MTP layer")
    path = (root / pool["directory"] / pool["layers"][0]["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("invalid MTP pool path")
    d, f = arch["hiddenSize"], arch["moeIntermediateSize"]
    count = d * f
    size, companion = count // 2, count // 64 * 2
    projection = size + companion * 2
    with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for expert in range(arch["numExperts"]):
            for i, (name, shape) in enumerate((("gate", (f, d)), ("up", (f, d)), ("down", (d, f)))):
                offset = expert * pool["expertStride"] + i * projection
                values = packed(data, offset, count, 4, offset + size, offset + size + companion)
                weights[f"mtp.layers.0.mlp.experts.{expert}.{name}_proj.weight"] = torch.from_numpy(values.reshape(shape))
    return arch, weights
