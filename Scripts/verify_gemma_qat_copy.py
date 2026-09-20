#!/usr/bin/env python3
"""Bounded independent source-copy check for the pinned native QAT install.

Run after MferenceRepack --verify-install. Reads fresh source headers, all 30
router matrices, and documented resident/expert samples; never stages a shard
or modifies the installation. This is byte-copy proof, not execution parity.
Uses Python's standard library only. Network reads are capped at 1 MiB except
the index/header metadata (4/16 MiB caps). Four sample requests may run at once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen

REPO = "mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned"
REVISION = "745a97a754ed4b7713163c7d0e9c11da41809e0c"
INDEX_SHA = "7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
SAMPLE_BYTES = 256
SELECTED_LAYERS = {0, 5, 29}  # Sliding and full attention, early and final layers.
SELECTED_EXPERTS = {0, 63, 127}


def fetch(name, start=None, size=None, cap=4 * 1024 * 1024):
    headers = {"Accept-Encoding": "identity"}
    url = BASE + name
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + size - 1}"
        # Keep differently ranged requests distinct in intermediary caches.
        url += f"?copy_check_range={start}-{size}"
        cap = size
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                data = response.read(cap + 1)
                if len(data) > cap:
                    raise ValueError(f"oversize response for {name}")
                if start is not None:
                    expected = f"bytes {start}-{start + size - 1}/"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                        raise ValueError(f"unexpected range response for {name}")
                    if len(data) != size:
                        raise ValueError(f"short range for {name}")
                return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    root = args.model.resolve()
    manifest_data = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    assert manifest["modelID"] == "gemma-4-26b-a4b-it-qat-q4_0-mlx-aligned"
    assert manifest["sourceSnapshotHash"] == "sha256:" + INDEX_SHA
    index_data = fetch("model.safetensors.index.json")
    assert hashlib.sha256(index_data).hexdigest() == INDEX_SHA
    weight_map = json.loads(index_data)["weight_map"]
    tensors = {}
    header_bytes = len(index_data)
    for shard in sorted(set(weight_map.values())):
        size = struct.unpack("<Q", fetch(shard, 0, 8))[0]
        assert 0 < size <= 16 * 1024 * 1024
        header = json.loads(fetch(shard, 8, size))
        header_bytes += 8 + size
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            assert name not in tensors and weight_map[name] == shard
            tensors[name] = (shard, 8 + size, tensor)
    assert set(tensors) == set(weight_map) and len(tensors) == 1279
    spans = []
    routers = set()

    def add(name, local_file, local_start, size, source_start=0, whole=False):
        shard, payload_base, tensor = tensors[name]
        bounds = tensor["data_offsets"]
        assert source_start + size <= bounds[1] - bounds[0]
        offsets = [0] if whole or size <= SAMPLE_BYTES else [0, size - SAMPLE_BYTES]
        for offset in offsets:
            count = size if whole else min(size, SAMPLE_BYTES)
            spans.append((shard, payload_base + bounds[0] + source_start + offset,
                          count, local_file, local_start + offset, name))

    with (root / "model_weights.bin").open("rb") as f:
        index_size, resident_size, count = struct.unpack("<QQQ", f.read(24))
        assert 24 + count * 72 <= index_size <= 8 * 1024 * 1024
        assert index_size + resident_size == (root / "model_weights.bin").stat().st_size
        f.seek(0)
        resident_index = f.read(index_size)
    for row in range(count):
        fields = struct.unpack_from("<I H B x Q Q 4I Q Q Q Q", resident_index, 24 + row * 72)
        name_offset, name_size, dtype, offset, size = fields[:5]
        name = resident_index[name_offset:name_offset + name_size].decode()
        router = name.endswith(".router.proj.weight")
        if ".layers." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            if layer not in SELECTED_LAYERS and not router:
                continue
        if router:
            assert dtype == 1 and tensors[name][2]["dtype"] == "BF16"
            assert size == 128 * 2816 * 2 and fields[10] == fields[12] == 0
            routers.add(name)
        add(name, "model_weights.bin", offset, size, whole=router)
        for component, component_offset, component_size in [
            ("scales", fields[9], fields[10]), ("biases", fields[11], fields[12])
        ]:
            if component_size:
                add(name.removesuffix("weight") + component, "model_weights.bin",
                    component_offset, component_size)
    assert len(routers) == 30

    layout = json.loads((root / "packed_experts/layout.json").read_bytes())
    for layer in layout["layers"]:
        if layer["layer"] not in SELECTED_LAYERS:
            continue
        for expert in layer["experts"]:
            if expert["expert"] not in SELECTED_EXPERTS:
                continue
            for role in ["gate", "up", "down"]:
                for key_suffix, tensor_suffix in [("", "weight"), ("_scales", "scales"), ("_biases", "biases")]:
                    region = expert["tensors"][role + key_suffix]
                    name = f"language_model.model.layers.{layer['layer']}.experts.switch_glu.{role}_proj.{tensor_suffix}"
                    assert region["size"] * 128 == tensors[name][2]["data_offsets"][1] - tensors[name][2]["data_offsets"][0]
                    add(name, "packed_experts/" + layer["file"], expert["offset"] + region["offset"],
                        region["size"], expert["expert"] * region["size"])

    groups = []
    for span in sorted(spans):
        shard, start, size, *_ = span
        end = start + size
        if groups and groups[-1][0] == shard and start <= groups[-1][2] + 2048 and end - groups[-1][1] <= 1024 * 1024:
            groups[-1][2] = max(groups[-1][2], end)
            groups[-1][3].append(span)
        else:
            groups.append([shard, start, end, [span]])

    def check(group):
        shard, begin, end, members = group
        data = fetch(shard, begin, end - begin)
        for _, start, size, local_file, local_start, name in members:
            with (root / local_file).open("rb") as f:
                f.seek(local_start)
                local = f.read(size)
            assert len(local) == size and local == data[start - begin:start - begin + size], name
        return end - begin

    with ThreadPoolExecutor(max_workers=4) as pool:
        transferred = sum(pool.map(check, groups))
    print(json.dumps({
        "status": "passed", "repo": REPO, "revision": REVISION,
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "source_tensor_count": len(tensors), "resident_entry_count": count,
        "full_router_matrices_checked": len(routers), "sampled_layers": sorted(SELECTED_LAYERS),
        "sampled_experts_per_layer": sorted(SELECTED_EXPERTS), "sample_bytes_per_edge": SAMPLE_BYTES,
        "compared_spans": len(spans), "compared_bytes": sum(s[2] for s in spans),
        "source_metadata_bytes": header_bytes, "payload_range_requests": len(groups),
        "payload_bytes_transferred_excluding_retries": transferred,
        "largest_payload_range": max(g[2] - g[1] for g in groups),
        "coverage_limit": "All routers; edges of resident roles in layers 0/5/29 plus embedding/final norm, and routed roles for experts 0/63/127 in those layers. Unsampled weight bytes rely on strict install verification. No execution parity claim.",
    }, indent=2))


if __name__ == "__main__":
    main()
