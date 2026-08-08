#!/usr/bin/env python3
"""Verify a local pinned Maple MLX snapshot without loading its weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "Tests/MferenceMapleParity/Fixtures/maple-snapshot.json"
EXPECTED_MANIFEST_SHA256 = "ac8b6d4b118d982b215c98697cca50ebe770ad3d8f68b7ef10a582fd52fb9a5c"
EXPECTED_REPO_ID = "deepgrove/maple-preview-2bit-mlx"
EXPECTED_REVISION = "361db5da5e74ff6fcdd852d478e1f266ce11013a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"unrecognized Maple snapshot manifest: {path}")
    manifest = json.loads(data)
    if manifest.get("schema") != "mference.maple.snapshot.v1":
        raise ValueError(f"unsupported Maple snapshot manifest: {path}")
    if manifest.get("repo_id") != EXPECTED_REPO_ID or manifest.get("revision") != EXPECTED_REVISION:
        raise ValueError(f"Maple source pin does not match the oracle: {path}")
    return manifest


def verify_snapshot(root: Path, manifest_path: Path = DEFAULT_MANIFEST, *, full_sha256: bool = False) -> dict:
    manifest = load_manifest(manifest_path)
    root = root.expanduser().resolve()
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise ValueError(f"pinned Maple snapshot is missing {path}")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"{path}: size does not match the pin")
        verification = entry["verification"]
        if verification == "sha256" or full_sha256:
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{path}: SHA-256 does not match the pin")
            continue
        if verification != "hf-lfs-metadata":
            raise ValueError(f"{path}: unsupported verification mode {verification!r}")
        metadata = root / ".cache/huggingface/download" / f"{entry['path']}.metadata"
        if not metadata.is_file():
            raise ValueError(f"{path}: missing Hugging Face LFS metadata; use --full-sha256")
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0] != manifest["revision"] or lines[1] != entry["sha256"]:
            raise ValueError(f"{metadata}: does not prove the pinned LFS object")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--full-sha256", action="store_true", help="audit all local weight bytes")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    manifest = verify_snapshot(args.snapshot, full_sha256=args.full_sha256)
    if not args.quiet:
        mode = "full-sha256" if args.full_sha256 else "sidecars+LFS-metadata"
        print(f"verified {args.snapshot.resolve()} {manifest['repo_id']}@{manifest['revision']} mode={mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
