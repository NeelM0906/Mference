#!/usr/bin/env python3
"""Fetch the explicitly requested, pinned Maple parity corpus into scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "Tests/MferenceMapleParity/Fixtures/RavenNormalization.json"
DEFAULT_OUTPUT = ROOT / "scratch/maple-parity/the-raven.txt"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
EXPECTED_MANIFEST_SHA256 = "9e877b1896c6f822d75ea58cda108e336e742a93f32f74acaf3dde6099347f25"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    data = path.read_bytes()
    if sha256(data) != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"unrecognized Raven manifest: {path}")
    manifest = json.loads(data)
    if manifest.get("schema") != "mference.maple.corpus.v1":
        raise ValueError(f"unsupported corpus manifest: {path}")
    return manifest


def normalize(raw: str, *, lines_per_stanza: int) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n") if line.strip()]
    if not lines or len(lines) % lines_per_stanza:
        raise ValueError("Raven source does not have complete stanzas")
    stanzas = ("\n".join(lines[index : index + lines_per_stanza])
               for index in range(0, len(lines), lines_per_stanza))
    return "\n\n".join(stanzas) + "\n"


def verify(data: bytes, manifest: dict, *, label: str) -> None:
    expected = manifest["expected"]
    if sha256(data) != expected["sha256"]:
        raise ValueError(f"{label}: SHA-256 does not match the pin")
    if len(data) != expected["utf8_bytes"]:
        raise ValueError(f"{label}: byte count does not match the pin")
    lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(lines) != expected["lines"]:
        raise ValueError(f"{label}: line count does not match the pin")


def fetch_archived_row(manifest: dict) -> str:
    archive = manifest["archive"]
    query = urllib.parse.urlencode({
        "dataset": archive["dataset"], "config": archive["config"],
        "split": archive["split"], "offset": archive["row_index"], "length": 1,
    })
    request = urllib.request.Request(
        f"{DATASET_ROWS_URL}?{query}", headers={"User-Agent": "Mference-Maple-Parity/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    rows = payload.get("rows", [])
    if len(rows) != 1 or rows[0].get("row_idx") != archive["row_index"]:
        raise ValueError("archive returned an unexpected Raven row")
    row = rows[0]["row"]
    if row.get("poem name", "").strip() != manifest["title"] or row.get("author", "").strip() != manifest["author"]:
        raise ValueError("archive provenance does not match the Raven pin")
    return row["content"]


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing corpus: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="verify an existing cache")
    args = parser.parse_args()
    manifest = load_manifest()
    output = args.output.expanduser().resolve()
    if output.exists():
        verify(output.read_bytes(), manifest, label=str(output))
        print(f"verified {output}")
        return 0
    if args.offline:
        raise FileNotFoundError(f"offline corpus cache is missing: {output}")
    normalized = normalize(fetch_archived_row(manifest), lines_per_stanza=manifest["normalization"]["lines_per_stanza"])
    data = normalized.encode("utf-8")
    verify(data, manifest, label="normalized Raven corpus")
    write_new(output, data)
    print(f"cached {output} sha256={sha256(data)} bytes={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
