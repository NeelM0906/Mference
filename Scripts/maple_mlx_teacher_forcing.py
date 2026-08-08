#!/usr/bin/env python3
"""Export canonical Maple MLX teacher-forcing top-k JSONL from a local pin."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse

from fetch_maple_raven import DEFAULT_OUTPUT, load_manifest as load_corpus_manifest, verify
from verify_maple_snapshot import verify_snapshot


SCHEMA = "mference.maple.teacher_forcing.v1"
MODEL_ID = "maple-preview-2bit-mlx"
SOURCE_REPO_ID = "deepgrove/maple-preview-2bit-mlx"
SOURCE_REVISION = "361db5da5e74ff6fcdd852d478e1f266ce11013a"
SOURCE_INDEX_SHA256 = "56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95"
SOURCE_SNAPSHOT_MANIFEST_SHA256 = "ac8b6d4b118d982b215c98697cca50ebe770ad3d8f68b7ef10a582fd52fb9a5c"
CONFIG_SHA256 = "57eb521da63629196ebda2c103be929c81c1027ddf2766e7b19e2d2427f77443"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
ENGINE_SOURCE_REVISION = "eba96c16158f032821b0bf374ea1421cfddef0a9"
EXPECTED_PYTHON = "3.12.13"
EXPECTED_PACKAGES = {
    "mlx": "0.32.0", "mlx-lm": "0.31.3", "transformers": "5.14.1",
    "tokenizers": "0.22.2", "numpy": "2.5.1",
}
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE_SOURCE = ROOT / "scratch/mlx-lm-deepgrove"
EXPECTED_TOKEN_COUNT = 1639
EXPECTED_TOKEN_IDS_SHA256 = "979f944999a0a5039bfe3a9074ca0886ea54a228663fc8ad828a8759871f261f"
TOKEN_POLICY = "raw-utf8-lf-nfc;add_special_tokens=false;bos=false;eos=false"
VOCAB_SIZE = 151_936


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_token_hash(token_ids: list[int]) -> str:
    return sha256(json.dumps(token_ids, separators=(",", ":")).encode("ascii"))


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def verify_installed_package(source: Path, installed: Path) -> None:
    def files(root: Path, *, source_tree: bool) -> dict[Path, Path]:
        return {
            path.relative_to(root): path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and not (source_tree and (path.suffix == ".md" or path.relative_to(root).parts[0] == "examples"))
        }

    expected = files(source / "mlx_lm", source_tree=True)
    actual = files(installed, source_tree=False)
    if expected.keys() != actual.keys():
        raise RuntimeError("installed mlx_lm file set does not match the pinned source")
    for relative, expected_path in expected.items():
        if expected_path.read_bytes() != actual[relative].read_bytes():
            raise RuntimeError(f"installed mlx_lm differs from pinned source at {relative}")


def inspect_pinned_runtime() -> tuple[dict[str, str], str]:
    if platform.python_version() != EXPECTED_PYTHON:
        raise RuntimeError(f"Python {platform.python_version()} != pinned {EXPECTED_PYTHON}")
    versions = {"python": platform.python_version()}
    for package, expected in EXPECTED_PACKAGES.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} {actual} != pinned {expected}")
        versions[package] = actual

    direct_url_text = importlib.metadata.distribution("mlx-lm").read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("mlx-lm has no direct_url.json")
    direct_url = json.loads(direct_url_text)
    directory_info = direct_url.get("dir_info", {})
    if not isinstance(directory_info, dict) or directory_info.get("editable", False):
        raise RuntimeError("mlx-lm must be a non-editable install from the pinned local fork")
    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise RuntimeError("mlx-lm must be installed from a local source directory")
    source = Path(unquote(parsed.path)).resolve()
    if source != EXPECTED_ENGINE_SOURCE.resolve():
        raise RuntimeError(f"mlx-lm source {source} != pinned {EXPECTED_ENGINE_SOURCE}")

    if git(source, "rev-parse", "HEAD") != ENGINE_SOURCE_REVISION:
        raise RuntimeError("mlx-lm source revision does not match the pin")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("mlx-lm source is dirty")
    if git(ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Mference harness checkout is dirty")
    harness_revision = git(ROOT, "rev-parse", "HEAD")
    if len(harness_revision) != 40 or any(character not in "0123456789abcdef" for character in harness_revision):
        raise RuntimeError("Mference harness revision is not a full lowercase commit hash")

    installed = Path(importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm")).resolve()
    verify_installed_package(source, installed)
    versions["harness"] = harness_revision
    return versions, ENGINE_SOURCE_REVISION


def run_preflight(snapshot: Path) -> None:
    subprocess.run(
        [str(ROOT / "Scripts/maple_oracle_preflight.sh"), str(snapshot), str(os.getpid())],
        check=True,
    )


def emit(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="complete local MLX snapshot")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=0, help="diagnostic prefix only; zero exports the complete corpus")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_positions < 0 or args.max_positions >= EXPECTED_TOKEN_COUNT:
        raise ValueError("--max-positions must be zero or a diagnostic prefix in 1..<1639")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing trace: {output}")
    if not args.model.is_dir():
        raise ValueError("--model must name a complete local MLX snapshot")

    run_preflight(args.model)
    runtime_versions, engine_revision = inspect_pinned_runtime()
    corpus_manifest = load_corpus_manifest()
    corpus_bytes = args.corpus.read_bytes()
    verify(corpus_bytes, corpus_manifest, label=str(args.corpus))
    verify_snapshot(args.model, full_sha256=True)
    index_sha256 = sha256((args.model / "model.safetensors.index.json").read_bytes())
    if index_sha256 != SOURCE_INDEX_SHA256:
        raise ValueError("Maple source index does not match the pin")

    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as error:
        raise RuntimeError("pinned MLX dependencies are missing") from error

    loaded_at = time.perf_counter()
    model, tokenizer, config = load(
        str(args.model), revision=SOURCE_REVISION, return_config=True,
        trust_remote_code=True, model_config={"use_flash_head": False},
    )
    load_seconds = time.perf_counter() - loaded_at
    token_ids = [int(token) for token in tokenizer.encode(corpus_bytes.decode("utf-8"), add_special_tokens=False)]
    if len(token_ids) != EXPECTED_TOKEN_COUNT or canonical_token_hash(token_ids) != EXPECTED_TOKEN_IDS_SHA256:
        raise ValueError("raw Raven tokenization does not match the Maple pin")
    vocab_size = int(config["vocab_size"])
    if vocab_size != VOCAB_SIZE:
        raise ValueError(f"vocabulary size {vocab_size} != pinned {VOCAB_SIZE}")
    positions = min(len(token_ids), args.max_positions or len(token_ids))
    token_ids_sha256 = canonical_token_hash(token_ids[:positions])
    acceptance_eligible = positions == EXPECTED_TOKEN_COUNT
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8", newline="\n") as handle:
        emit(handle, {
            "record_type": "metadata", "schema": SCHEMA, "engine": "mlx",
            "engine_version": runtime_versions["mlx-lm"], "engine_source_revision": engine_revision,
            "engine_source_dirty": False,
            "engine_binary_sha256": sha256_file(Path(sys.executable)),
            "command": [sys.executable, *sys.argv], "runtime_versions": runtime_versions,
            "model_id": MODEL_ID, "source_repo_id": SOURCE_REPO_ID,
            "model_revision": SOURCE_REVISION, "source_index_sha256": index_sha256,
            "source_snapshot_manifest_sha256": SOURCE_SNAPSHOT_MANIFEST_SHA256,
            "config_sha256": CONFIG_SHA256, "tokenizer_sha256": TOKENIZER_SHA256,
            "corpus_source_url": corpus_manifest["source_url"], "corpus_sha256": corpus_manifest["expected"]["sha256"],
            "corpus_token_count": len(token_ids), "token_ids_sha256": token_ids_sha256,
            "positions": positions, "token_policy": TOKEN_POLICY,
            "top_k": 10, "top_k_tie_break": "logit-desc-token-id-asc",
            "vocab_size": vocab_size, "logit_dtype": "float32-export-from-mlx", "exact_lm_head": True,
            "use_flash_head": False, "expert_cache_slots": 0,
            "integrity_policy": "full-sha256",
            "acceptance_eligible": acceptance_eligible,
        })
        cache = model.make_cache()
        started = time.perf_counter()
        for position, input_id in enumerate(token_ids[:positions]):
            row = model(mx.array([[input_id]], dtype=mx.int32), cache=cache)[0, -1].astype(mx.float32)
            mx.eval(row)
            values = [float(value) for value in row.tolist()]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite vocabulary logit at position {position}")
            top = sorted(enumerate(values), key=lambda pair: (-pair[1], pair[0]))[:10]
            emit(handle, {
                "record_type": "position", "position": position, "input_id": input_id,
                "target_id": token_ids[position + 1] if position + 1 < len(token_ids) else None,
                "top": [{"id": token_id, "logit": value} for token_id, value in top],
            })
            done = position + 1
            if done % 32 == 0 or done == positions:
                elapsed = time.perf_counter() - started
                print(f"[mlx teacher forcing {done}/{positions} {done / elapsed:.3f} positions/s]", file=sys.stderr, flush=True)
        elapsed = time.perf_counter() - started
        emit(handle, {
            "record_type": "summary", "positions": positions, "load_seconds": load_seconds,
            "elapsed_seconds": elapsed, "positions_per_second": positions / elapsed if elapsed else 0.0,
            "exit_code": 0,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
