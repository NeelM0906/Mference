#!/usr/bin/env python3
"""Create a reproducible source candidate. Does not tag, upload or publish.

The archive is the committed tree, never a copy of the local working directory.
Tests/ contains small tracked reference fixtures; installed models are excluded.
CI/test success is deliberately not inferred by this packaging-only utility.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from check_source_archive import validate_archive


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def package(root, output, version):
    if not re.fullmatch(r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?", version):
        raise ValueError("version must be a plain semantic version, e.g. 0.1.0-rc.1")
    if git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("commit tracked changes before packaging; the candidate must identify exactly one committed tree")
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    tree = git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    timestamp = int(git(root, "show", "-s", "--format=%ct", "HEAD"))
    raw = git(root, "-c", "tar.umask=0022", "archive", "--format=tar", commit)
    count = validate_archive(raw)
    base = f"mference-{version.removeprefix('v')}"
    archive_name = base + ".tar.gz"
    manifest_name = base + ".json"
    checksums_name = base + "-SHA256SUMS"
    # Git fixes tar metadata to the commit, gzip fixes its timestamp/filename.
    # GzipFile (not gzip.compress) also fixes the OS header across Python hosts.
    import io
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=9) as compressed:
        compressed.write(git(root, "-c", "tar.umask=0022", "archive", "--format=tar", f"--prefix={base}/", commit))
    archive = stream.getvalue()
    digest = hashlib.sha256(archive).hexdigest()
    manifest = (json.dumps({
        "schema_version": 1, "version": version.removeprefix("v"),
        "commit": commit, "tree": tree, "commit_timestamp": timestamp,
        "distribution": "source-only", "archive_entries": count,
        "archive": {"file": archive_name, "sha256": digest, "bytes": len(archive)},
        "qualification": "Packaging checks only. Consult the commit's release validation and support records; this manifest does not certify model quality, hardware coverage or CI success.",
    }, indent=2, sort_keys=True) + "\n").encode()
    checksums = (f"{digest}  {archive_name}\n"
                 f"{hashlib.sha256(manifest).hexdigest()}  {manifest_name}\n").encode()
    payloads = {archive_name: archive, manifest_name: manifest, checksums_name: checksums}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() or (output / name).is_symlink() for name in payloads):
        raise ValueError("refusing to overwrite an existing release artifact")
    # Exclusive creation also protects against a concurrent packaging process.
    for name, data in payloads.items():
        with (output / name).open("xb") as destination:
            destination.write(data)
    return {"commit": commit, "files": [str(output / name) for name in payloads]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = package(Path(__file__).resolve().parents[1], args.output, args.version)
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        parser.exit(1, f"source release refused: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
