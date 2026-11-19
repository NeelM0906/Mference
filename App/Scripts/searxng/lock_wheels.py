#!/usr/bin/env python3
"""Pins one wheel per requirement for CPython 3.12 on macOS arm64.

Reads `name==version` lines (uv pip compile output) and writes
`filename url sha256` lines from PyPI's JSON API, preferring a native
cp312 arm64 wheel, then abi3, then universal2, then pure Python. Run it only
to change the pins; prepare.sh consumes its output (wheels.lock).
Usage: lock_wheels.py compiled.txt > wheels.lock
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request


def rank(filename: str) -> int | None:
    if not filename.endswith(".whl"):
        return None
    tags = filename[:-4].split("-")[-3:]
    py, abi, plat = tags
    mac_arm = "macosx" in plat and ("arm64" in plat or "universal2" in plat)
    if py == "cp312" and abi == "cp312" and mac_arm:
        return 0 if "arm64" in plat else 2
    m = re.fullmatch(r"cp3(\d+)", py.split(".")[0])
    if abi == "abi3" and mac_arm and m and int(m.group(1)) <= 12:
        return 1 if "arm64" in plat else 3
    if plat == "any" and abi == "none" and any(p in ("py3", "py2.py3") or p.startswith("py3") for p in py.split(".")):
        return 4
    return None


def main() -> None:
    for line in open(sys.argv[1]):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as response:
            files = json.load(response)["urls"]
        candidates = sorted((r, f["filename"]) for f in files if (r := rank(f["filename"])) is not None)
        if not candidates:
            sys.exit(f"no macOS arm64 cp312 wheel for {name}=={version}")
        chosen = next(f for f in files if f["filename"] == candidates[0][1])
        print(chosen["filename"], chosen["url"], chosen["digests"]["sha256"])


if __name__ == "__main__":
    main()
