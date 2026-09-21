#!/usr/bin/env python3
"""Check the distributable tracked source, without extracting or downloading."""
import io
from pathlib import PurePosixPath
import subprocess
import tarfile


def validate_archive(archive):
    """Validate unprefixed git archive bytes, including when Python uses -O."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members = tar.getmembers()
        entries = {item.name: item for item in members}
    if len(entries) != len(members):
        raise ValueError("duplicate source archive member")
    required = ["Package.swift", "Package.resolved", "LICENSE", "LICENSE-APACHE", "LICENSE-MLX",
                "README.md", "mference-ui.sh", "Scripts/test.sh", "Scripts/openwebui-mference.py",
                "Scripts/openwebui-configure-models.py", "docs/SOURCE_RELEASE.md"]
    for name in required:
        if name not in entries or not entries[name].isfile():
            raise ValueError(f"missing source-release file: {name}")
    for name in ["mference-ui.sh", "Scripts/test.sh"]:
        if not entries[name].mode & 0o111:
            raise ValueError(f"source archive loses executable bit: {name}")
    for prefix in ["Sources/Mference/Metal/", "Sources/MferenceCLI/", "Sources/MferenceServer/",
                   "Sources/MferenceRepack/", "Tests/"]:
        if not any(name.startswith(prefix) and item.isfile() for name, item in entries.items()):
            raise ValueError(f"missing source-release directory: {prefix}")
    for name, item in entries.items():
        parts = PurePosixPath(name).parts
        if (name.startswith("/") or ".." in parts or not parts
                or any(part in {".git", ".build", ".env", ".venv", ".codex", "__pycache__",
                               "scratch", "webui-secret-key"}
                       or part.endswith(".gturbo") or part.startswith(".env.") for part in parts)):
            raise ValueError(f"unexpected private/build/model payload: {name}")
        # The package currently needs only regular files and directories. Do
        # not distribute links or special entries with extraction side effects.
        if not (item.isfile() or item.isdir()):
            raise ValueError(f"unsupported source archive member: {name}")
    return len(entries)


def main():
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], check=True, capture_output=True).stdout
    count = validate_archive(archive)
    print(f"source archive: {count} entries; required files, scripts, resources and exclusions pass")


if __name__ == "__main__":
    main()
