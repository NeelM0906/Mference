#!/usr/bin/env python3
"""Check the distributable tracked source, without extracting or downloading."""
import io
from pathlib import PurePosixPath
import subprocess
import tarfile


def main():
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        entries = {item.name: item for item in tar.getmembers()}
    required = ["Package.swift", "Package.resolved", "LICENSE", "LICENSE-APACHE", "LICENSE-MLX",
                "README.md", "mference-ui.sh", "Scripts/test.sh", "Scripts/openwebui-mference.py",
                "Scripts/openwebui-configure-models.py", "docs/SOURCE_RELEASE.md"]
    for name in required:
        assert name in entries and entries[name].isfile(), f"missing source-release file: {name}"
    for name in ["mference-ui.sh", "Scripts/test.sh"]:
        assert entries[name].mode & 0o111, f"source archive loses executable bit: {name}"
    for prefix in ["Sources/Mference/Metal/", "Sources/MferenceCLI/", "Sources/MferenceServer/",
                   "Sources/MferenceRepack/", "Tests/"]:
        assert any(name.startswith(prefix) and item.isfile() for name, item in entries.items()), prefix
    for name in entries:
        parts = PurePosixPath(name).parts
        assert not any(part in {".git", ".build", ".env", "scratch", "webui-secret-key"}
                       or part.endswith(".gturbo") for part in parts), f"unexpected private/build/model payload: {name}"
    print(f"source archive: {len(entries)} entries; required files, scripts, resources and exclusions pass")


if __name__ == "__main__":
    main()
