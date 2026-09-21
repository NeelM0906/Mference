import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_source_archive import validate_archive
from package_source_release import package


class SourceReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "release-test@example.invalid")
        self.git("config", "user.name", "Source Release Test")
        for name in ["Package.swift", "Package.resolved", "LICENSE", "LICENSE-APACHE", "LICENSE-MLX",
                     "README.md", "mference-ui.sh", "Scripts/test.sh", "Scripts/openwebui-mference.py",
                     "Scripts/openwebui-configure-models.py", "docs/SOURCE_RELEASE.md",
                     "Sources/Mference/Metal/probe.metal", "Sources/MferenceCLI/probe.swift",
                     "Sources/MferenceServer/probe.swift", "Sources/MferenceRepack/probe.swift", "Tests/probe.swift"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test fixture\n")
            if name.endswith(".sh"):
                path.chmod(0o755)
        self.git("add", ".")
        self.git("commit", "-qm", "source fixture")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True).stdout

    def output(self, name="out"):
        return Path(self.temporary.name) / name

    def test_reproducible_checksums_provenance_and_modes(self):
        first = package(self.root, self.output(), "0.1.0-rc.1")
        second = package(self.root, self.output("again"), "0.1.0-rc.1")
        for a, b in zip(first["files"], second["files"]):
            self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())
        archive, manifest, checksums = map(Path, first["files"])
        metadata = json.loads(manifest.read_text())
        self.assertEqual(metadata["commit"], self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(metadata["archive"]["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        for line in checksums.read_text().splitlines():
            digest, name = line.split("  ")
            self.assertEqual(digest, hashlib.sha256((self.output() / name).read_bytes()).hexdigest())
        with tarfile.open(archive) as tar:
            self.assertTrue(tar.getmember("mference-0.1.0-rc.1/mference-ui.sh").mode & 0o111)
            self.assertTrue(all(m.name.startswith("mference-0.1.0-rc.1") for m in tar.getmembers()))

    def test_untracked_models_and_secrets_never_enter_archive(self):
        (self.root / ".env").write_text("NOT_A_REAL_SECRET=test\n")
        (self.root / "scratch").mkdir()
        (self.root / "scratch/model.gturbo").write_text("not weights\n")
        result = package(self.root, self.output(), "0.1.0")
        with tarfile.open(result["files"][0]) as tar:
            self.assertFalse(any(".env" in m.name or "scratch" in m.name for m in tar.getmembers()))

    def test_dirty_tracked_and_overwrite_rejected(self):
        package(self.root, self.output(), "0.1.0")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            package(self.root, self.output(), "0.1.0")
        (self.root / "README.md").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "commit tracked"):
            package(self.root, self.output("dirty"), "0.1.0")

    def test_invalid_version_rejected(self):
        for version in ["../escape", "v1", "1.02.3", "1.2.3-rc.01", "$(command)", "1.2.3/else"]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                package(self.root, self.output(), version)

    def test_missing_file_and_executable_bit_rejected(self):
        (self.root / "mference-ui.sh").chmod(0o644)
        self.git("add", ".")
        self.git("commit", "-qm", "lose execute bit")
        with self.assertRaisesRegex(ValueError, "executable"):
            package(self.root, self.output(), "0.1.0")
        self.git("rm", "LICENSE")
        self.git("commit", "-qm", "missing license")
        with self.assertRaisesRegex(ValueError, "missing.*LICENSE"):
            package(self.root, self.output(), "0.1.0")

    def test_unsafe_archive_members_rejected(self):
        raw = self.git("archive", "--format=tar", "HEAD")
        for name, kind in [("../escape", tarfile.REGTYPE), (".env.production", tarfile.REGTYPE),
                           ("scratch/model.gturbo/manifest.json", tarfile.REGTYPE),
                           ("link", tarfile.SYMTYPE), ("fifo", tarfile.FIFOTYPE),
                           ("README.md", tarfile.REGTYPE)]:
            output = io.BytesIO()
            with tarfile.open(fileobj=io.BytesIO(raw)) as source, tarfile.open(fileobj=output, mode="w") as dest:
                for member in source.getmembers():
                    dest.addfile(member, source.extractfile(member) if member.isfile() else None)
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "../../outside" if kind == tarfile.SYMTYPE else ""
                dest.addfile(member)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_archive(output.getvalue())


if __name__ == "__main__":
    unittest.main()
