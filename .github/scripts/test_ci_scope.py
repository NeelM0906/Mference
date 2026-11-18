#!/usr/bin/env python3
"""Tests for .github/scripts/ci-scope.sh (stdlib only, no model, no network)."""

import pathlib
import subprocess
import unittest

SCRIPT = pathlib.Path(__file__).resolve().with_name("ci-scope.sh")
REPO = SCRIPT.parents[2]
# Markdown paths no CI code names (the script searches the checkout).
PROBE = "docs/ci-scope-probe-unreferenced.md"
PROBE_NESTED = "docs/ci-scope-probe-dir/unreferenced.md"


def scope(*paths: str) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        input="\n".join(paths) + ("\n" if paths else ""),
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class CIScopeTests(unittest.TestCase):
    def test_empty_list_runs_everything(self):
        self.assertEqual(scope(), "full")
        self.assertEqual(scope("", ""), "full")

    def test_documentation_only(self):
        self.assertEqual(scope(PROBE), "docs")
        self.assertEqual(scope(PROBE, PROBE_NESTED, ".github/ISSUE_TEMPLATE/probe.md"), "docs")

    def test_docs_that_ci_reads_are_code(self):
        # Read by FamilyContractDocTests (MferenceTestsCore) with no env gate.
        self.assertEqual(scope("docs/FAMILY_CONTRACT.md"), "full")
        self.assertEqual(scope("docs/FAMILY_CONTRACT.md", "Sources/MferenceEngine/A.swift"), "full")
        # Non-Markdown under docs/ (tensor maps, prompt sets) is test input.
        self.assertEqual(scope("docs/families/qwen38flashnext-unsloth-ud-iq3_xxs.tensors.json"), "full")
        self.assertEqual(scope("docs/benchmark-prompts/release-screen-v1/cases.json"), "full")
        self.assertEqual(scope("docs/assets/probe.png"), "full")
        self.assertEqual(scope("site/index.html"), "full")
        self.assertEqual(scope("LICENSE-GGML"), "full")
        # A Markdown file inside a directory that CI code names is code.
        self.assertEqual(scope("docs/benchmark-prompts/real-generation-v1/NOTES.md"), "full")

    def test_markdown_inside_code_trees_is_code(self):
        self.assertEqual(scope("Tests/Mference/Fixtures/x/README.md"), "full")
        self.assertEqual(scope("Vendor/swift-jinja/README.md"), "full")
        self.assertEqual(scope("Sources/MferenceServer/NOTES.md"), "t0")
        self.assertEqual(scope("Sources/LlamaCpp/vendoring/VENDORING.md"), "t0")

    def test_engine_server_llama_and_scripts_are_t0(self):
        self.assertEqual(
            scope(
                "Sources/MferenceEngine/EngineHost.swift",
                "Tests/MferenceEngine/EngineHostTests.swift",
                "Sources/LlamaCpp/src/llama.cpp",
                "Tests/LlamaCpp/SmokeTests.swift",
                "Sources/MferenceServer/Core/HTTPServer.swift",
                "Tests/MferenceServer/Fixtures/a.json",
                "Scripts/engine-validate.sh",
                "mference-ui.sh",
                PROBE,
            ),
            "t0",
        )

    def test_runtime_and_shared_inputs_are_full(self):
        for path in (
            "Sources/Mference/Runtime/Inference/Model.swift",
            "Sources/Mference/Metal/Attention/x.metal",
            "Sources/MferenceRepack/Core/Planning/P.swift",
            "Sources/MferenceCLI/Args.swift",
            "Sources/MferenceValidation/Support/Tolerance.swift",
            "Tests/Mference/Core/Kernels/Attention/T.swift",
            "Tests/MferenceRepack/Core/X.swift",
            "Vendor/swift-jinja/Sources/Jinja/Lexer.swift",
            "Package.swift",
            "Package.resolved",
            ".github/workflows/ci.yml",
            ".github/scripts/ci-scope.sh",
            ".gitattributes",
            "Scripts/test.sh",
            "bringup-check.sh",
        ):
            with self.subTest(path=path):
                self.assertEqual(scope(PROBE, "Sources/MferenceEngine/A.swift", path), "full")
                self.assertEqual(scope(path, PROBE), "full")


if __name__ == "__main__":
    unittest.main()
