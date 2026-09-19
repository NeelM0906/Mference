# Swift Jinja compatibility snapshot

Source: https://github.com/huggingface/swift-jinja
Version: **2.5.1**
Revision: `4588064a20f3fc093c95f2f7d3359999bf30cae5`
License: Apache 2.0; the upstream `LICENSE` and source notices are retained.

This directory contains the upstream package manifest, source, and tests.
Mference uses this local package to fix whitespace control in `Lexer.swift`.
Upstream 2.5.1 inserts spaces when a literal `{` precedes a whitespace-controlled
tag. That changes Google's canonical Gemma tool schemas and their token IDs.
The local lexer consumes whitespace controls at token boundaries, preserving
literal braces without rewriting delimiters or quoted string contents.
Google's bundled template is unmodified.

Only `Sources/Jinja/Lexer.swift` differs from the upstream source snapshot.
Mference's `GemmaJinjaWhitespaceTests` covers the defect, and
`GemmaThinkingTests` compares all canonical prompt bytes and IDs against an
independent Python oracle. The root package's `JinjaCompatibilityTests` target
runs the unmodified upstream tests through `Scripts/test.sh`.

When an upstream release fixes this defect, replace the local package only
after the same oracle, dependency tests, and model-family template tests pass.
