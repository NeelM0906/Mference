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

`Sources/Jinja/Interpreter.swift` also distinguishes an omitted conditional
else (undefined) from explicit null, and provides a scoped null-output policy.
Gemma QAT uses that policy to emit Python Jinja2's `None` for null tool
arguments. Other checkpoints retain the existing empty null output. The QAT
source template is unmodified and its bytes and token IDs are compared against
an independent Python oracle in `GemmaQATChatTests`.

`Sources/Jinja/Lexer.swift` and `Sources/Jinja/Interpreter.swift` differ from the upstream source snapshot.
Mference's `GemmaJinjaWhitespaceTests` covers the defect, and
`GemmaThinkingTests` compares all canonical prompt bytes and IDs against an
independent Python oracle. The root package's `JinjaCompatibilityTests` target
runs the unmodified upstream tests through `Scripts/test.sh`.

When an upstream release fixes this defect, replace the local package only
after the same oracle, dependency tests, and model-family template tests pass.
