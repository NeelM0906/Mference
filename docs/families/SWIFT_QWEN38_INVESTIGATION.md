# Swift-Qwen token-budget and tool-parser investigation — 2026-09-18

This is a diagnostic investigation, not a replacement for the frozen release
screen, community performance protocol, or model-quality qualification.
The user explicitly requested that the GLM range installation continue in the
background. Consequently, none of the wall times below establish performance.

## Upstream evidence

The [model card](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) reports
relative token reductions, not a promise that thinking fits into 1,024 tokens.
Its reproduction settings are temperature 1, top-p 0.95, top-k 20, repetition
penalty 1, xhigh, and five seeds. Its task-dependent output caps are much larger
than our community generation cap. Its INT4 results also use a different
quantization pipeline from Mference's all-projection affine INT4 group-64.

[Discussion #7](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/discussions/7)
reports more reasoning tokens and looping on a 30-problem AIME comparison.
The author requests matched configuration and multiple seeds, and acknowledges
a training issue involving a math-related token. The report compares BF16
Swift with FP8 base and uses different decoding settings, so it is not a
controlled reproduction of the card or proof of a Mference bug.
[Discussion #2](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/discussions/2)
contains positive feedback alongside an anecdote about lengthy xhigh reasoning;
[#5](https://huggingface.co/ukisai/Swift-Qwen3.8-27b/discussions/5) is positive
but supplies no measurements. These reports are mixed, not a known universal
empty-answer defect. Discussions #7 and #5 were read through the public Hub API
after the web page reader failed.

## Template and comparison audit

The installed Swift template, pinned upstream revision
`1b30aaaf753fe5c1cb51ada2ea0367a53445359c`, and current upstream template all hash
to `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
There is no upstream template drift to fix. The installed base-Qwen template
has that same hash too.

Swift executes that source template: xhigh adds an explicit reasoning system
instruction, medium omits the effort instruction, low requests brief reasoning,
and none closes the thinking block. Existing independent Jinja oracle tests
cover ordinary chat, history and tool-result history. Stop IDs match upstream's
generation config: 248046 and 248044. No sampling/default changes were made.

**The legacy base-Qwen serving path is not an effort-matched control.** Ordinary
base chat uses the hand-written ChatML renderer without the xhigh instruction;
its tool path explicitly disables thinking. Swift uses source-default xhigh
for both. Default-vs-default product comparisons remain useful, but cannot
attribute differences solely to the fine-tune. A future matched-effort evaluation
must verify rendered prompts/IDs, including tools and history, before running.
The frozen 512-token functional screen remains unchanged; it is not a
reproduction of the upstream efficiency evaluation.

## Real-model cutoff experiment

Runtime source `7505e8233c759c4d19af7703ea1fcd11c29b9098`, invoked from checkout
`7bcd01c6382ebb7dcc6dcf631f53823e0d8e8b34` while parser-only changes were being
prepared. Mac Studio Mac15,14, M3 Ultra, 32 CPU cores, 256 GiB; macOS 26.3
(25D125); Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3`). Preflight: 97% memory free,
678 GiB disk free, completed seven-file Swift install receipt and no other
model/test owner. The server uses strict default verification. GLM Repack was
the only other Mference process and was left running.

```sh
env MFERENCE_MTP=0 /tmp/mference-phase1-build.sXnNTs/release/MferenceServer \
  --model scratch/swiftqwen38.gturbo --port 18489 --max-context 16384 \
  --prompt-cache-mode off
python3 Scripts/swift_qwen_diagnostic.py --profile xhigh-legacy \
  --output /tmp/mference-swift-investigation-xhigh-community-seed.json
python3 Scripts/swift_qwen_diagnostic.py --profile xhigh-legacy --max-tokens 1024 \
  --output /tmp/mference-swift-investigation-xhigh-cap1024.json
```

Both requests use the unchanged short-explanation community prompt, seed
20260721, temperature 0.2, top-k 64, top-p 0.95, repetition penalty 1, xhigh,
MTP off and no tools or prefix reuse. They differ only in completion cap.

| Cap | Stop | Generated | Reasoning | Visible |
| --- | --- | --- | --- | --- |
| 1,024 | length | 1,024 | 1,024 | 0 |
| 4,096 | stop | 1,952 | 1,620 | 330 |

The 4,040-character reasoning from the capped run is an **exact prefix** of
the 5,414-character reasoning in the completed run. The completed run returns
a substantive visible answer. Thus this particular failure is a budget cutoff,
not a hidden completed answer, malformed tool argument, or broken thinking-end
parser. It does not settle the longer community cases or overall quality.

An exploratory run of the same 4,096-cap profile with seed 20260918 completed
at 1,673 generated / 1,359 reasoning / 312 visible tokens; its evidence is
`/tmp/mference-swift-investigation-xhigh-legacy.json`. It preceded selection of
the original community seed and is not pooled with the controlled comparison.
The diagnostic script's initial default seed was 20260918; its now-explicit
`--seed` default is 20260721. Full requests are saved with each response.

Protocol deviations from a performance measurement: concurrent GLM download;
16,384 context instead of 4,096; diagnostic cap increased to 4,096 in named
runs; no warmup/repetition series. These results never overwrite the prior
truncations or establish an efficiency percentage.

Additional source-sampling diagnostics (seed 20260721, cap 4,096, temperature 1,
top-k 20, top-p 0.95, repetition penalty 1):

```sh
python3 Scripts/swift_qwen_diagnostic.py --profile xhigh-source \
  --output /tmp/mference-swift-investigation-xhigh-source.json
python3 Scripts/swift_qwen_diagnostic.py --profile low-source \
  --output /tmp/mference-swift-investigation-low-source.json
```

| Effort | Stop | Generated | Reasoning | Visible |
| --- | --- | --- | --- | --- |
| xhigh | stop | 2,684 | 2,193 | 489 |
| low | stop | 839 | 235 | 602 |

All five text diagnostic commands exited 0. Complete server completion footers:

```text
[2026-09-18T17:25:33Z] request chatcmpl-1d4c2a8b8cf948f4be4fb6186b3bcfc5 completed in 48.6s prompt=102 cached=0 completion=1673 finish=stop
[2026-09-18T17:26:48Z] request chatcmpl-34e5835a1fb147cbadb7cbfd26e3c8f5 completed in 56.5s prompt=102 cached=0 completion=1952 finish=stop
[2026-09-18T17:27:49Z] request chatcmpl-5f90ec9b9f694a5fab671157d73bb299 completed in 29.7s prompt=102 cached=0 completion=1024 finish=length
[2026-09-18T17:29:45Z] request chatcmpl-6ade6bf511ef4b82933d425762734308 completed in 116.2s prompt=102 cached=0 completion=2684 finish=stop
[2026-09-18T17:30:21Z] request chatcmpl-db489b48121a46b89e245d4fd6c7e251 completed in 35.7s prompt=90 cached=0 completion=839 finish=stop
```

These are successful protocol completions, **not independently scored quality
passes**. Low effort changes both reasoning and answer content; its response
contains quantitative claims that need checking. One seed/task cannot establish
token savings versus base Qwen or justify silently changing the default.

## Parser defects found and corrected

1. **String arguments were guessed as JSON types.** Before the fix, the real
   model's `echo(text: string)` call requesting the literal `123` returned
   `{"text":123}`. The server now passes tool schemas into the Qwen parser;
   an explicit `type: string` preserves the raw string, including numeric,
   boolean, null and JSON-looking text. Other values retain existing inference.
   This is not full JSON Schema validation or resolution of ambiguous unions
   and references; clients must still validate arguments and apply permissions.
2. **Tool-looking text in reasoning could become an actual call or parse
   error.** The ChatML decoder now handles thinking-channel boundaries before
   attempting tool parsing. A real visible-channel call still owns its entire
   payload, so literal `<think>` text inside string arguments stays intact.
   This second defect is established by regression fixtures, not observed as
   the cause of the real tool-free cutoff.

Neither correction changes logits, sampling, reasoning effort, or token caps.
The string defect was reproduced with temperature 0, top-k/top-p 1, effort none,
512 output tokens, seed 20260918: prompt 268, completion 27, finish tool_calls,
HTTP success. No generated tool was executed.

## Verification

Implementation commit: `7ac56b22c20787afefeb1184abbf20d8737620dd`. Hardware and
toolchain are as above. The diagnostic server was stopped cleanly (exit 0)
before package tests; GLM installation continued. No cache purge or model copy.

The focused command was:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --filter 'SwiftQwenChatTests|QwenToolCallParserTests|QwenToolCallParserDepthTests|ChatMLDecoderTests|MapleTokenizationTests|SwiftQwenServerTests|HTTPServerTests'
```

The first incremental build exited 1 at link time: an existing server-test
object still referenced the original decoder initializer symbol after its
signature gained a parameter. The fix retains the original public initializer
overloads and adds schema-aware overloads, rather than purging build caches.
The focused rerun, before the final analogous parser-API forwarding overload,
exited 0 (`/tmp/mference-swift-parser-regressions-v2.log`):

```text
Build complete! (9.70s)
Test run with 72 tests in 8 suites passed after 3.222 seconds.
```

This includes both Qwen profiles, source-template rendering, reasoning/tool
isolation, schema strings and history round-trip, numeric inference, unknown
tool rejection, size/depth limits, Maple compatibility and HTTP/SSE fixtures.
Independent Python Jinja rendering was also repeated: all 12 committed oracle
cases matched exactly. Python compilation, Markdown links (70 files), and the
source-archive check (827 tracked entries before this report was committed)
passed. The focused run alone is not full installed-model qualification.

The complete suite on `7ac56b2` subsequently exited 0:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  Scripts/test.sh --scratch-path /tmp/mference-phase1-build.sXnNTs
```

Full footer (`/tmp/mference-swift-parser-full-suite.log`):

```text
Build complete! (9.35s)
Test run with 1248 tests in 222 suites passed after 283.921 seconds with 1 known issue.
```

The known issue remains the absent optional Flash-Next toy checkpoint.
Environment-gated real-model tests do not execute in this ordinary suite.

Release server build on the same implementation commit exited 0:

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift build -c release --scratch-path /tmp/mference-phase1-build.sXnNTs \
  --product MferenceServer
```

Full footer: `Build of product 'MferenceServer' complete! (57.84s)`.
Log: `/tmp/mference-swift-parser-release-server-build.log`. This did not relink
the running release installer. Fresh preflight before restarting the same
loopback server: no model/test owner, 97% memory free, 659 GiB disk free, completed
Swift install. The GLM installer remained active.

Live post-fix command, exit 0:

```sh
python3 Scripts/swift_qwen_diagnostic.py --profile tool-string --seed 20260918 \
  --output /tmp/mference-swift-investigation-tool-string-fixed.json
```

The identical failing request now returns `{"text":"123"}` with the same
268 prompt / 27 completion tokens and `finish_reason=tool_calls`. A separate
SSE request changed the literal to `true`, enabled `stream` and
`stream_options.include_usage`, and returned `{"text":"true"}`, normal
`tool_calls` termination, usage (265 prompt / 25 completion), and `[DONE]`.
Neither string was coerced to a number or boolean. Both client commands exited 0.

The SSE command was:

```sh
curl -fsS -N http://127.0.0.1:18489/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"swift-qwen3.8-27b-int4g64","reasoning_effort":"none","messages":[{"role":"user","content":"Call echo with text exactly true. Do not answer in prose."}],"tools":[{"type":"function","function":{"name":"echo","description":"Echo the exact text","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}],"temperature":0,"top_k":1,"top_p":1,"repetition_penalty":1,"seed":20260918,"max_completion_tokens":512,"stream":true,"stream_options":{"include_usage":true}}'
```

A history follow-up appended the exact returned assistant call, a **simulated**
tool result `123`, and the user request `Return that result as a JSON string and
nothing else.` The server accepted the history and completed, but emitted
`{"result":"123"}` rather than the requested JSON scalar `"123"`. This is an
**instruction-following failure**, not a parser success being counted as task
quality. No tool code was executed. All live tests used the one server above;
it was stopped cleanly afterwards (exit 0), leaving GLM downloading.

```sh
jq '. as $r | .request | .messages += [$r.response.choices[0].message, {role:"tool", tool_call_id:$r.response.choices[0].message.tool_calls[0].id, content:"123"}, {role:"user",content:"Return that result as a JSON string and nothing else."}]' \
  /tmp/mference-swift-investigation-tool-string-fixed.json | \
  curl -fsS http://127.0.0.1:18489/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

Complete post-fix server completion footers:

```text
[2026-09-18T17:39:14Z] request chatcmpl-bab4ffc900854cb0938a53ad36aaf643 completed in 5.0s prompt=268 cached=0 completion=27 finish=tool_calls
[2026-09-18T17:39:32Z] request chatcmpl-1194aad492ab4b95905535071a6e5155 completed in 3.7s prompt=265 cached=0 completion=25 finish=tool_calls
[2026-09-18T17:39:51Z] request chatcmpl-18c61f7e341e43999ce3774acb4de7ed completed in 3.9s prompt=331 cached=0 completion=8 finish=stop
```

## Remaining qualification

- Prepare genuinely matched base/Swift prompt policies, then perform the frozen
  functional screen and a separately predeclared sufficiently budgeted,
  multi-seed efficiency comparison after the GLM download is no longer competing.
- Include failures and quality scores; fewer tokens with a worse/incomplete
  answer do not qualify a replacement. MTP stays off until independently measured.
