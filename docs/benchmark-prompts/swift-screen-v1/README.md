# Swift screen v1

Frozen before execution: 12 deterministic functional cases, two each for
everyday arithmetic, reasoning, code reading, exact instruction following,
tool selection/arguments, and supplied-report synthesis. No model-generated
code or tools are executed. Rubrics and expected answers are in cases.json.

Primary metric: exact success count; any HTTP failure, truncation, empty answer,
extra prose in JSON or wrong/multiple tool calls is a failure. Secondary metric:
total generated tokens, including reasoning. Whitespace around visible answers
is ignored. This small screen is not evidence of broad quality preservation.

Run the existing release loopback library server with MFERENCE_MTP=0 and
4096 context, then Scripts/swift_qwen_task_screen.py --output NEW_FILE.jsonl.
All requests use temperature 0, top-p 1, top-k 1, repetition penalty 1, seed
20260916 and at most 512 completion tokens. Base has no effort parameter;
Swift is tested at medium (no added effort instruction), xhigh (its proposed
default), low and none. Base's no-tool template opens thinking without an
effort instruction; its legacy tool template disables thinking. Therefore
tool cases are not an effort-matched base-versus-Swift comparison, and thinking-
off versus thinking-on savings must not be attributed solely to fine-tuning.

The JSONL records corpus hash, settings, every request/response, usage, errors,
and wall time. Times include cold-load/swap and are operational diagnostics,
not comparable throughput measurements. No warmups or repeated measurements
are performed; formal latency claims require the separate frozen community
benchmark and the broader task-screen protocol in the execution plan.
Exit 0 means the screen finished recording every case, not that every case
passed; inspect the `passed` fields and failure responses.

Preflight correction: the initial request configuration used top-k 0, which
the server rejects (its range is 1–256). That attempted batch produced only
HTTP 400 errors and no model answers. Corrected to top-k 1 before scoring any
generated responses; temperature 0 remains deterministic greedy decoding.
