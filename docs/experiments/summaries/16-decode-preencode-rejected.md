# Pre-encoded decode command sequences: rejected on encode cost, M1 Max/32 GB, 2026-11-15

## Hypothesis

Moving per-call parameters into buffers would let each layer's decode command
sequence be encoded once into an indirect command buffer (ICB) and replayed
every token. That would remove CPU encode time from the critical path of a
decode step. This was the optional step D3 of the M1 Max performance program;
it was to be pursued only if CPU encode turned out to be the largest remaining
decode term.

## Measurement

These were standalone Metal probes, lead diagnostics rather than protocol
benchmarks. The host was a MacBook Pro M1 Max (24-core GPU), 32 GB, macOS
27.2.1. One "token" is 900 dispatches of a small kernel with four buffer
bindings and two constants, which is Qwen 3.6's documented dispatch count. The
probe ran 2 blocks of 500 tokens, after 30 warm-up tokens per mode, and reports
medians.

| Encoding | CPU encode µs / token | µs / dispatch | GPU µs / token |
| --- | ---: | ---: | ---: |
| One compute encoder per dispatch (the decode path's shape) | 540–548 | 0.60–0.61 | 2,717–3,049 |
| One encoder, `memoryBarrier(scope: .buffers)` between dispatches | 255–259 | 0.28–0.29 | 1,867–1,904 |
| Pre-encoded ICB, one `executeCommandsInBuffer` per token | 11–13 | 0.012–0.015 | 2,090–2,110 |

On the same host, Qwen 3.6 decodes at about 34 tok/s (about 29 ms per token)
once the M1 keep-warm fix is in (PR #54). The GPU is busy for 43–47% of the
decode span. Each synchronous command-buffer round trip costs 165–330 µs, of
which 4–90 µs is GPU work.

## Disposition

**Rejected.** Encoding is about 0.6 µs per dispatch, or about 0.5 ms of a
~29 ms token, roughly 2%. It is not the largest remaining term. The idle gaps
come from tens of synchronous CPU–GPU handoffs per token, so the work goes to
removing those handoffs (resident experts with GPU routing, and halt-and-resume
on the slot path) instead.

## Lesson

On this hardware CPU encode is cheap. The repository's earlier M5 estimate was
1.0–1.3 µs per dispatch, and the M1 Max measured 0.6. Measure encode on the
target host before restructuring how parameters are passed.

The probe also measured GPU-side overhead. One encoder per dispatch costs about
1 µs more GPU time per dispatch than a single encoder with barriers. That
matters only where dispatches are tiny.

An ICB is still worth keeping in mind for skipping work on the GPU: an ICB run
with a GPU-written execution range of length 0 skipped 800 dispatches in about
9 µs. That is a different use from pre-encoding, and it was not adopted here.
