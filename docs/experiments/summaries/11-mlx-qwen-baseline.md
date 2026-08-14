# mlx-lm Qwen 3.6 baseline attempt — M5/24 GB, 2026-08-07

Goal: measure mlx-lm decoding `mlx-community/Qwen3.6-35B-A3B-4bit` on the
24 GB M5 (`Mac17,8`-class host `Mac17,2`) with the three frozen community
prompts, as the same-host bar for the residency-ladder program.

Setup: fresh venv, mlx-lm 0.31.3, checkpoint downloaded complete. Runner
applies the identical chat messages and sampling (temp 0.2, top-k 64,
top-p 0.95) via `mlx_lm.load`/`generate`.

## Result: DNF (out of GPU memory)

Every attempt — including a 128-token short run in a fresh process with
nothing else active — failed inside generation with:

```
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

The arithmetic says this is structural, not incidental: the 4-bit checkpoint
is ~19.5 GB of weights that mlx-lm wires for the GPU, while macOS caps the
GPU working set on a 24 GB machine at roughly 18 GB
(`recommendedMaxWorkingSetSize` ≈ 75% of RAM). The model does not fit the
cap even before KV cache and activations. This is consistent with the
repository's earlier mlx-lm comparison using Gemma 4 (≈15 GB, which fits)
rather than Qwen.

## Consequence for the Phase 1 claim

On a 24 GB Mac, the honest Qwen 3.6 headline is not a speed ratio:

> Mference decodes Qwen 3.6 35B-A3B at 24–29 tok/s in ~1.5 GB of process
> memory on a host where mlx-lm cannot run the model at all.

The head-to-head kernel race against mlx-lm on identical weights needs
either a model both engines can hold (Gemma 4 on this host: mlx-lm
76–82 tok/s vs Mference 31–35 — see BENCHMARKS.md) or a larger-memory host
for Qwen. The kernel program keeps its own justification regardless: Qwen
decode here runs at ~27% of the ~90 tok/s bandwidth roofline (≈1.7 GB
active bytes/token over ~153 GB/s), so the headroom is real and local.
