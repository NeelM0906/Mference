# Inkling 2 Mini reference tools

The base reference for the Inkling 2 Mini family is the shipped weights file
run through stock upstream llama.cpp, built locally. These are the tools that
build it, run it safely on a shared 32 GB Mac, and write the goldens in
[`Tests/Mference/Fixtures/Inkling2/`](../../../Tests/Mference/Fixtures/Inkling2/README.md).
Nothing here is part of any product target, and nothing downloads anything.

| File | What it does |
|---|---|
| `llama/Makefile`, `llama/embed_metal.sh` | Builds upstream llama.cpp (a clean checkout at a release tag) without cmake: ggml-base, the CPU backend (native, llamafile, CPU repack, Accelerate), the BLAS backend (Accelerate), the Metal backend with its kernels embedded as upstream's default build does, libllama, llama-common, mtmd and the server libraries. Tools: `llama-cli`, `llama-server`, `llama-bench`, `llama-completion`, `llama-eval-callback`, `llama-tokenize`, `llama-gguf`, `ink-golden`. Writes `BUILD_INFO.txt` (tag, commit, flags, binary SHA-256s). |
| `ink_golden.cpp` | `tokenize`, `chat` (vocabulary only) and `generate` (greedy continuation, top-K log-probabilities, the per-visit trace through the scheduler's eval callback). |
| `gguf_header_dump.py`, `tensors_tables.py` | Header and tensor table from the GGUF without loading weights (stdlib; type ids and block sizes from the checkout's `gguf-py/gguf/constants.py`). |
| `guarded_run.sh` | Wraps one model command: preflight, no other model process, AC power for timing runs, swap sampled every 2 s (any growth stops the run, exit 70), peak footprint. Refuses to run outside a `lock.sh run` hold, and refuses tools and settings that read the weights into memory instead of mapping them (`llama-gguf`, quantize/split/hash tools, any `-lm`/`--load-mode` other than `auto` or `mmap`, `LLAMA_ARG_LOAD_MODE`, older `--no-mmap`/`--mlock` spellings). After a trip: INT, TERM, then KILL, each bounded. |
| `vocab_goldens.sh` | Tokenizer and chat-template goldens. |
| `golden_stage.sh` | One weight-loading golden stage (Metal, Metal again, CPU-only: for `--ngl 0` ink-golden opens no GPU device and turns op offload and CPU weight repacking off, so the weights stay mapped). |
| `baseline_stage.sh` | One baseline timing stage (llama-bench at 512, 4,096 and 16,384 tokens; llama-completion on real text for time to first token). Waits until at most one other swift build or test job runs. |
| `run_held_stages.sh` | All the weight-loading stages in order, each as its own machine-lock hold. |
| `compare_goldens.py` | Agreement between two `generate` outputs: token identity, top-1 agreement, log-probability deltas, approximate KL over the top 20, hidden-state error and routed-expert agreement per layer visit. |
| `ignore_merges_words.py` | The 337 vocabulary entries plain BPE merges split, from the GGUF header (tokens, merges) without touching tensor data. |
| `inputs/` | The frozen inputs: tokenizer texts, chat cases, generation prompt sets 1-3 (`generate_prompts*.json`; a prompt may set `logprobs_last` to record only its last N positions). |

Build (one build job; the checkout must be clean and at the tag):

```bash
make -f Scripts/parity/inkling2/llama/Makefile LL=<llama.cpp checkout> OUT=<build dir> TAG=b12641 -j4 all ink-golden
```

`llama-gguf <file> r` is built but must not be pointed at a large GGUF: its
read mode allocates every tensor in heap memory. `guarded_run.sh` refuses it.
