# Third-party software and model terms

Mference is licensed under the [MIT License](LICENSE). It does not relicense
model weights or third-party packages.

## TurboFieldfare

Mference's runtime, installer, and application code are derived from
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey
Mikhaylov and contributors, licensed under the
[Apache License 2.0](LICENSE-APACHE). The derived portions remain governed by
that license. Significant changes from the original include: the Mference
rebrand throughout; support for Qwen 3.6, DeepSeek-V4-Flash, Inkling-Small,
Maple, Qwen3.8-Flash-Next, and MiniCPM5-2B across architecture configuration,
streaming installation, Metal kernels, forward runners, tokenization, and
products; model-derived kernel specialization; and expanded validation and
performance documentation.

This file records the dependency review performed on 2026-08-09. It is an
attribution aid, not legal advice. Anyone distributing a compiled product must
also preserve the license and NOTICE material required by the exact dependency
versions included in that product.

## Apple MLX and DeepGrove MLX-LM

Maple's shape-specific Metal arithmetic, including vector SDPA, was adapted
from Apple MLX 0.32.0 and the Maple model implementation in
[`deepgrove-ai/mlx-lm-deepgrove`](https://github.com/deepgrove-ai/mlx-lm-deepgrove)
at revision `eba96c16158f032821b0bf374ea1421cfddef0a9`. Both upstream trees
carry the MIT license. The applicable Apple and DeepGrove copyright notices
are preserved in [`LICENSE-MLX`](LICENSE-MLX).

## Model weights

Model weights are not included in this repository. The installer downloads a
pinned revision of one of seven checkpoints and repacks it locally. Five are
pre-quantized community conversions; Qwen3.8-Flash-Next and MiniCPM5-2B are
read from their vendors' own BF16 repositories and quantized in flight.

Gemma 4: revision `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` of
[`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit).
Its model card describes it as an Apache-2.0 quantization of Google's Gemma 4
26B-A4B instruction checkpoint. Google publishes Gemma 4 under the
[Apache License 2.0](https://ai.google.dev/gemma/apache_2); note that earlier
Gemma generations use the separate Gemma Terms of Use instead.

Qwen 3.6: revision `38740b847e4cb78f352aba30aa41c76e08e6eb46` of
[`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit),
a quantization of Alibaba's
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) checkpoint, whose
[license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE)
governs those weights.

DeepSeek-V4-Flash: revision `722bf559b7de93575b2320973cf2002e05bfe6c9` of
[`mlx-community/DeepSeek-V4-Flash-2bit-DQ`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ).

Inkling-Small: revision `9d6e4720ab7002af25d6129c88ccea6cd9f19372` of
[`pipenetwork/Inkling-Small-MLX-4bit`](https://huggingface.co/pipenetwork/Inkling-Small-MLX-4bit).

Maple Preview: revision `361db5da5e74ff6fcdd852d478e1f266ce11013a` of
[`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx).
The installer verifies its source-index SHA-256
`56000110535c5023b43209a5c142035e12c1cde7b1118759cc9f86335d46ef95`.
That pinned repository revision does not declare a license for the checkpoint
weights in a model card, LICENSE file, or Hub license tag. Downloading, using,
or redistributing those weights therefore requires separately establishing
that the necessary rights have been obtained; Mference makes no grant of
rights to them.

Qwen3.8-Flash-Next: revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
Alibaba's own BF16 checkpoint rather than a community conversion — the
installer reads its 131 shards and quantizes to INT4 group-64 in flight. The
installer verifies its source-index SHA-256
`99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`.
That revision carries a Hub `license: other` tag and a
[LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
file naming the **Qwen Community License 1.0** (Copyright (c) 2026 Qwen). It is
a permissive grant to use, modify, distribute, sublicense, sell, host,
fine-tune, and create derivative works from the weights, subject to two
conditions that a redistributor must check for itself: attribution of the model
name on the user interface of any commercial product or service exceeding 100
million monthly active users or US$20 million monthly revenue, and a separate
license from Qwen before commercial use in a "Model as a Service" or "AI Work
Assistant" business as that license defines them. The license text governs;
this summary is an attribution aid, not legal advice. No reference code from
that repository is imported into Mference — the runner, kernels and repack path
are written against the published architecture, not copied from it.

MiniCPM5-2B: revision `cd199ce3ee67549c42ef7372f809f2c63599a3e9` of
[`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B), the
vendor's own BF16 upload, quantized to INT4 affine group-64 at install; the
installer verifies its source-index SHA-256
`6d839cd76e8395de548a0e6cc310386f66d1ecbb2c75d198a8dfd3d70892b756`. OpenBMB
publishes the model and its weights under the
[Apache License 2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)
(`license: apache-2.0` in the model card). The W2.1b quantizer-quality control,
revision `35ac38ee7bdb0bf7fa748d0700eeb6d6675760a3` of
[`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)
(index SHA-256 `ccf202e0a06fe3c7eb8f354cfb29412a5e64956ad895413d4d9267ae4b3a6045`),
is the vendor's own conversion under the same license. No reference code was
imported for this family: its architecture was read from
`transformers` v5.6.2 (Apache-2.0) and nothing was copied from it.

Downloaded weights remain a separate work governed by their source terms. Do
not redistribute weights as part of Mference releases.

## Swift package graph

The following table covers the complete graph reported by
`swift package show-dependencies` from the checked-in
[`Package.resolved`](Package.resolved). Exact revisions are recorded there.

| Package | Version | License in locked checkout |
| --- | --- | --- |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache-2.0 |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | 2.3.6 | Apache-2.0 |
| [swift-huggingface](https://github.com/huggingface/swift-huggingface) | 0.9.0 | Apache-2.0 |
| [EventSource](https://github.com/mattt/EventSource) | 1.4.1 | MIT |
| [swift-nio](https://github.com/apple/swift-nio) | 2.99.0 | Apache-2.0; upstream NOTICE applies |
| [swift-atomics](https://github.com/apple/swift-atomics) | 1.3.0 | Apache-2.0 with Runtime Library Exception |
| [swift-collections](https://github.com/apple/swift-collections) | 1.5.1 | Apache-2.0 with Runtime Library Exception |
| [swift-system](https://github.com/apple/swift-system) | 1.6.4 | Apache-2.0 with Runtime Library Exception |
| [swift-crypto](https://github.com/apple/swift-crypto) | 4.5.0 | Apache-2.0; upstream NOTICE applies |
| [swift-asn1](https://github.com/apple/swift-asn1) | 1.7.0 | Apache-2.0; upstream NOTICE applies |
| [yyjson](https://github.com/ibireme/yyjson) | 0.12.0 | MIT |

No copyleft or custom non-commercial license was found in this resolved graph.
The dependency license files remain authoritative. For binary distribution,
collect their license and NOTICE files from the exact revisions in
`Package.resolved`; do not treat this summary as a substitute for that bundle.
