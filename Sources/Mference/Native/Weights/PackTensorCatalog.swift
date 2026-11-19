import Foundation

extension Native {
    /// The tensors a llama.cpp `qwen35` model must carry, by GGUF name, with
    /// the shapes `Qwen35Config` implies (ggml `ne` order: `[inputs, outputs]`
    /// for matrices). The pack store checks a manifest against it at load, so
    /// a pack whose tensors do not match its config fails before any kernel
    /// reads it.
    ///
    /// Names (llama.cpp `LLM_ARCH_QWEN35`):
    /// - global: `token_embd.weight`, `output_norm.weight`, and
    ///   `output.weight` unless the embeddings are tied;
    /// - every block: `attn_norm`, `post_attention_norm`, `ffn_gate`,
    ///   `ffn_up`, `ffn_down`;
    /// - full-attention blocks: `attn_q` (query and gate, `2 * heads * key`
    ///   outputs), `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`,
    ///   `attn_k_norm`;
    /// - Gated DeltaNet blocks: `attn_qkv` (the conv input), `attn_gate`,
    ///   `ssm_alpha`, `ssm_beta`, `ssm_out`, `ssm_conv1d`, `ssm_dt.bias`,
    ///   `ssm_a`, `ssm_norm`;
    /// - `nextn` (MTP) blocks `blockCount ..< blockCount + nextnPredictLayers`:
    ///   a full-attention block plus `nextn.eh_proj`, `nextn.enorm`,
    ///   `nextn.hnorm`; `nextn.shared_head_norm`, `nextn.embed_tokens` and
    ///   `nextn.shared_head_head` are optional (llama.cpp falls back to the
    ///   main model's).
    public enum Qwen35TensorCatalog {
        public struct Entry: Equatable, Sendable {
            public var name: String
            public var shape: [Int]
            public var required: Bool
        }

        /// Every tensor `config` implies, required ones and optional ones.
        public static func entries(for c: Qwen35Config) -> [Entry] {
            let h = c.hiddenSize
            var out: [Entry] = [
                Entry(name: "token_embd.weight", shape: [h, c.vocabSize], required: true),
                Entry(name: "output_norm.weight", shape: [h], required: true),
            ]
            if !c.tiedEmbeddings {
                out.append(Entry(name: "output.weight", shape: [h, c.vocabSize], required: true))
            }
            func block(_ i: Int, fullAttention: Bool) {
                func add(_ suffix: String, _ shape: [Int], required: Bool = true) {
                    out.append(Entry(name: "blk.\(i).\(suffix)", shape: shape, required: required))
                }
                add("attn_norm.weight", [h])
                add("post_attention_norm.weight", [h])
                add("ffn_gate.weight", [h, c.ffnSize])
                add("ffn_up.weight", [h, c.ffnSize])
                add("ffn_down.weight", [c.ffnSize, h])
                if fullAttention {
                    add("attn_q.weight", [h, 2 * c.headCount * c.keyLength])
                    add("attn_k.weight", [h, c.kvHeadCount * c.keyLength])
                    add("attn_v.weight", [h, c.kvHeadCount * c.valueLength])
                    add("attn_output.weight", [c.headCount * c.valueLength, h])
                    add("attn_q_norm.weight", [c.keyLength])
                    add("attn_k_norm.weight", [c.keyLength])
                } else {
                    add("attn_qkv.weight", [h, c.ssmConvChannels])
                    add("attn_gate.weight", [h, c.ssmInnerSize])
                    add("ssm_alpha.weight", [h, c.ssmTimeStepRank])
                    add("ssm_beta.weight", [h, c.ssmTimeStepRank])
                    add("ssm_out.weight", [c.ssmInnerSize, h])
                    add("ssm_conv1d.weight", [c.ssmConvKernel, c.ssmConvChannels])
                    add("ssm_dt.bias", [c.ssmTimeStepRank])
                    add("ssm_a", [c.ssmTimeStepRank])
                    add("ssm_norm.weight", [c.ssmStateSize])
                }
            }
            for i in 0..<c.blockCount {
                block(i, fullAttention: c.layerKind(i) == .fullAttention)
            }
            for j in 0..<c.nextnPredictLayers {
                let i = c.blockCount + j
                block(i, fullAttention: true)
                out.append(Entry(name: "blk.\(i).nextn.eh_proj.weight", shape: [2 * h, h], required: true))
                out.append(Entry(name: "blk.\(i).nextn.enorm.weight", shape: [h], required: true))
                out.append(Entry(name: "blk.\(i).nextn.hnorm.weight", shape: [h], required: true))
                out.append(Entry(name: "blk.\(i).nextn.shared_head_norm.weight", shape: [h], required: false))
                out.append(Entry(name: "blk.\(i).nextn.embed_tokens.weight", shape: [h, c.vocabSize], required: false))
                out.append(Entry(name: "blk.\(i).nextn.shared_head_head.weight", shape: [h, c.vocabSize],
                                 required: false))
            }
            return out
        }

        /// Checks that every required tensor is present, that every catalogued
        /// tensor present has the implied shape, that `tiedEmbeddings` agrees
        /// with the presence of `output.weight`, and that `vocabSize` is the
        /// embedding's row count. Tensors outside the catalog are allowed.
        public static func validate(_ manifest: PackManifest) throws {
            let c = manifest.config
            guard c.blockCount > 0, c.fullAttentionInterval > 0, c.hiddenSize > 0, c.vocabSize > 0,
                  c.nextnPredictLayers >= 0 else {
                throw NativeError.invalidShape("qwen35 config \(c)")
            }
            var shapes: [String: [Int]] = [:]
            for t in manifest.tensors { shapes[t.name] = t.shape }
            if c.tiedEmbeddings == (shapes["output.weight"] != nil) {
                throw NativeError.integrity("tiedEmbeddings is \(c.tiedEmbeddings) but output.weight is "
                                            + (c.tiedEmbeddings ? "present" : "absent"))
            }
            for e in entries(for: c) {
                guard let shape = shapes[e.name] else {
                    if e.required { throw NativeError.missing("tensor \(e.name)") }
                    continue
                }
                guard shape == e.shape else {
                    throw NativeError.invalidShape("\(e.name): shape \(shape), config implies \(e.shape)")
                }
            }
        }
    }
}
