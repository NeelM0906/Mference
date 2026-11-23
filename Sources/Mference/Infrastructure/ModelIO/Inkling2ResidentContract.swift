import Foundation

/// The resident tensors of an `inkling2Loop` install: every name the family's
/// accessors (`Model+Inkling2.swift`) read, its row-major shape (the GGUF's
/// `ne` reversed), and the formats it may be stored in. `Model.load` checks
/// the install's resident index against it, so an install with a missing,
/// misshapen, mistyped or unexpected tensor fails at load by name rather than
/// in a kernel.
///
/// Shapes follow from the baseline (`ArchConfig`); formats are what the
/// family's kernels read: Q8_0 or a float type for the projection matrices,
/// a float type for everything else. The shipped file stores the matrices as
/// Q8_0, `attn_sink_proj` and the dynamic-convolution projections as BF16,
/// and everything else as F32.
public enum Inkling2ResidentContract {
    public struct Tensor: Sendable, Equatable {
        public let name: String
        public let shape: [Int]
        public let formats: Set<GGMLBlockFormat>
    }

    static let matrixFormats: Set<GGMLBlockFormat> = [.q8_0, .bf16, .f16, .f32]
    static let floatFormats: Set<GGMLBlockFormat> = [.f32, .bf16, .f16]
    /// Read on the CPU or by F32 kernels: the router, its bias, the loop
    /// gates and the per-expert factors.
    static let f32Only: Set<GGMLBlockFormat> = [.f32]

    /// Every resident tensor the family carries, in no particular order.
    public static func expectedTensors(config c: ArchConfig) -> [Tensor] {
        let h = c.hiddenSize, heads = c.numHeads, kv = c.numKVHeads, hd = c.headDim
        let dRel = c.relativePosition.dRel, conv = c.sconvKernelSize, experts = c.numExperts
        let ink = c.inkling2
        var out: [Tensor] = [
            .init(name: "token_embd.weight", shape: [c.vocabSize, h], formats: matrixFormats),
            .init(name: "token_embd_norm.weight", shape: [h], formats: floatFormats),
            .init(name: "output.weight", shape: [c.vocabSize, h], formats: matrixFormats),
            .init(name: "output_norm.weight", shape: [h], formats: floatFormats),
            .init(name: "output_res_query.weight", shape: [h], formats: floatFormats),
        ]
        for L in 0..<c.numLayers {
            let full = c.layerIsFull(L)
            func add(_ suffix: String, _ shape: [Int], _ formats: Set<GGMLBlockFormat>) {
                out.append(.init(name: "blk.\(L).\(suffix)", shape: shape, formats: formats))
            }
            add("attn_q.weight", [heads * hd, h], matrixFormats)
            add("attn_k.weight", [kv * hd, h], matrixFormats)
            add("attn_v.weight", [kv * hd, h], matrixFormats)
            add("attn_output.weight", [h, heads * hd], matrixFormats)
            add("attn_gate.weight", [heads * hd, h], matrixFormats)
            add("attn_q_clip.weight", [heads, hd], floatFormats)
            add("attn_sink_proj.weight", [heads, h], matrixFormats)
            add("attn_rel.weight", [heads * dRel, h], matrixFormats)
            add("attn_rel_bias.weight", [full ? c.relativePosition.extent : c.slidingWindow, dRel], floatFormats)
            add("attn_rel_clip.weight", [heads, dRel], floatFormats)
            add("attn_k_conv.weight", [kv * hd, conv], floatFormats)
            add("attn_v_conv.weight", [kv * hd, conv], floatFormats)
            add("attn_post_conv.weight", [h, conv], floatFormats)
            add("ffn_post_conv.weight", [h, conv], floatFormats)
            if full {
                add("attn_q_inp_norm.weight", [h], floatFormats)
                add("attn_k_inp_norm.weight", [h], floatFormats)
                add("attn_v_inp_norm.weight", [h], floatFormats)
                add("attn_k_conv_proj.weight", [conv * kv, h], matrixFormats)
                add("attn_v_conv_proj.weight", [conv * kv, h], matrixFormats)
            } else {
                add("attn_norm.weight", [h], floatFormats)
            }
            if L > 0 { add("attn_res_query.weight", full ? [3, h] : [h], floatFormats) }
            add("ffn_res_query.weight", [h], floatFormats)
            add("ffn_norm.weight", [h], floatFormats)
            add("ffn_gate_inp.weight", [experts, h], f32Only)
            if c.routerGateBias { add("exp_probs_b.bias", [experts], f32Only) }
            if ink.routedExpertFormats.indices.contains(L), ink.routedExpertFormats[L] == GGMLBlockFormat.nvfp4.name {
                add("ffn_gate_up_exps.scale", [experts], f32Only)
                add("ffn_down_exps.scale", [experts], f32Only)
            }
            if ink.layerIsLooped(L) {
                add("loop_msg_query.weight", [Inkling2MessageGate.allCases.count, h], f32Only)
                for gate in Inkling2MessageGate.allCases {
                    add("loop_\(gate.ggufInfix)_ctrl_down.weight", [ink.loopGateRank, 2 * h], f32Only)
                    add("loop_\(gate.ggufInfix)_ctrl_gain.weight", [h, ink.loopGateRank], f32Only)
                    add("loop_\(gate.ggufInfix)_ctrl_forget.weight", [h, ink.loopGateRank], f32Only)
                }
            }
        }
        return out
    }

    /// Refuses an index that does not hold exactly the expected tensors, each
    /// with its shape, an allowed format, and the byte size that shape and
    /// format imply.
    static func validate(index: ResidentIndex, config: ArchConfig) throws {
        func invalid(_ detail: String) -> ModelError { .indexCorrupt(detail: "inkling2 resident: " + detail) }
        let expected = expectedTensors(config: config)
        for t in expected {
            guard let entry = index.entries[t.name] else { throw ModelError.tensorNotFound(name: t.name) }
            let stored = [entry.shape.0, entry.shape.1, entry.shape.2, entry.shape.3].map(Int.init)
            let shape = Array(stored.prefix { $0 != 0 })
            guard shape == t.shape, stored.dropFirst(shape.count).allSatisfy({ $0 == 0 }) else {
                throw invalid("\(t.name) has shape \(shape), expected \(t.shape)")
            }
            guard let format = GGMLBlockFormat(residentDtype: entry.dtype), t.formats.contains(format) else {
                throw invalid("\(t.name) has dtype \(entry.dtype), expected one of "
                    + t.formats.map(\.name).sorted().joined(separator: ", "))
            }
            guard let rowBytes = format.rowBytes(elements: t.shape.last!),
                  entry.sizeBytes == UInt64(t.shape.dropLast().reduce(rowBytes, *)) else {
                throw ModelError.tensorSizeMismatch(
                    name: t.name,
                    expected: UInt64(t.shape.dropLast().reduce(format.rowBytes(elements: t.shape.last!) ?? 0, *)),
                    actual: entry.sizeBytes)
            }
        }
        let names = Set(expected.map(\.name))
        if let stray = index.entries.keys.filter({ !names.contains($0) }).sorted().first {
            throw invalid("unexpected tensor \(stray)")
        }
    }
}
