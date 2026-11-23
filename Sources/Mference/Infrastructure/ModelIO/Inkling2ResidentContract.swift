import Foundation

/// The resident tensors of an `inkling2Loop` install: every name the family's
/// accessors (`Model+Inkling2.swift`) read, its row-major shape (the GGUF's
/// `ne` reversed), and the formats it may be stored in. `Model.load` checks
/// the install's resident index against it, so an install with a missing,
/// misshapen, mistyped or unexpected tensor fails at load by name rather than
/// in a kernel.
///
/// Shapes follow from the baseline (`ArchConfig`). Each tensor has exactly
/// one format, the one the shipped file stores and the family's kernels read
/// in place (they hard-code it): Q8_0 for the attention projections, the
/// embedding and the head; BF16 for `attn_sink_proj` and the
/// dynamic-convolution projections; F32 for everything else. A requantized
/// file with another format is refused at load rather than misread.
///
/// Every tensor also starts on a 256-byte boundary, as the installer writes
/// it, so it meets every kernel's offset alignment (Q8_0 2, BF16 8, F32 16).
public enum Inkling2ResidentContract {
    public struct Tensor: Sendable, Equatable {
        public let name: String
        public let shape: [Int]
        public let formats: Set<GGMLBlockFormat>
    }

    static let q8: Set<GGMLBlockFormat> = [.q8_0]
    static let bf16: Set<GGMLBlockFormat> = [.bf16]
    static let f32: Set<GGMLBlockFormat> = [.f32]
    /// The installer's resident tensor alignment.
    public static let offsetAlignment: UInt64 = 256

    /// Every resident tensor the family carries, in no particular order.
    public static func expectedTensors(config c: ArchConfig) -> [Tensor] {
        let h = c.hiddenSize, heads = c.numHeads
        let dRel = c.relativePosition.dRel, conv = c.sconvKernelSize, experts = c.numExperts
        let ink = c.inkling2
        var out: [Tensor] = [
            .init(name: "token_embd.weight", shape: [c.vocabSize, h], formats: q8),
            .init(name: "token_embd_norm.weight", shape: [h], formats: f32),
            .init(name: "output.weight", shape: [c.vocabSize, h], formats: q8),
            .init(name: "output_norm.weight", shape: [h], formats: f32),
            .init(name: "output_res_query.weight", shape: [h], formats: f32),
        ]
        for L in 0..<c.numLayers {
            let full = c.layerIsFull(L)
            let kv = full ? c.numFullKVHeads : c.numKVHeads, hd = full ? c.fullHeadDim : c.headDim
            func add(_ suffix: String, _ shape: [Int], _ formats: Set<GGMLBlockFormat>) {
                out.append(.init(name: "blk.\(L).\(suffix)", shape: shape, formats: formats))
            }
            add("attn_q.weight", [heads * hd, h], q8)
            add("attn_k.weight", [kv * hd, h], q8)
            add("attn_v.weight", [kv * hd, h], q8)
            add("attn_output.weight", [h, heads * hd], q8)
            add("attn_gate.weight", [heads * hd, h], q8)
            add("attn_q_clip.weight", [heads, hd], f32)
            add("attn_sink_proj.weight", [heads, h], bf16)
            add("attn_rel.weight", [heads * dRel, h], q8)
            add("attn_rel_bias.weight", [full ? c.relativePosition.extent : c.slidingWindow, dRel], f32)
            add("attn_rel_clip.weight", [heads, dRel], f32)
            add("attn_k_conv.weight", [kv * hd, conv], f32)
            add("attn_v_conv.weight", [kv * hd, conv], f32)
            add("attn_post_conv.weight", [h, conv], f32)
            add("ffn_post_conv.weight", [h, conv], f32)
            if full {
                add("attn_q_inp_norm.weight", [h], f32)
                add("attn_k_inp_norm.weight", [h], f32)
                add("attn_v_inp_norm.weight", [h], f32)
                add("attn_k_conv_proj.weight", [conv * kv, h], bf16)
                add("attn_v_conv_proj.weight", [conv * kv, h], bf16)
            } else {
                add("attn_norm.weight", [h], f32)
            }
            if L > 0 { add("attn_res_query.weight", full ? [3, h] : [h], f32) }
            add("ffn_res_query.weight", [h], f32)
            add("ffn_norm.weight", [h], f32)
            add("ffn_gate_inp.weight", [experts, h], f32)
            if c.routerGateBias { add("exp_probs_b.bias", [experts], f32) }
            if ink.routedExpertFormats.indices.contains(L), ink.routedExpertFormats[L] == GGMLBlockFormat.nvfp4.name {
                add("ffn_gate_up_exps.scale", [experts], f32)
                add("ffn_down_exps.scale", [experts], f32)
            }
            if ink.layerIsLooped(L) {
                add("loop_msg_query.weight", [Inkling2MessageGate.allCases.count, h], f32)
                for gate in Inkling2MessageGate.allCases {
                    add("loop_\(gate.ggufInfix)_ctrl_down.weight", [ink.loopGateRank, 2 * h], f32)
                    add("loop_\(gate.ggufInfix)_ctrl_gain.weight", [h, ink.loopGateRank], f32)
                    add("loop_\(gate.ggufInfix)_ctrl_forget.weight", [h, ink.loopGateRank], f32)
                }
            }
        }
        return out
    }

    /// The family's stacked projections read a layer's dense tensors from one
    /// weight buffer, so the resident region must map as one `MTLBuffer`
    /// (`ResidentBuffer` makes one whenever it fits `maxBufferLength`; the
    /// shipped file's region is about 1.82 GB).
    static func validateOneBuffer(chunkCount: Int, residentSize: UInt64) throws {
        guard chunkCount == 1 else {
            throw ModelError.indexCorrupt(detail: "inkling2 resident: \(residentSize) bytes map as "
                + "\(chunkCount) Metal buffers; the family's kernels need one")
        }
    }

    /// Refuses an index that does not hold exactly the expected tensors, each
    /// with its shape, its format, the byte size that shape and format imply,
    /// and a 256-byte-aligned offset.
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
                throw invalid("\(t.name) has dtype \(entry.dtype), expected "
                    + t.formats.map(\.name).sorted().joined(separator: ", "))
            }
            guard entry.fileOffset % offsetAlignment == 0 else {
                throw invalid("\(t.name) starts at \(entry.fileOffset), not a multiple of \(offsetAlignment)")
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
