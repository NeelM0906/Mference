import Foundation
@testable import Mference

/// A tiny synthetic `qwen35` GGUF (random weights, the `RefTest.tiny` shapes
/// and a byte-level vocabulary), written to disk so the pack writer, the
/// store, the runner and the scheduler can run end to end with no model.
enum RefSynthetic {
    struct Tensor {
        var name: String
        var type: Native.GGMLType
        var shape: [Int]
        var bytes: [UInt8]
    }

    /// Minimal GGUF writer (v3 by default; `version`, `alignment` and raw
    /// tensor offsets can be overridden to build hostile files).
    struct GGUFBuilder {
        var kv: [(String, UInt32, [UInt8])] = []
        var tensors: [Tensor] = []
        var version: UInt32 = 3
        var alignment = 32
        /// Tensor index → offset written in its info (data is still laid out
        /// at the natural offsets).
        var offsetOverride: [Int: UInt64] = [:]
        /// Tensor index → raw `ne` written in its info.
        var dimsOverride: [Int: [UInt64]] = [:]

        static func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }
        static func str(_ s: String) -> [UInt8] { le(UInt64(s.utf8.count)) + Array(s.utf8) }

        mutating func u32(_ k: String, _ v: UInt32) { kv.append((k, 4, Self.le(v))) }
        mutating func f32(_ k: String, _ v: Float) { kv.append((k, 6, Self.le(v.bitPattern))) }
        mutating func string(_ k: String, _ v: String) { kv.append((k, 8, Self.str(v))) }
        mutating func strings(_ k: String, _ v: [String]) {
            kv.append((k, 9, Self.le(UInt32(8)) + Self.le(UInt64(v.count)) + v.flatMap(Self.str)))
        }
        mutating func i32s(_ k: String, _ v: [Int32]) {
            kv.append((k, 9, Self.le(UInt32(5)) + Self.le(UInt64(v.count)) + v.flatMap { Self.le($0) }))
        }

        func data() -> Data {
            var out = Array("GGUF".utf8) + Self.le(version) + Self.le(UInt64(tensors.count)) + Self.le(UInt64(kv.count))
            for (k, t, v) in kv { out += Self.str(k) + Self.le(t) + v }
            var offset = 0
            var offsets: [Int] = []
            for t in tensors {
                offsets.append(offset)
                offset = (offset + t.bytes.count + alignment - 1) / alignment * alignment
            }
            for (i, (t, o)) in zip(tensors, offsets).enumerated() {
                let dims = dimsOverride[i] ?? t.shape.map { UInt64($0) }
                out += Self.str(t.name) + Self.le(UInt32(dims.count))
                for d in dims { out += Self.le(d) }
                out += Self.le(t.type.rawValue) + Self.le(offsetOverride[i] ?? UInt64(o))
            }
            while out.count % alignment != 0 { out.append(0) }
            let base = out.count
            for (t, o) in zip(tensors, offsets) {
                while out.count < base + o { out.append(0) }
                out += t.bytes
            }
            return Data(out)
        }
    }

    static let vocabulary: (tokens: [String], types: [Int32], merges: [String]) = {
        var tokens = (0..<256).map { String(decoding: Native.ReferenceUnicode.byteToUTF8[$0], as: UTF8.self) }
        let merges = ["h e", "l l", "he ll", "hell o", "Ġ w", "Ġw o"]
        tokens += merges.map { $0.replacingOccurrences(of: " ", with: "") }
        tokens += ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        while tokens.count < RefTest.tiny.vocabSize { tokens.append("<|reserved_\(tokens.count)|>") }
        var types = [Int32](repeating: 1, count: tokens.count)
        for i in 262..<tokens.count { types[i] = 3 }
        return (tokens, types, merges)
    }()

    /// Tensor types per role, a mix of every type Quick uses.
    static func tensors(seed: UInt64) -> [Tensor] {
        let c = RefTest.tiny
        var seed = seed
        func q(_ name: String, _ type: Native.GGMLType, _ cols: Int, _ rows: Int) -> Tensor {
            seed += 1
            let format = Native.QuantBlockFormat.of(type)!
            return Tensor(name: name, type: type, shape: [cols, rows],
                          bytes: RefTest.randomBlocks(type, blocks: rows * cols / format.elements, seed: seed))
        }
        func f(_ name: String, _ values: [Float], shape: [Int]? = nil) -> Tensor {
            Tensor(name: name, type: .f32, shape: shape ?? [values.count], bytes: values.withUnsafeBytes { Array($0) })
        }
        var rng = RefTest.SplitMix64(seed &* 31)
        func norm(_ n: Int) -> [Float] { (0..<n).map { _ in 1 + rng.unit() * 0.2 } }
        var out: [Tensor] = [
            q("token_embd.weight", .q6_K, c.hiddenSize, c.vocabSize),
            f("output_norm.weight", norm(c.hiddenSize)),
        ]
        for i in 0..<c.blockCount {
            let p = "blk.\(i)."
            out.append(f(p + "attn_norm.weight", norm(c.hiddenSize)))
            out.append(f(p + "post_attention_norm.weight", norm(c.hiddenSize)))
            out.append(q(p + "ffn_gate.weight", i % 2 == 0 ? .iq4_xs : .q4_K, c.hiddenSize, c.ffnSize))
            out.append(q(p + "ffn_up.weight", .q5_K, c.hiddenSize, c.ffnSize))
            out.append(q(p + "ffn_down.weight", i % 2 == 0 ? .q6_K : .q4_K, c.ffnSize, c.hiddenSize))
            if c.layerKind(i) == .fullAttention {
                out.append(q(p + "attn_q.weight", .q4_K, c.hiddenSize, c.headCount * c.keyLength * 2))
                out.append(q(p + "attn_k.weight", .q5_K, c.hiddenSize, c.kvHeadCount * c.keyLength))
                out.append(q(p + "attn_v.weight", .q6_K, c.hiddenSize, c.kvHeadCount * c.valueLength))
                out.append(q(p + "attn_output.weight", .q8_0, c.headCount * c.valueLength, c.hiddenSize))
                out.append(f(p + "attn_q_norm.weight", norm(c.keyLength)))
                out.append(f(p + "attn_k_norm.weight", norm(c.keyLength)))
            } else {
                out.append(q(p + "attn_qkv.weight", .q5_K, c.hiddenSize, c.ssmConvChannels))
                out.append(q(p + "attn_gate.weight", .q5_K, c.hiddenSize, c.ssmInnerSize))
                out.append(q(p + "ssm_alpha.weight", .f16, c.hiddenSize, c.ssmTimeStepRank))
                out.append(q(p + "ssm_beta.weight", .f16, c.hiddenSize, c.ssmTimeStepRank))
                out.append(q(p + "ssm_out.weight", .q8_0, c.ssmInnerSize, c.hiddenSize))
                out.append(f(p + "ssm_conv1d.weight", (0..<c.ssmConvChannels * c.ssmConvKernel).map { _ in rng.unit() * 0.5 },
                             shape: [c.ssmConvKernel, c.ssmConvChannels]))
                out.append(f(p + "ssm_dt.bias", (0..<c.ssmTimeStepRank).map { _ in rng.unit() }))
                out.append(f(p + "ssm_a", (0..<c.ssmTimeStepRank).map { _ in -abs(rng.unit()) - 0.05 }))
                out.append(f(p + "ssm_norm.weight", norm(c.ssmStateSize)))
            }
        }
        return out
    }

    static func gguf(seed: UInt64 = 1000) -> Data {
        let c = RefTest.tiny
        var b = GGUFBuilder()
        b.string("general.architecture", "qwen35")
        b.string("general.name", "tiny-reference")
        b.u32("qwen35.block_count", UInt32(c.blockCount))
        b.u32("qwen35.context_length", UInt32(c.contextLength))
        b.u32("qwen35.embedding_length", UInt32(c.hiddenSize))
        b.u32("qwen35.feed_forward_length", UInt32(c.ffnSize))
        b.u32("qwen35.attention.head_count", UInt32(c.headCount))
        b.u32("qwen35.attention.head_count_kv", UInt32(c.kvHeadCount))
        b.i32s("qwen35.rope.dimension_sections", c.ropeSections.map { Int32($0) })
        b.f32("qwen35.rope.freq_base", c.ropeFreqBase)
        b.f32("qwen35.attention.layer_norm_rms_epsilon", c.rmsNormEpsilon)
        b.u32("qwen35.attention.key_length", UInt32(c.keyLength))
        b.u32("qwen35.attention.value_length", UInt32(c.valueLength))
        b.u32("qwen35.ssm.conv_kernel", UInt32(c.ssmConvKernel))
        b.u32("qwen35.ssm.state_size", UInt32(c.ssmStateSize))
        b.u32("qwen35.ssm.group_count", UInt32(c.ssmGroupCount))
        b.u32("qwen35.ssm.time_step_rank", UInt32(c.ssmTimeStepRank))
        b.u32("qwen35.ssm.inner_size", UInt32(c.ssmInnerSize))
        b.u32("qwen35.full_attention_interval", UInt32(c.fullAttentionInterval))
        b.u32("qwen35.rope.dimension_count", UInt32(c.ropeDimensionCount))
        b.string("tokenizer.ggml.model", "gpt2")
        b.string("tokenizer.ggml.pre", "qwen35")
        b.strings("tokenizer.ggml.tokens", vocabulary.tokens)
        b.i32s("tokenizer.ggml.token_type", vocabulary.types)
        b.strings("tokenizer.ggml.merges", vocabulary.merges)
        b.u32("tokenizer.ggml.eos_token_id", 263)
        b.string("tokenizer.chat_template", "{% for m in messages %}<|im_start|>{{ m.content }}<|im_end|>{% endfor %}")
        b.tensors = tensors(seed: seed)
        return b.data()
    }

    /// A tiled layout for every quantized type that fits (F32 tensors are the
    /// ops' small weights, which PackLayoutTable v1 and the ops keep in .gguf).
    static func tiled(_ type: Native.GGMLType, _ rows: Int, _ blocks: Int) -> Native.TileLayout {
        guard type != .f32 else { return .gguf }
        let t = Native.TileLayout(rowsPerTile: 4, blocksPerTile: 2, planes: true)
        return t.fits(rows: rows, blocksPerRow: blocks) ? t : .gguf
    }

    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mference-ref-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
