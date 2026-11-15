import CryptoKit
import Foundation
import Testing
@testable import MferenceRepackCore

/// A tiny qwen4exp GGUF: 4 layers (layer 3 full attention), 2 experts of
/// hidden 256 / FF 256, mixed ggml types per layer like the real UD-IQ3_XXS
/// file, an IQ4_NL n-gram table and gpt2 tokenizer metadata.
private struct TinyQwen4ExpGGUF {
    static let hidden: UInt64 = 256, ff: UInt64 = 256, experts: UInt64 = 2, layers = 4, vocab = 8

    /// (name, ne, type) in file order.
    static func tensorSpecs() -> [(String, [UInt64], GGMLType)] {
        var t: [(String, [UInt64], GGMLType)] = [
            ("token_embd.weight", [hidden, UInt64(vocab)], .q8_0),
            ("output.weight", [hidden, UInt64(vocab)], .q6_K),
            ("per_layer_token_embd.weight", [160, 8], .iq4_nl),
        ]
        for l in 0..<layers {
            let (gu, down): (GGMLType, GGMLType) = switch l {
            case 1: (.iq2_s, .q2_0)
            case 2: (.iq3_xxs, .iq4_nl)
            case 3: (.iq4_xs, .iq4_nl)
            default: (.iq2_s, .iq4_nl)
            }
            t.append(("blk.\(l).ffn_gate_exps.weight", [hidden, ff, experts], gu))
            t.append(("blk.\(l).ffn_up_exps.weight", [hidden, ff, experts], gu))
            t.append(("blk.\(l).ffn_down_exps.weight", [ff, hidden, experts], down))
            t.append(("blk.\(l).ffn_gate_inp.weight", [hidden, experts], .f32))
        }
        t.append(("blk.3.attn_q.weight", [hidden, 2 * 2 * 64], .q6_K))
        t.append(("blk.3.indexer.k_proj.weight", [hidden, 32], .bf16))
        return t
    }

    static func size(_ ne: [UInt64], _ type: GGMLType) -> UInt64 {
        type.rowBytes(elements: ne[0])! * ne.dropFirst().reduce(1, *)
    }

    static func file() -> (Data, [String: (offset: Int, size: Int)]) {
        var g = SyntheticGGUF()
        func u32(_ k: String, _ v: UInt32) { g.addU32(k, v) }
        func f32(_ k: String, _ v: Float) { g.kv.append((k, 6, SyntheticGGUF.le(v.bitPattern))) }
        func i32s(_ k: String, _ v: [Int32]) {
            g.kv.append((k, 9, SyntheticGGUF.le(UInt32(5)) + SyntheticGGUF.le(UInt64(v.count))
                         + v.flatMap { SyntheticGGUF.le($0) }))
        }
        func boolean(_ k: String, _ v: Bool) { g.kv.append((k, 7, [v ? 1 : 0])) }
        g.addString("general.architecture", "qwen4exp")
        g.kv.append(("split.no", 2, SyntheticGGUF.le(UInt16(0))))
        g.kv.append(("split.count", 2, SyntheticGGUF.le(UInt16(1))))
        g.kv.append(("split.tensors.count", 5, SyntheticGGUF.le(Int32(tensorSpecs().count))))
        u32("qwen4exp.block_count", UInt32(layers))
        u32("qwen4exp.embedding_length", UInt32(hidden))
        u32("qwen4exp.attention.head_count", 2)
        u32("qwen4exp.attention.head_count_kv", 1)
        u32("qwen4exp.attention.key_length", 64)
        u32("qwen4exp.attention.value_length", 64)
        u32("qwen4exp.rope.dimension_count", 16)
        f32("qwen4exp.rope.freq_base", 10_000_000)
        u32("qwen4exp.expert_count", UInt32(experts))
        u32("qwen4exp.expert_used_count", 1)
        u32("qwen4exp.expert_feed_forward_length", UInt32(ff))
        u32("qwen4exp.expert_shared_feed_forward_length", UInt32(ff))
        u32("qwen4exp.full_attention_interval", 4)
        i32s("qwen4exp.attention.compress_ratios", [0, 0, 0, 4])
        u32("qwen4exp.ssm.conv_kernel", 4)
        u32("qwen4exp.ssm.state_size", 32)
        u32("qwen4exp.ssm.group_count", 1)
        u32("qwen4exp.ssm.time_step_rank", 2)
        u32("qwen4exp.ssm.inner_size", 128)
        u32("qwen4exp.hyper_connection.count", 4)
        u32("qwen4exp.hyper_connection.low_rank", 32)
        u32("qwen4exp.attention.indexer.head_count", 2)
        u32("qwen4exp.attention.indexer.key_length", 32)
        u32("qwen4exp.attention.indexer.top_k", 16)
        i32s("qwen4exp.ple.layers", [1])
        u32("qwen4exp.ple.conv_kernel", 4)
        u32("qwen4exp.ple.ngram_size", 2)
        u32("qwen4exp.ple.heads_per_ngram", 2)
        u32("qwen4exp.ple.eos_token_id", 5)
        g.addI64Array("qwen4exp.ple.layer_multipliers", [35_184_372_088_831, 7])
        g.addI64Array("qwen4exp.ple.head_offsets", [0, 3])
        g.addI64Array("qwen4exp.ple.head_vocab_sizes", [3, 4])
        g.addString("tokenizer.ggml.model", "gpt2")
        g.addString("tokenizer.ggml.pre", "qwen35")
        g.addStringArray("tokenizer.ggml.tokens", ["Ġ", "a", "b", "Ġa", "[PAD4]", "<|endoftext|>", "<|im_end|>", "<think>"])
        i32s("tokenizer.ggml.token_type", [1, 1, 1, 1, 5, 3, 3, 4])
        g.addStringArray("tokenizer.ggml.merges", ["Ġ a"])
        u32("tokenizer.ggml.bos_token_id", 5)
        u32("tokenizer.ggml.eos_token_id", 6)
        u32("tokenizer.ggml.padding_token_id", 5)
        boolean("tokenizer.ggml.add_bos_token", false)
        g.addString("tokenizer.chat_template", "{% for m in messages %}{{ m.content }}{% endfor %}")

        var offset: UInt64 = 0
        var rel: [String: (UInt64, UInt64)] = [:]
        for (name, ne, type) in tensorSpecs() {
            let s = size(ne, type)
            g.tensors.append((name, ne, type.rawValue, offset))
            rel[name] = (offset, s)
            offset = (offset + s + 31) / 32 * 32
        }
        var header = g.header()
        while header.count % 32 != 0 { header.append(0) }
        // Deterministic non-zero payload so byte identity is meaningful.
        var payload = [UInt8](repeating: 0, count: Int(offset))
        var x: UInt32 = 0x9E37_79B9
        for i in payload.indices { x ^= x << 13; x ^= x >> 17; x ^= x << 5; payload[i] = UInt8(truncatingIfNeeded: x) }
        let abs = rel.mapValues { (offset: header.count + Int($0.0), size: Int($0.1)) }
        return (Data(header + payload), abs)
    }
}

/// Shares the serialized `RemotePayloadCopyTests` suite because it touches
/// `FakeHFURLProtocol` statics.
extension RemotePayloadCopyTests {
    @Test func ggufInstallCopiesEveryTensorVerbatimWithPerLayerStrides() async throws {
        resetFakeHF()
        let (file, tensors) = TinyQwen4ExpGGUF.file()
        FakeHFURLProtocol.files["model.gguf"] = file
        let sha = SHA256.hash(data: file).map { String(format: "%02x", $0) }.joined()
        FakeHFURLProtocol.etagOverrides["model.gguf"] = "\"\(sha)\""
        let pin = GGUFSourcePin(commit: FakeHFURLProtocol.commit,
                                shards: [.init(path: "model.gguf", sha256: sha, size: UInt64(file.count))])
        let root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mference-gguf-install-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let out = (root as NSString).appendingPathComponent("tiny.gturbo")
        var options = remoteOptions(outputDir: out, session: fakeHFSession(), repoID: "owner/gguf",
                                    revision: FakeHFURLProtocol.commit)
        options.ggufPin = pin
        let result = try await RemoteStreamingRepacker(options: options).run()
        #expect(!result.dryRun)

        func read(_ rel: String) throws -> Data {
            try Data(contentsOf: URL(fileURLWithPath: (out as NSString).appendingPathComponent(rel)))
        }
        func source(_ name: String) throws -> Data {
            let t = try #require(tensors[name])
            return file.subdata(in: t.offset..<(t.offset + t.size))
        }

        // Resident: every tensor's bytes at its planned offset, unchanged.
        let weights = try read("model_weights.bin")
        for entry in result.plan.resident.entries {
            let o = Int(entry.fileOffset)
            #expect(weights.subdata(in: o..<(o + Int(entry.sizeBytes))) == (try source(entry.name)), "\(entry.name)")
        }

        // Experts: per-layer strides; expert 1's gate slice equals the source slice.
        let strides = result.plan.layers.map(\.expertStride)
        #expect(Set(strides).count > 1)
        for layer in result.plan.layers {
            let blob = try read(String(format: "packed_experts/layer_%02d.bin", layer.layerIndex))
            #expect(blob.count == Int(layer.fileSize))
            for slice in layer.subTensors {
                let src = try source("blk.\(layer.layerIndex).ffn_\(slice.role)_exps.weight")
                let per = Int(slice.sizeInExpertBlob)
                let dst = Int(layer.expertStride) + Int(slice.offsetInExpertBlob)
                #expect(blob.subdata(in: dst..<(dst + per)) == src.subdata(in: per..<(2 * per)))
            }
        }

        // PLE: IQ4_NL rows verbatim at the head of the first block.
        let ple = try read("ple/layer_01_ngram_rows.bin")
        #expect(ple.prefix(720) == (try source("per_layer_token_embd.weight")))

        // Manifest: GGUF source block, per-layer strides, gguf quant scheme, hash constants.
        let manifest = try #require(try JSONSerialization.jsonObject(with: read("manifest.json")) as? [String: Any])
        let src = try #require(manifest["source"] as? [String: Any])
        #expect(src["kind"] as? String == "gguf" && src["pinFingerprint"] as? String == pin.fingerprint)
        #expect((manifest["expertStrides"] as? [NSNumber])?.map(\.uint64Value) == strides)
        let quant = try #require(manifest["quant"] as? [String: Any])
        #expect((quant["routedExpert"] as? [String: Any])?["scheme"] as? String == "gguf")
        #expect((manifest["arch"] as? [String: Any])?["family"] as? String == "qwen38flashnext")
        let pool = try #require(((manifest["plePool"] as? [String: Any])?["layers"] as? [[String: Any]])?.first)
        #expect(pool["storage"] as? String == "iq4nl")
        #expect(((pool["hash"] as? [String: Any])?["headVocabSizes"] as? [NSNumber])?.map(\.intValue) == [3, 4])

        // Layout and tokenizer sidecars are recorded outputs.
        let layout = try #require(try JSONSerialization.jsonObject(with: read("packed_experts/layout.json")) as? [String: Any])
        let l1 = try #require((layout["layers"] as? [[String: Any]])?[1])
        #expect((l1["expertStride"] as? NSNumber)?.uint64Value == strides[1])
        let tok = try #require(try JSONSerialization.jsonObject(with: read("tokenizer/tokenizer.json")) as? [String: Any])
        #expect((tok["model"] as? [String: Any])?["type"] as? String == "BPE")
        let files = try #require(manifest["files"] as? [String: Any])
        for rel in ["model_weights.bin", "packed_experts/layout.json", "ple/layer_01_ngram_rows.bin",
                    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja"] {
            #expect(files[rel] != nil, "\(rel) not recorded")
        }
        #expect(FileManager.default.fileExists(atPath: (out as NSString).appendingPathComponent("verified-install.json")))
    }
}
