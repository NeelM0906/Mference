import CryptoKit
import Foundation
import Metal
import Testing
@testable import MferenceRepackCore
@testable import Mference

/// Writes synthetic GGUF v3 files: metadata of every value type the reader
/// parses, a tensor table, and payloads at the file's alignment. Payload bytes
/// are a seeded pattern of their absolute file offset, so a test can prove an
/// install copied every byte unchanged.
struct GGUFFixture {
    enum Value {
        case u32(UInt32), f32(Float), bool(Bool), string(String)
        case u32s([UInt32]), i32s([Int32]), bools([Bool]), strings([String])
    }

    struct Tensor {
        let name: String
        let ne: [UInt64]
        let type: GGMLType

        var bytes: UInt64 { ne.dropFirst().reduce(type.rowBytes(elements: ne[0])!, *) }
    }

    var metadata: [(String, Value)] = []
    var tensors: [Tensor] = []
    let alignment: UInt64 = 32

    static func payloadByte(_ offset: UInt64) -> UInt8 {
        UInt8(truncatingIfNeeded: (offset &* 2_654_435_761) >> 13)
    }

    /// The header bytes, the data offset and every tensor's relative offset.
    func header() -> (bytes: Data, dataOffset: UInt64, relative: [UInt64]) {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func str(_ s: String) { u64(UInt64(s.utf8.count)); d.append(contentsOf: Array(s.utf8)) }
        d.append(contentsOf: Array("GGUF".utf8))
        u32(3); u64(UInt64(tensors.count)); u64(UInt64(metadata.count))
        for (k, v) in metadata {
            str(k)
            switch v {
            case .u32(let x): u32(4); u32(x)
            case .f32(let x): u32(6); u32(x.bitPattern)
            case .bool(let x): u32(7); d.append(x ? 1 : 0)
            case .string(let x): u32(8); str(x)
            case .u32s(let xs): u32(9); u32(4); u64(UInt64(xs.count)); xs.forEach(u32)
            case .i32s(let xs): u32(9); u32(5); u64(UInt64(xs.count)); xs.forEach { u32(UInt32(bitPattern: $0)) }
            case .bools(let xs): u32(9); u32(7); u64(UInt64(xs.count)); xs.forEach { d.append($0 ? 1 : 0) }
            case .strings(let xs): u32(9); u32(8); u64(UInt64(xs.count)); xs.forEach(str)
            }
        }
        var relative: [UInt64] = []
        var cursor: UInt64 = 0
        for t in tensors {
            cursor = (cursor + alignment - 1) / alignment * alignment
            relative.append(cursor)
            cursor += t.bytes
        }
        for (t, off) in zip(tensors, relative) {
            str(t.name); u32(UInt32(t.ne.count)); t.ne.forEach(u64); u32(t.type.rawValue); u64(off)
        }
        let dataOffset = (UInt64(d.count) + alignment - 1) / alignment * alignment
        d.append(Data(count: Int(dataOffset) - d.count))
        return (d, dataOffset, relative)
    }

    /// Header plus payloads; returns each tensor's absolute offset too.
    func encode() -> (data: Data, offsets: [String: UInt64]) {
        var (d, dataOffset, relative) = header()
        var offsets: [String: UInt64] = [:]
        for (t, off) in zip(tensors, relative) {
            let start = dataOffset + off
            if UInt64(d.count) < start { d.append(Data(count: Int(start) - d.count)) }
            offsets[t.name] = start
            d.append(contentsOf: (0..<t.bytes).map { Self.payloadByte(start + $0) })
        }
        return (d, offsets)
    }

    /// The file size the header describes (its last tensor's end).
    func describedFileSize() -> UInt64 {
        let (_, dataOffset, relative) = header()
        return dataOffset + (relative.last ?? 0) + (tensors.last?.bytes ?? 0)
    }
}

/// The Inkling 2 tensor table and metadata, at any shape: production
/// (`.production`) or a toy small enough to write and install in a test.
struct Inkling2Shape {
    var layers = 2, hidden = 128, heads = 2, kvHeads = 1, headDim = 64, width = 64
    var experts = 2, topK = 1, window = 16, dRel = 4, extent = 16, loopStart = 1, loopEnd = 2
    var passes = 2, gateRank = 4, block = 1, conv = 4
    var tokens = ["a", "b", "ab", "c", "<|end|>", "<|pad|>"]
    var specials: Set<Int> = [4, 5]
    var vocab = 8
    var formats: [GGMLType] = [.nvfp4, .q8_0]
    var full: Set<Int> = [1]

    static let production: Inkling2Shape = {
        var s = Inkling2Shape()
        s.layers = 24; s.hidden = 2304; s.heads = 24; s.kvHeads = 8; s.headDim = 128; s.width = 1152
        s.experts = 128; s.topK = 8; s.window = 1024; s.dRel = 16; s.extent = 1024
        s.loopStart = 6; s.loopEnd = 18; s.passes = 3; s.gateRank = 64; s.block = 6
        s.tokens = (0..<200_058).map { "t\($0)" }
        s.specials = Set(199_998..<200_058)
        s.vocab = 201_024
        s.formats = [GGMLType](repeating: .nvfp4, count: 20) + [GGMLType](repeating: .q8_0, count: 4)
        s.full = [5, 11, 17, 23]
        return s
    }()

    func fixture() -> GGUFFixture {
        let a = "inkling2_loop."
        let md: [(String, GGUFFixture.Value)] = [
            ("general.architecture", .string("inkling2_loop")),
            (a + "block_count", .u32(UInt32(layers))),
            (a + "context_length", .u32(1_048_576)),
            (a + "embedding_length", .u32(UInt32(hidden))),
            (a + "feed_forward_length", .u32(UInt32(width))),
            (a + "attention.head_count", .u32s([UInt32](repeating: UInt32(heads), count: layers))),
            (a + "attention.head_count_kv", .u32s([UInt32](repeating: UInt32(kvHeads), count: layers))),
            (a + "attention.layer_norm_rms_epsilon", .f32(1e-6)),
            (a + "expert_count", .u32(UInt32(experts))),
            (a + "expert_used_count", .u32(UInt32(topK))),
            (a + "attention.key_length", .u32(UInt32(headDim))),
            (a + "attention.value_length", .u32(UInt32(headDim))),
            (a + "loop.count", .u32(UInt32(passes))),
            (a + "loop.start", .u32(UInt32(loopStart))),
            (a + "loop.end", .u32(UInt32(loopEnd))),
            (a + "loop.gate_rank", .u32(UInt32(gateRank))),
            (a + "vocab_size", .u32(UInt32(vocab))),
            (a + "vocab_size_unpadded", .u32(UInt32(tokens.count))),
            (a + "expert_feed_forward_length", .u32(UInt32(width))),
            (a + "attention.key_length_swa", .u32(UInt32(headDim))),
            (a + "attention.value_length_swa", .u32(UInt32(headDim))),
            (a + "rope.dimension_count", .u32(0)),
            (a + "attention.sliding_window", .u32(UInt32(window))),
            (a + "attention.sliding_window_pattern", .bools((0..<layers).map { !full.contains($0) })),
            (a + "shortconv.l_cache", .u32(UInt32(conv))),
            (a + "expert_weights_scale", .f32(8)),
            (a + "activation.situ_beta", .f32(4)),
            (a + "activation.situ_linear_beta", .f32(25)),
            (a + "attn_res.block_size", .u32(UInt32(block))),
            (a + "final_logit_softcapping", .f32(30)),
            (a + "attention.relative_length", .u32(UInt32(dRel))),
            (a + "attention.relative_extent", .u32(UInt32(extent))),
            (a + "attention.qk_clip", .f32(Float(22.627416997969522))),
            (a + "attention.relative_clip", .f32(8)),
            (a + "attention.log_scale_floor", .u32(8192)),
            (a + "attention.log_scale_alpha", .f32(0.1)),
            ("tokenizer.ggml.model", .string("gpt2")),
            ("tokenizer.ggml.pre", .string("inkling2")),
            ("tokenizer.ggml.tokens", .strings(tokens)),
            ("tokenizer.ggml.token_type", .i32s(tokens.indices.map { specials.contains($0) ? 3 : 1 })),
            ("tokenizer.ggml.merges", .strings(layers == 24 ? [] : ["a b"])),
            ("tokenizer.ggml.eos_token_id", .u32(UInt32(specials.min()!))),
            ("tokenizer.ggml.padding_token_id", .u32(UInt32(specials.max()!))),
            ("tokenizer.ggml.add_bos_token", .bool(false)),
            ("tokenizer.chat_template", .string("{{ messages[0]['content'] }}")),
            (a + "loop.varying", .bool(false)),
            ("general.sampling.temp", .f32(1.0)),
            ("general.sampling.top_k", .u32(0)),
            ("general.sampling.top_p", .f32(1.0)),
            ("general.sampling.min_p", .f32(0.0)),
        ]
        _ = md.count
        let h = UInt64(hidden), w = UInt64(width), e = UInt64(experts)
        let q = UInt64(heads * headDim), kv = UInt64(kvHeads * headDim)
        var t: [GGUFFixture.Tensor] = [
            .init(name: "output.weight", ne: [h, UInt64(vocab)], type: .q8_0),
            .init(name: "output_norm.weight", ne: [h], type: .f32),
            .init(name: "output_res_query.weight", ne: [h], type: .f32),
            .init(name: "token_embd.weight", ne: [h, UInt64(vocab)], type: .q8_0),
            .init(name: "token_embd_norm.weight", ne: [h], type: .f32),
        ]
        for L in 0..<layers {
            func add(_ suffix: String, _ ne: [UInt64], _ type: GGMLType) {
                t.append(.init(name: "blk.\(L).\(suffix)", ne: ne, type: type))
            }
            add("attn_gate.weight", [h, q], .q8_0)
            add("attn_k.weight", [h, kv], .q8_0)
            add("attn_k_conv.weight", [UInt64(conv), kv], .f32)
            if full.contains(L) {
                add("attn_k_conv_proj.weight", [h, UInt64(conv * kvHeads)], .bf16)
                add("attn_k_inp_norm.weight", [h], .f32)
            } else {
                add("attn_norm.weight", [h], .f32)
            }
            add("attn_output.weight", [q, h], .q8_0)
            add("attn_post_conv.weight", [UInt64(conv), h], .f32)
            add("attn_q.weight", [h, q], .q8_0)
            add("attn_q_clip.weight", [UInt64(headDim), UInt64(heads)], .f32)
            if full.contains(L) { add("attn_q_inp_norm.weight", [h], .f32) }
            add("attn_rel.weight", [h, UInt64(heads * dRel)], .q8_0)
            add("attn_rel_bias.weight", [UInt64(dRel), UInt64(full.contains(L) ? extent : window)], .f32)
            add("attn_rel_clip.weight", [UInt64(dRel), UInt64(heads)], .f32)
            if L > 0 { add("attn_res_query.weight", full.contains(L) ? [h, 3] : [h], .f32) }
            add("attn_sink_proj.weight", [h, UInt64(heads)], .bf16)
            add("attn_v.weight", [h, kv], .q8_0)
            add("attn_v_conv.weight", [UInt64(conv), kv], .f32)
            if full.contains(L) {
                add("attn_v_conv_proj.weight", [h, UInt64(conv * kvHeads)], .bf16)
                add("attn_v_inp_norm.weight", [h], .f32)
            }
            add("exp_probs_b.bias", [e], .f32)
            if formats[L] == .nvfp4 { add("ffn_down_exps.scale", [e], .f32) }
            add("ffn_down_exps.weight", [w, h, e], formats[L])
            add("ffn_gate_inp.weight", [h, e], .f32)
            if formats[L] == .nvfp4 { add("ffn_gate_up_exps.scale", [e], .f32) }
            add("ffn_gate_up_exps.weight", [h, 2 * w, e], formats[L])
            add("ffn_norm.weight", [h], .f32)
            add("ffn_post_conv.weight", [UInt64(conv), h], .f32)
            add("ffn_res_query.weight", [h], .f32)
            if L >= loopStart && L < loopEnd {
                add("loop_msg_query.weight", [h, 4], .f32)
                for g in ["q", "k", "v", "ffn"] {
                    add("loop_\(g)_ctrl_down.weight", [2 * h, UInt64(gateRank)], .f32)
                    add("loop_\(g)_ctrl_forget.weight", [UInt64(gateRank), h], .f32)
                    add("loop_\(g)_ctrl_gain.weight", [UInt64(gateRank), h], .f32)
                }
            }
        }
        return GGUFFixture(metadata: md, tensors: t)
    }

    /// The runtime baseline this toy shape installs as.
    var archConfig: ArchConfig {
        ArchConfig(
            hiddenSize: hidden, intermediateSize: width, moeIntermediateSize: width,
            numHeads: heads, numKVHeads: kvHeads, numFullKVHeads: kvHeads,
            headDim: headDim, fullHeadDim: headDim, vocabSize: vocab, slidingWindow: window,
            finalLogitSoftcap: 30, ropeTheta: 0, fullRopeTheta: 0, partialRotaryFactor: 0,
            numLayers: layers, numExperts: experts, topKExperts: topK,
            tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: (0..<layers).map { full.contains($0) ? 1 : 0 },
            hiddenActivation: "situ", family: .inkling2Loop, attnOutputGate: false,
            attentionScale: pow(Double(headDim), -0.5), embeddingScaledBySqrtHidden: false,
            routerScaled: false, ffnSandwichNorms: false, sharedExpertGated: false,
            ropeNeoxSubdim: false, routerScoringFunc: "sigmoid", routedScalingFactor: 8,
            relativePosition: RelativePositionConfig(
                dRel: dRel, extent: extent, projDim: heads * dRel, logScalingFloor: 8192,
                logScalingAlpha: Double(Float(0.1))),
            sconvKernelSize: conv, numSharedExperts: 0, embedNormEnabled: true,
            routerGateBias: true, routerNormAfterTopK: true, unpaddedVocabSize: tokens.count,
            qkNorm: false, attentionSinks: true,
            inkling2: Inkling2Config(
                loopCoreStart: loopStart, loopCoreEnd: loopEnd, loopNumPasses: passes,
                loopGateRank: gateRank, attnResBlockSize: block,
                qkClipThreshold: Double(Float(22.627416997969522)), relClipThreshold: 8,
                attentionSinkProjected: true, expertGateSoftcap: 4, expertUpSoftcap: 25,
                routedExpertFormats: formats.map(\.name)))
    }
}

@Suite(.serialized) struct Inkling2InstallerTests {

    static func tmp(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("ink2-\(tag)-\(UUID().uuidString)")
    }

    /// Writes the toy GGUF; returns its path, bytes and tensor offsets.
    static func writeToy(_ shape: Inkling2Shape = Inkling2Shape()) throws
        -> (path: String, data: Data, offsets: [String: UInt64]) {
        let (data, offsets) = shape.fixture().encode()
        let dir = tmp("src")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let path = (dir as NSString).appendingPathComponent("toy.gguf")
        try data.write(to: URL(fileURLWithPath: path))
        return (path, data, offsets)
    }

    static func pin(for data: Data, path: String = "dir/toy.gguf") -> GGUFSourcePin {
        let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return GGUFSourcePin(commit: String(repeating: "a", count: 40),
                             shards: [.init(path: path, sha256: sha, size: UInt64(data.count))])
    }

    // MARK: - Reader

    @Test func headerParsesTypesOffsetsAndRefusals() throws {
        let fixture = Inkling2Shape().fixture()
        let (data, offsets) = fixture.encode()
        guard case .complete(let header) = try GGUF.parseHeader(path: "toy", fileSize: UInt64(data.count),
                                                                prefix: data) else {
            Issue.record("toy header did not parse"); return
        }
        #expect(header.tensors.count == fixture.tensors.count)
        #expect(header.metadata["inkling2_loop.block_count"]?.uint64 == 2)
        for t in header.tensors { #expect(t.absoluteOffset == offsets[t.name], "\(t.name)") }
        let nvfp4 = try #require(header.tensors.first { $0.name == "blk.0.ffn_gate_up_exps.weight" })
        // 128 rows (gate and up) of 128 values (two 36-byte blocks), two experts.
        #expect(nvfp4.type == .nvfp4 && nvfp4.sizeBytes == 128 * 2 * 36 * 2)
        // A short prefix asks for more.
        guard case .needBytes(let n) = try GGUF.parseHeader(path: "toy", fileSize: UInt64(data.count),
                                                            prefix: data.prefix(64)) else {
            Issue.record("short prefix parsed"); return
        }
        #expect(n > 64)
        // A tensor type the installer cannot size (Q4_K, id 12) is refused:
        // patch the type of the first tensor info, `output.weight`, whose
        // name is followed by `n_dims` (4 bytes) and two `ne` (16 bytes).
        var bytes = data
        let at = try #require(data.range(of: Data("output.weight".utf8))).upperBound + 4 + 16
        #expect(bytes[at] == UInt8(GGMLType.q8_0.rawValue))
        bytes[at] = 12
        #expect(throws: RepackError.self) {
            _ = try GGUF.parseHeader(path: "bad", fileSize: UInt64(bytes.count), prefix: bytes)
        }
    }

    /// The toy's arch is read in full from its metadata; the pinned-shape
    /// check refuses it.
    @Test func archIsReadFromTheGGUFAndPinChecked() throws {
        let data = Inkling2Shape().fixture().encode().data
        guard case .complete(let header) = try GGUF.parseHeader(path: "toy", fileSize: UInt64(data.count),
                                                                prefix: data) else {
            Issue.record("toy header did not parse"); return
        }
        let arch = try ArchInfo.fromGGUFInkling2(metadata: header.metadata, tensors: header.tensors, path: "toy")
        #expect(arch.family == .inkling2Loop && arch.numLayers == 2 && arch.hiddenSize == 128)
        #expect(arch.fullAttentionLayerMask == [0, 1])
        #expect(arch.inkling2?.routedExpertFormats == ["NVFP4", "Q8_0"])
        #expect(arch.inkling2?.qkClipThreshold == Double(Float(22.627416997969522)))
        #expect(arch.inkling2?.relLogScalingAlpha == Double(Float(0.1)))
        #expect(arch.inkling2?.unpaddedVocabSize == 6)
        #expect(throws: RepackError.self) { try ArchInfo.requirePinnedInkling2Shape(arch, path: "toy") }
    }

    /// The production shape, as a header only (its tensors are described, not
    /// written): the reader yields exactly the pinned arch, the planner the
    /// per-layer strides, and the layout it writes validates in the runtime.
    @Test func productionHeaderPlansThePinnedInstall() throws {
        let fixture = Inkling2Shape.production.fixture()
        let (bytes, _, _) = fixture.header()
        guard case .complete(let header) = try GGUF.parseHeader(
            path: "prod", fileSize: fixture.describedFileSize(), prefix: bytes) else {
            Issue.record("production header did not parse"); return
        }
        #expect(header.tensors.count == 744)
        let arch = try ArchInfo.fromGGUFInkling2(metadata: header.metadata, tensors: header.tensors, path: "prod")
        try ArchInfo.requirePinnedInkling2Shape(arch, path: "prod")
        #expect(arch == .inkling2Mini)

        let out = Self.tmp("plan")
        let plan = try Inkling2GGUFPlanner.plan(header: header, shardPath: "prod", arch: arch,
                                                outputDir: out, matchedModelID: nil)
        #expect(plan.resident.entries.count == 696)
        #expect(plan.layers.prefix(20).allSatisfy { $0.expertStride == 4_489_216 })
        #expect(plan.layers.suffix(4).allSatisfy { $0.expertStride == 8_470_528 })
        #expect(plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize } == 15_829_303_296)
        // Every byte is an identity copy of a source range.
        #expect(plan.resident.entries.allSatisfy { $0.weightTransform == .identity && $0.sourceScales == nil })
        let embd = try #require(plan.resident.entries.first { $0.name == "token_embd.weight" })
        #expect(embd.dtype == GGMLBlockFormat.q8_0.residentDtype && embd.logicalShape4 == [201_024, 2304, 0, 0])
        let sink = try #require(plan.resident.entries.first { $0.name == "blk.0.attn_sink_proj.weight" })
        #expect(sink.dtype == GGMLBlockFormat.bf16.residentDtype)
        #expect(plan.resident.entries.allSatisfy { $0.fileOffset % 256 == 0 })

        // The layout the installer writes is the one the runtime validates.
        let dir = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("packed_experts"),
                                                withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try GTurboJSON.encodeInkling2Layout(plan: plan)
            .write(to: dir.appendingPathComponent("packed_experts/layout.json"))
        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)
        let view = try Inkling2ExpertLayout.validated(layout: layout, config: .inkling2Mini, pageSize: 16_384)
        #expect(view.layers[0].gateUp.size == 2_985_984 && view.layers[0].down.offset == 2_985_984)
        #expect(view.layers[23].expertStride == 8_470_528)
    }

    @Test func tokenizerSidecarsFollowTheVendorPipeline() throws {
        let data = Inkling2Shape().fixture().encode().data
        guard case .complete(let header) = try GGUF.parseHeader(path: "toy", fileSize: UInt64(data.count),
                                                                prefix: data) else {
            Issue.record("toy header did not parse"); return
        }
        let sidecars = try GGUFTokenizer.synthesize(metadata: header.metadata, path: "toy")
        let tok = try #require(try JSONSerialization.jsonObject(with: sidecars.tokenizerJSON) as? [String: Any])
        let model = try #require(tok["model"] as? [String: Any])
        #expect(model["ignore_merges"] as? Bool == true)
        #expect((model["vocab"] as? [String: Int])?["ab"] == 2)
        #expect((model["merges"] as? [[String]]) == [["a", "b"]])
        #expect(tok["normalizer"] is NSNull)
        let added = try #require(tok["added_tokens"] as? [[String: Any]])
        #expect(added.map { $0["id"] as? Int } == [4, 5])
        #expect(added.allSatisfy { $0["special"] as? Bool == true })
        let pre = try #require((tok["pre_tokenizer"] as? [String: Any])?["pretokenizers"] as? [[String: Any]])
        let regex = try #require((pre[0]["pattern"] as? [String: String])?["Regex"])
        #expect(regex.hasPrefix(#"[^\r\n\p{L}\p{N}]?[\p{Lu}"#) && regex.hasSuffix(#"\s+(?!\S)|\s+"#))
        #expect(pre[1]["trim_offsets"] as? Bool == true && pre[1]["add_prefix_space"] as? Bool == false)
        let config = try #require(try JSONSerialization.jsonObject(with: sidecars.tokenizerConfigJSON)
            as? [String: Any])
        #expect(config["eos_token"] as? String == "<|end|>" && config["pad_token"] as? String == "<|pad|>")
        #expect(config["add_bos_token"] as? Bool == false && config["bos_token"] is NSNull)
        #expect(sidecars.chatTemplate == "{{ messages[0]['content'] }}")
        // The vendor's generation_config.json: sampling on at temperature 1,
        // no top-k / top-p / min-p, EOS and padding ids.
        let generationData = try #require(sidecars.generationConfigJSON)
        let generation = try #require(try JSONSerialization.jsonObject(with: generationData) as? [String: Any])
        #expect(Set(generation.keys) == ["do_sample", "temperature", "eos_token_id", "pad_token_id"])
        #expect(generation["do_sample"] as? Bool == true && generation["temperature"] as? Double == 1.0)
        #expect(generation["eos_token_id"] as? Int == 4 && generation["pad_token_id"] as? Int == 5)
    }

    /// One GGUF, one `tokenizer.json`: members sorted by key bytes, the vocab
    /// in id order (`ab` after `b`), nothing taken from a dictionary's hash
    /// order. The real file's vocab came out in a different order per run
    /// through `JSONSerialization`'s `.sortedKeys`.
    @Test func tokenizerJSONBytesAreFixed() throws {
        let data = Inkling2Shape().fixture().encode().data
        guard case .complete(let header) = try GGUF.parseHeader(path: "toy", fileSize: UInt64(data.count),
                                                                prefix: data) else {
            Issue.record("toy header did not parse"); return
        }
        let sidecars = try GGUFTokenizer.synthesize(metadata: header.metadata, path: "toy")
        let regex = try #require(GGUFTokenizer.preTokenizers["inkling2"]?.regex)
            .replacingOccurrences(of: #"\"#, with: #"\\"#)
        let special = #""lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true}"#
        let parts: [String] = [
            #"{"added_tokens":[{"content":"<|end|>","id":4,"#, special,
            #",{"content":"<|pad|>","id":5,"#, special, "],",
            #""decoder":{"add_prefix_space":true,"trim_offsets":true,"type":"ByteLevel","use_regex":true},"#,
            #""model":{"byte_fallback":false,"continuing_subword_prefix":null,"dropout":null,"#,
            #""end_of_word_suffix":null,"fuse_unk":false,"ignore_merges":true,"merges":[["a","b"]],"#,
            #""type":"BPE","unk_token":null,"vocab":{"a":0,"b":1,"ab":2,"c":3}},"#,
            #""normalizer":null,"padding":null,"#,
            #""post_processor":{"add_prefix_space":true,"trim_offsets":false,"type":"ByteLevel","use_regex":true},"#,
            #""pre_tokenizer":{"pretokenizers":[{"behavior":"Isolated","invert":false,"pattern":{"Regex":""#,
            regex, #""},"type":"Split"},"#,
            #"{"add_prefix_space":false,"trim_offsets":true,"type":"ByteLevel","use_regex":false}],"#,
            #""type":"Sequence"},"truncation":null,"version":"1.0"}"#,
        ]
        let expected = parts.joined()
        #expect(String(decoding: sidecars.tokenizerJSON, as: UTF8.self) == expected)
        let again = try GGUFTokenizer.synthesize(metadata: header.metadata, path: "toy")
        #expect(again.tokenizerJSON == sidecars.tokenizerJSON)
    }

    @Test func orderedJSONKeepsMemberOrderAndEscapes() throws {
        typealias J = GGUFTokenizer.OrderedJSON
        let text = "q\"b\\s/\u{1}\u{1F}\n\t\u{8}\u{C}\r\u{7F}é"
        let value: J = .object([("z", .int(-3)), ("a", .array([.null, .bool(true), .bool(false)])),
                                ("s", .string(text))])
        #expect(String(decoding: value.bytes, as: UTF8.self)
                == #"{"z":-3,"a":[null,true,false],"s":"q\"b\\s/\u0001\u001f\n\t\b\f\r"# + "\u{7F}é\"}")
        let back = try #require(try JSONSerialization.jsonObject(with: Data(value.bytes)) as? [String: Any])
        #expect(back["s"] as? String == text && back["z"] as? Int == -3)
        // Sorted by UTF-8 bytes: case-sensitive, not numeric.
        let sorted: J = .sorted(["b": .null, "a9": .null, "B": .null, "a10": .null])
        #expect(String(decoding: sorted.bytes, as: UTF8.self) == #"{"B":null,"a10":null,"a9":null,"b":null}"#)
    }

    // MARK: - Local install end to end

    static func install(_ gguf: (path: String, data: Data, offsets: [String: UInt64]), output: String,
                        pin: GGUFSourcePin, resume: Bool = false) async throws -> RemoteStreamingRepackResult {
        var options = RemoteStreamingRepackOptions(
            repoID: "owner/toy", revision: pin.commit, outputDir: output, requireKnownSource: false,
            rangeChunkBytes: 4096, minFreeReserveBytes: 0, overwrite: true, resume: resume,
            baseURL: URL(string: "https://hf.test")!)
        options.ggufPin = pin
        options.localGGUFPath = gguf.path
        return try await RemoteStreamingRepacker(options: options).run()
    }

    @Test func localInstallCopiesEveryByteAndLoads() async throws {
        let shape = Inkling2Shape()
        let gguf = try Self.writeToy(shape)
        let output = Self.tmp("install") + ".gturbo"
        defer {
            try? FileManager.default.removeItem(atPath: output)
            try? FileManager.default.removeItem(atPath: (gguf.path as NSString).deletingLastPathComponent)
        }
        let pin = Self.pin(for: gguf.data)
        let result = try await Self.install(gguf, output: output, pin: pin)
        #expect(!result.dryRun && result.expertLayerCount == 2)

        // Every tensor's bytes are the GGUF's.
        let verified = try GGUFCopyVerifier.run(installDirectory: output, ggufPath: gguf.path)
        #expect(verified.expertTensors == 4)
        #expect(verified.residentTensors == shape.fixture().tensors.count - 4)
        #expect(verified.bytesCompared == shape.fixture().tensors.reduce(0) { $0 + $1.bytes })

        // The receipt verifies, the manifest records the pin.
        let receipt = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputGTurbo: output))
        #expect(receipt.fileCount > 0)
        let manifest = try #require(try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: output).appendingPathComponent("manifest.json")))
            as? [String: Any])
        let source = try #require(manifest["source"] as? [String: Any])
        #expect(source["kind"] as? String == "gguf" && source["pinFingerprint"] as? String == pin.fingerprint)
        #expect(manifest["sourceSnapshotHash"] as? String == "sha256:" + pin.fingerprint)
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"] {
            #expect(FileManager.default.fileExists(atPath: "\(output)/tokenizer/\(name)"))
        }

        // The runtime loads it against the toy baseline and reads the GGUF's bytes.
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: URL(fileURLWithPath: output), device: device,
                                   expecting: shape.archConfig)
        let layout = try model.inkling2ExpertLayout()
        #expect(layout.layers.map(\.format) == [.nvfp4, .q8_0])
        let q = try model.inkling2AttnQ(layer: 1)
        #expect(q.dtype == GGMLBlockFormat.q8_0.residentDtype && q.length == 128 * 4 * 34)
        let src = try #require(gguf.offsets["blk.1.attn_q.weight"])
        let qBytes = UnsafeRawBufferPointer(start: q.buffer.contents().advanced(by: Int(q.offset)), count: Int(q.length))
        #expect(qBytes.elementsEqual(gguf.data[Int(src)..<Int(src + q.length)]))
        #expect(try model.inkling2ExpertGateUpScale(layer: 0).length == 8)
        #expect(throws: ModelError.self) { _ = try model.inkling2ExpertGateUpScale(layer: 1) }
        #expect(try model.inkling2LoopGateDown(layer: 1, .ffn).shape.0 == 4)
        // Expert 1 of layer 1 (Q8_0): its down rows are the GGUF's.
        let expert = try model.routedExpert(layer: 1, expert: 1)
        let down = layout.layers[1].down
        let downSrc = try #require(gguf.offsets["blk.1.ffn_down_exps.weight"]) + down.size
        let blob = UnsafeRawBufferPointer(start: expert.buffer.contents().advanced(by: Int(expert.offset)),
                                          count: Int(layout.layers[1].expertStride))
        #expect(blob[Int(down.offset)..<Int(down.offset + down.size)]
            .elementsEqual(gguf.data[Int(downSrc)..<Int(downSrc + down.size)]))
        // Resident streaming and the trusted-receipt policy read each layer
        // at its own stride too: the receipt validates without falling back,
        // and layer 0's expert 1 (NVFP4, 16 KiB stride) is the GGUF's bytes.
        let resident = try Model.load(directoryURL: URL(fileURLWithPath: output), device: device,
                                      expecting: shape.archConfig, streamingMode: .resident,
                                      integrityPolicy: .trustedReceiptWhenValid)
        #expect(resident.integrityFallbackDetail == nil)
        let nvfp4 = try resident.routedExpert(layer: 0, expert: 1)
        let gateUp = layout.layers[0].gateUp
        #expect(layout.layers[0].expertStride == 16_384 && layout.layers[1].expertStride == 32_768)
        let gateUpSrc = try #require(gguf.offsets["blk.0.ffn_gate_up_exps.weight"]) + gateUp.size
        let nvfp4Blob = UnsafeRawBufferPointer(start: nvfp4.buffer.contents().advanced(by: Int(nvfp4.offset)),
                                               count: Int(gateUp.size))
        #expect(nvfp4Blob.elementsEqual(gguf.data[Int(gateUpSrc)..<Int(gateUpSrc + gateUp.size)]))
        // The family stays gated at the product entry point.
        #expect(throws: ModelError.familyRunnerNotImplemented(
            family: "inkling2Loop", missingAxes: ManifestReader.inkling2RequiredAxes)) {
            _ = try ManifestReader.peekFamily(directoryURL: URL(fileURLWithPath: output))
        }
    }

    @Test func localCopyMustMatchThePin() async throws {
        let gguf = try Self.writeToy()
        defer { try? FileManager.default.removeItem(atPath: (gguf.path as NSString).deletingLastPathComponent) }
        var wrong = Self.pin(for: gguf.data)
        wrong = GGUFSourcePin(commit: wrong.commit, shards: [.init(path: wrong.shards[0].path,
            sha256: String(repeating: "0", count: 64), size: wrong.shards[0].size)])
        let output = Self.tmp("wrongsha") + ".gturbo"
        defer { try? FileManager.default.removeItem(atPath: output) }
        await #expect(throws: RepackError.self) { _ = try await Self.install(gguf, output: output, pin: wrong) }
        #expect(!FileManager.default.fileExists(atPath: output))
        let short = GGUFSourcePin(commit: wrong.commit, shards: [.init(path: wrong.shards[0].path,
            sha256: Self.pin(for: gguf.data).shards[0].sha256, size: UInt64(gguf.data.count) - 1)])
        await #expect(throws: RepackError.self) { _ = try await Self.install(gguf, output: output, pin: short) }
    }

    /// An install cancelled mid-copy keeps its checkpoint and resumes to the
    /// same bytes, reusing the ranges it had finished.
    @Test func interruptedLocalInstallResumes() async throws {
        let gguf = try Self.writeToy()
        let output = Self.tmp("resume") + ".gturbo"
        defer {
            try? FileManager.default.removeItem(atPath: output)
            try? FileManager.default.removeItem(atPath: output + ".partial")
            try? FileManager.default.removeItem(atPath: (gguf.path as NSString).deletingLastPathComponent)
        }
        let pin = Self.pin(for: gguf.data)
        var options = RemoteStreamingRepackOptions(
            repoID: "owner/toy", revision: pin.commit, outputDir: output, requireKnownSource: false,
            rangeChunkBytes: 4096, minFreeReserveBytes: 0, overwrite: true,
            baseURL: URL(string: "https://hf.test")!)
        options.ggufPin = pin
        options.localGGUFPath = gguf.path
        final class Canceller: @unchecked Sendable {
            let lock = NSLock()
            var task: Task<RemoteStreamingRepackResult, Error>?
            func cancel() { lock.withLock { task?.cancel() } }
            func set(_ value: Task<RemoteStreamingRepackResult, Error>) { lock.withLock { task = value } }
        }
        let canceller = Canceller()
        let first = Task { [options] in
            try await RemoteStreamingRepacker(options: options).run { progress in
                if case .copyingPayload(_, let copied, _) = progress, copied > 0 { canceller.cancel() }
            }
        }
        canceller.set(first)
        await #expect(throws: (any Error).self) { _ = try await first.value }
        #expect(!FileManager.default.fileExists(atPath: output))

        let resumed = try await Self.install(gguf, output: output, pin: pin, resume: true)
        #expect(resumed.reusedBytes > 0)
        #expect(resumed.reusedBytes < resumed.remoteBytesToDownload)
        let verified = try GGUFCopyVerifier.run(installDirectory: output, ggufPath: gguf.path)
        #expect(verified.expertTensors == 4)
    }

    // MARK: - Remote (pinned Hub) path

    /// The pinned remote path installs the same bytes as the local one, and
    /// refuses a file whose ETag and Xet hash both disagree with the pin.
    @Test func remoteInstallMatchesTheLocalInstall() async throws {
        let gguf = try Self.writeToy()
        let local = Self.tmp("local") + ".gturbo"
        let remote = Self.tmp("remote") + ".gturbo"
        defer {
            for path in [local, remote, (gguf.path as NSString).deletingLastPathComponent] {
                try? FileManager.default.removeItem(atPath: path)
            }
        }
        let pin = Self.pin(for: gguf.data)
        _ = try await Self.install(gguf, output: local, pin: pin)

        Inkling2FakeHub.reset(file: gguf.data, path: pin.shards[0].path, commit: pin.commit,
                              etag: pin.shards[0].sha256)
        var options = RemoteStreamingRepackOptions(
            repoID: "owner/toy", revision: pin.commit, outputDir: remote, requireKnownSource: false,
            rangeChunkBytes: 4096, minFreeReserveBytes: 0, overwrite: true,
            downloadSession: RemoteDownloadSession(protocolClasses: [Inkling2FakeHub.self]),
            baseURL: URL(string: "https://ink2.test")!, retryBaseDelayNs: 0)
        options.ggufPin = pin
        let result = try await RemoteStreamingRepacker(options: options).run()
        #expect(result.rangeRequestCount > 1)
        #expect(Inkling2FakeHub.rangeRequests() >= result.rangeRequestCount)
        _ = try GGUFCopyVerifier.run(installDirectory: remote, ggufPath: gguf.path)
        for file in ["model_weights.bin", "packed_experts/layout.json", "packed_experts/layer_00.bin",
                     "packed_experts/layer_01.bin", "manifest.json", "tokenizer/tokenizer.json"] {
            let a = try Data(contentsOf: URL(fileURLWithPath: "\(local)/\(file)"))
            let b = try Data(contentsOf: URL(fileURLWithPath: "\(remote)/\(file)"))
            #expect(a == b, "\(file) differs between the local and the remote install")
        }

        // A Hub file that does not carry the pinned identity is refused.
        Inkling2FakeHub.reset(file: gguf.data, path: pin.shards[0].path, commit: pin.commit,
                              etag: String(repeating: "f", count: 64))
        try? FileManager.default.removeItem(atPath: remote)
        await #expect {
            _ = try await RemoteStreamingRepacker(options: options).run()
        } throws: { error in
            guard case RepackError.sourceFingerprintRejected = error else { return false }
            return true
        }

        // A Xet-stored file whose ETag is not the SHA-256 is accepted on the
        // pinned Xet hash (with the size).
        let xet = String(repeating: "e", count: 64)
        var xetPin = pin
        xetPin = GGUFSourcePin(commit: pin.commit, shards: [.init(
            path: pin.shards[0].path, sha256: pin.shards[0].sha256, size: pin.shards[0].size, xetHash: xet)])
        Inkling2FakeHub.reset(file: gguf.data, path: pin.shards[0].path, commit: pin.commit,
                              etag: String(repeating: "f", count: 64), xetHash: xet)
        options.ggufPin = xetPin
        _ = try await RemoteStreamingRepacker(options: options).run()
        _ = try GGUFCopyVerifier.run(installDirectory: remote, ggufPath: gguf.path)
        // The same file under a different size is refused even with the hash.
        Inkling2FakeHub.reset(file: gguf.data.dropLast(32), path: pin.shards[0].path, commit: pin.commit,
                              etag: String(repeating: "f", count: 64), xetHash: xet)
        try? FileManager.default.removeItem(atPath: remote)
        await #expect(throws: RepackError.self) { _ = try await RemoteStreamingRepacker(options: options).run() }
    }

    /// A local file changed after its hash is refused before a range from it
    /// is committed.
    @Test func localFileChangedAfterVerificationIsRefused() async throws {
        let gguf = try Self.writeToy()
        defer { try? FileManager.default.removeItem(atPath: (gguf.path as NSString).deletingLastPathComponent) }
        let pin = Self.pin(for: gguf.data)
        let identity = try LocalGGUFSource.verify(path: gguf.path, against: pin.shards[0], audit: RepackAudit())
        // Same bytes rewritten: a new modification time is a different identity.
        try await Task.sleep(nanoseconds: 20_000_000)
        try gguf.data.write(to: URL(fileURLWithPath: gguf.path))
        #expect(throws: RepackError.self) {
            _ = try LocalGGUFSource.readHeader(path: gguf.path, size: UInt64(gguf.data.count), identity: identity)
        }
        let out = Self.tmp("changed")
        try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: out) }
        let target = (out as NSString).appendingPathComponent("dst.bin")
        FileManager.default.createFile(atPath: target, contents: Data(count: 64))
        let copy = CoalescedRangeCopy(id: "range-00000000", shardID: "s", sourceOffset: 0, size: 64,
                                      destinations: [RangeCopy(shardID: "s", sourceOffset: 0, size: 64,
                                                               destinationPath: target, destinationOffset: 0)])
        let provider = LocalFileSourceByteProvider(files: ["s": gguf.path], identities: ["s": identity])
        var committed = 0
        await #expect(throws: RepackError.self) {
            try await provider.copyBatch([copy], completedRangeIDs: [], partialDirectory: out,
                                         temporaryPath: (out as NSString).appendingPathComponent("t"),
                                         audit: RepackAudit(), progress: { _ in }, commit: { _ in committed += 1 })
        }
        #expect(committed == 0)
    }

    /// The verifier fails on a changed install byte and on a data section the
    /// tensors do not tile.
    @Test func verifierDetectsDifferences() async throws {
        let gguf = try Self.writeToy()
        let output = Self.tmp("verify") + ".gturbo"
        defer {
            try? FileManager.default.removeItem(atPath: output)
            try? FileManager.default.removeItem(atPath: (gguf.path as NSString).deletingLastPathComponent)
        }
        _ = try await Self.install(gguf, output: output, pin: Self.pin(for: gguf.data))
        _ = try GGUFCopyVerifier.run(installDirectory: output, ggufPath: gguf.path)
        // One byte of layer 1's expert 1 changed in the install.
        let layer = URL(fileURLWithPath: "\(output)/packed_experts/layer_01.bin")
        var bytes = try Data(contentsOf: layer)
        let at = 32_768 + 17_408 + 100
        bytes[at] ^= 0xFF
        try bytes.write(to: layer)
        #expect(throws: RepackError.self) { _ = try GGUFCopyVerifier.run(installDirectory: output, ggufPath: gguf.path) }
        bytes[at] ^= 0xFF
        try bytes.write(to: layer)
        // A GGUF with trailing bytes past its last tensor is not tiled.
        let padded = (gguf.path as NSString).deletingLastPathComponent + "/padded.gguf"
        try (gguf.data + Data(count: 64)).write(to: URL(fileURLWithPath: padded))
        #expect(throws: RepackError.self) { _ = try GGUFCopyVerifier.run(installDirectory: output, ggufPath: padded) }
    }

    /// The shipped source's fingerprint is its own pin's, and maps to its model ID.
    @Test func shippedSourcePinIsConsistent() throws {
        let source = SupportedModelSource.inkling2Mini
        let pin = try #require(source.ggufPin)
        #expect(source.kind == .gguf && source.revision == pin.commit)
        #expect(source.sourceIndexSHA256 == pin.fingerprint)
        #expect(SourceFingerprint.modelID(forIndexSha256: pin.fingerprint) == "inkling-2-mini-ud-q4_k_xl")
        #expect(pin.totalBytes == 17_629_760_736 && pin.shards.count == 1)
        let options = source.installOptions(outputDirectory: URL(fileURLWithPath: "/tmp/x.gturbo"),
                                            overwrite: false, token: nil)
        #expect(options.ggufPin == pin && options.requireKnownSource && options.localGGUFPath == nil)
    }
}

/// A one-file Hugging Face stand-in for the remote GGUF path: answers HEAD
/// with the commit, size, ETag and range support, and GET with `Range`.
/// Its statics are its own, so it does not race the serialized remote suite.
final class Inkling2FakeHub: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var file = Data()
    nonisolated(unsafe) private static var path = ""
    nonisolated(unsafe) private static var commit = ""
    nonisolated(unsafe) private static var etag = ""
    nonisolated(unsafe) private static var xetHash: String?
    nonisolated(unsafe) private static var ranges = 0

    static func reset(file: Data, path: String, commit: String, etag: String, xetHash: String? = nil) {
        lock.withLock {
            (self.file, self.path, self.commit, self.etag, self.xetHash, ranges) = (file, path, commit, etag, xetHash, 0)
        }
    }

    static func rangeRequests() -> Int { lock.withLock { ranges } }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "ink2.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let (file, path, commit, etag, xetHash) = Self.lock.withLock {
            (Self.file, Self.path, Self.commit, Self.etag, Self.xetHash)
        }
        let url = request.url!
        func respond(_ status: Int, _ headers: [String: String] = [:], _ body: Data = Data()) {
            client?.urlProtocol(self, didReceive: HTTPURLResponse(url: url, statusCode: status, httpVersion: nil,
                                                                  headerFields: headers)!,
                                cacheStoragePolicy: .notAllowed)
            if !body.isEmpty { client?.urlProtocol(self, didLoad: body) }
            client?.urlProtocolDidFinishLoading(self)
        }
        guard url.path.hasSuffix("/resolve/\(commit)/\(path)") else { return respond(404) }
        var headers = ["X-Repo-Commit": commit, "X-Linked-Size": "\(file.count)",
                       "X-Linked-ETag": "\"\(etag)\"", "Accept-Ranges": "bytes",
                       "Content-Encoding": "identity"]
        // As the Hub (and `FakeHFURLProtocol`) answer a Xet-stored file.
        if let xetHash { headers["X-Xet-Hash"] = xetHash; headers["ETag"] = "\"\(xetHash)\"" }
        if request.httpMethod == "HEAD" {
            headers["Content-Length"] = "\(file.count)"
            return respond(200, headers)
        }
        guard let range = request.value(forHTTPHeaderField: "Range"), range.hasPrefix("bytes="),
              case let parts = range.dropFirst(6).split(separator: "-"), parts.count == 2,
              let start = Int(parts[0]), let end = Int(parts[1]), start <= end, end < file.count else {
            return respond(416)
        }
        Self.lock.withLock { Self.ranges += 1 }
        headers["Content-Length"] = "\(end - start + 1)"
        headers["Content-Range"] = "bytes \(start)-\(end)/\(file.count)"
        respond(206, headers, Data(file[start...end]))
    }
}
