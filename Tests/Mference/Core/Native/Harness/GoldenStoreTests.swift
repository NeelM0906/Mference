import CryptoKit
import Foundation
import Testing
@testable import Mference

/// The golden loader (`GoldenStore`) and the committed prompt set and slice.
/// Model-free; the suite at the end reads the real goldens only when
/// `MFERENCE_NATIVE_GOLDENS` is set.
@Suite("GoldenStoreTests")
struct GoldenStoreTests {
    static let quick = Native.Qwen35Config(
        blockCount: 32, hiddenSize: 2560, ffnSize: 9216, headCount: 16, kvHeadCount: 4,
        keyLength: 256, valueLength: 256, rmsNormEpsilon: 1e-6, ropeFreqBase: 1e7,
        ropeDimensionCount: 64, ropeSections: [11, 11, 10, 0], fullAttentionInterval: 4,
        ssmConvKernel: 4, ssmStateSize: 128, ssmGroupCount: 16, ssmTimeStepRank: 32,
        ssmInnerSize: 4096, contextLength: 262_144, nextnPredictLayers: 0,
        vocabSize: 248_320, tiedEmbeddings: true)

    // MARK: synthetic run

    /// Writes a two-tensor golden for prompt "p" under a fresh temp root.
    static func syntheticRun(logitsBytes: Int? = nil, format: String = "mfe-golden-v1") throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("golden-\(UUID().uuidString)", isDirectory: true)
        let dir = root.appendingPathComponent("p", isDirectory: true)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("tensors"), withIntermediateDirectories: true)
        // logits: 2 rows x 4 cols at positions 2, 3
        var logits: [Float] = [0, 1, 2, 3, 3, 2, 1, 0]
        var data = Data(bytes: &logits, count: logits.count * 4)
        if let n = logitsBytes { data = data.prefix(n) }
        try data.write(to: dir.appendingPathComponent("logits.f32"))
        // l_out-3: positions 0, 1, 3 (ranges [0,2), [3,4)), 2 cols
        var lout: [Float] = [10, 11, 20, 21, 40, 41]
        try Data(bytes: &lout, count: lout.count * 4).write(to: dir.appendingPathComponent("tensors/l_out-3.f32"))
        // Kcur-3.1: second node of that name, 1 row
        var k: [Float] = [7, 8, 9]
        try Data(bytes: &k, count: k.count * 4).write(to: dir.appendingPathComponent("tensors/Kcur-3.1.f32"))
        try "<|im_start|>user\nhi<|im_end|>\n".write(to: dir.appendingPathComponent("prompt.txt"), atomically: true, encoding: .utf8)
        let manifest = """
        {"format": "\(format)", "prompt_id": "p", "prompt_set": "quick-v1", "mode": "golden",
         "source": {"engine": "llama.cpp", "commit": "b1b770d90d94cad9e9ea5b993bdbc383a522fb28", "backend": "metal",
           "model_desc": "toy", "model_path": "/x.gguf", "n_batch": 2048, "n_ubatch": 512, "kv": "f16",
           "flash_attn": "auto", "threads": 0, "n_ctx": 256},
         "enable_thinking": false, "rendered_prompt_file": "prompt.txt", "n_vocab": 4,
         "prompt_tokens": [5, 6, 7], "generated_tokens": [3, 0], "stopped_on_eog": false,
         "logits": {"file": "logits.f32", "dtype": "f32", "rows": 2, "cols": 4, "position_ranges": [[2, 4]]},
         "timing_s": {"prefill": 0, "pass_a": 0, "pass_b": 0},
         "capture": {"chunk": 512, "selected_ranges": [[0, 3]], "decode_steps": 1, "state_steps": 1},
         "tensors": [
          {"key": "l_out-3", "name": "l_out-3", "base": "l_out", "layer": 3, "occurrence": 0, "op": "ADD",
           "ne": [2, 3, 1, 1], "dtype": "f32", "cols": 2, "rows": 3, "position_ranges": [[0, 2], [3, 4]],
           "file": "tensors/l_out-3.f32"},
          {"key": "Kcur-3.1", "name": "Kcur-3", "base": "Kcur", "layer": 3, "occurrence": 1, "op": "ROPE",
           "ne": [3, 1, 1, 1], "dtype": "f32", "cols": 3, "rows": 1, "position_ranges": [[3, 4]],
           "file": "tensors/Kcur-3.1.f32"}
         ],
         "skipped": []}
        """
        try manifest.write(to: dir.appendingPathComponent("golden.json"), atomically: true, encoding: .utf8)
        return root
    }

    @Test func readsSyntheticRun() throws {
        let root = try Self.syntheticRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let g = try GoldenStore.open("p", root: root)
        #expect(g.promptTokens == [5, 6, 7])
        #expect(g.generatedTokens == [3, 0])
        #expect(g.teacherForcedTokens == [5, 6, 7, 3])
        #expect(try g.renderedPrompt().hasPrefix("<|im_start|>user"))

        let logits = try g.logits()
        #expect(logits.rows == 2 && logits.cols == 4)
        #expect(logits.positions == [2, 3])
        #expect(logits.row(1) == [3, 2, 1, 0])
        #expect(logits.row(atPosition: 2) == [0, 1, 2, 3])
        #expect(logits.row(atPosition: 9) == nil)

        let lout = try g.tensor(named: "l_out-3")
        #expect(lout.positions == [0, 1, 3])
        #expect(lout.row(atPosition: 3) == [40, 41])
        #expect(lout.rowIndex(ofPosition: 2) == nil)
        let sum = lout.withRow(1) { $0.reduce(0, +) }
        #expect(sum == 41)

        let k = try g.tensor(named: "Kcur-3", occurrence: 1)
        #expect(k.row(0) == [7, 8, 9])
        #expect(g.entry(named: "Kcur-3") == nil)
        #expect(throws: GoldenError.self) { try g.tensor(named: "missing-0") }
    }

    @Test func rejectsWrongSizeAndFormat() throws {
        let short = try Self.syntheticRun(logitsBytes: 12)
        defer { try? FileManager.default.removeItem(at: short) }
        let g = try GoldenStore.open("p", root: short)
        #expect(throws: GoldenError.self) { try g.logits() }

        let old = try Self.syntheticRun(format: "mfe-golden-v0")
        defer { try? FileManager.default.removeItem(at: old) }
        #expect(throws: GoldenError.self) { try GoldenStore.open("p", root: old) }
        #expect(throws: GoldenError.notConfigured) { try GoldenStore.open("p", root: nil) }
    }

    @Test func positionRanges() throws {
        #expect(try GoldenManifest.positions([[0, 2], [5, 7]]) == [0, 1, 5, 6])
        #expect(try GoldenManifest.positions([]) == [])
        #expect(throws: GoldenError.self) { try GoldenManifest.positions([[3, 1]]) }
    }

    @Test func tapPointsMapToTheQwen35Graph() {
        let c = Self.quick
        #expect(GoldenTapMap.graphTensor(for: .embedding, config: c).name == "model.input_embed")
        #expect(GoldenTapMap.graphTensor(for: .mixerInput(layer: 0), config: c).name == "attn_norm-0")
        // block 3 is full attention, block 2 Gated DeltaNet
        #expect(GoldenTapMap.graphTensor(for: .mixerOutput(layer: 3), config: c).name == "attn_output-3")
        #expect(GoldenTapMap.graphTensor(for: .mixerOutput(layer: 2), config: c).name == "linear_attn_out-2")
        #expect(GoldenTapMap.graphTensor(for: .ffnOutput(layer: 31), config: c).name == "ffn_out-31")
        #expect(GoldenTapMap.graphTensor(for: .layerOutput(layer: 7), config: c).name == "l_out-7")
        #expect(GoldenTapMap.graphTensor(for: .finalNorm, config: c).name == "result_norm")
        #expect(GoldenTapMap.graphTensor(for: .logits, config: c).name == "result_output")
        #expect(GoldenTapMap.graphTensor(.attentionOutput, layer: 3).name == "attn_gated-3")
        #expect(GoldenTapMap.graphTensor(.gdnState, layer: 0).name == "state_predelta-0")
    }

    // MARK: committed fixtures

    @Test func promptSetIsTheFrozenQuickV1() throws {
        let set = try GoldenStore.promptSet()
        #expect(set.name == "quick-v1" && set.version == 1)
        let ids = set.prompts.map(\.id)
        #expect(Set(ids).count == ids.count)
        #expect(ids == ["short-chat", "json-answer", "tool-call", "tool-result", "explain-thinking",
                        "essay-512", "review-512", "doc-2k", "synthesis-3k", "doc-4k", "long-32k"])
        #expect(set.prompts.allSatisfy { $0.maxNew >= 128 })
        let url = GoldenStore.fixturesDirectory.appendingPathComponent("prompts/quick-v1.json")
        let digest = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        #expect(digest == GoldenSlice.promptSetSHA256, "quick-v1.json changed; a change needs quick-v2")
    }

    @Test func sliceMatchesThePromptSet() throws {
        guard let slice = try GoldenSlice.load() else { return }
        let set = try GoldenStore.promptSet()
        #expect(slice.promptSet == set.name)
        #expect(Set(slice.prompts.keys) == Set(set.prompts.map(\.id)))
        for p in set.prompts {
            guard let s = slice.prompts[p.id] else { continue }
            #expect(s.generatedTokens.count <= p.maxNew, "\(p.id)")
            #expect(s.generatedTokens.count == s.logitsTop.count, "\(p.id)")
            #expect(!s.generatedTokens.isEmpty, "\(p.id)")
            // the greedy token is the top logit of its row
            for (g, top) in zip(s.generatedTokens, s.logitsTop) {
                #expect(top.ids.first == g, "\(p.id)")
            }
        }
    }
}

/// `Tests/Mference/Fixtures/native/goldens/quick-v1-<run>-slice.json`: a
/// small committed digest of a golden run (Scripts/native/golden/make_slice.py).
struct GoldenSlice: Decodable {
    /// SHA-256 of the committed `prompts/quick-v1.json`.
    static let promptSetSHA256 = "f245fbc76d264953ed3515649ebe3a0832c64b8bcce5701c3c36ddcf0f2744d4"

    struct Top: Decodable {
        var ids: [Int32]
        var values: [Float]
        var logSumExp: Double
        enum CodingKeys: String, CodingKey {
            case ids, values
            case logSumExp = "log_sum_exp"
        }
    }

    struct Prompt: Decodable {
        var promptTokenCount: Int
        var promptTokensSHA256: String
        var renderedPromptSHA256: String
        var generatedTokens: [Int32]
        var stoppedOnEog: Bool
        var logitsTop: [Top]
        /// SHA-256 of `logits.f32` and `prompt.txt`.
        var files: [String: String]
        var tensorCount: Int
        /// SHA-256 of the sorted `"<file> <sha256>\n"` lines of every tensor file.
        var tensorsDigest: String
        enum CodingKeys: String, CodingKey {
            case files
            case tensorCount = "tensor_count"
            case tensorsDigest = "tensors_digest"
            case promptTokenCount = "prompt_token_count"
            case promptTokensSHA256 = "prompt_tokens_sha256"
            case renderedPromptSHA256 = "rendered_prompt_sha256"
            case generatedTokens = "generated_tokens"
            case stoppedOnEog = "stopped_on_eog"
            case logitsTop = "logits_top"
        }
    }

    var format: String
    var promptSet: String
    var run: String
    var prompts: [String: Prompt]
    enum CodingKeys: String, CodingKey {
        case format, run, prompts
        case promptSet = "prompt_set"
    }

    static func url(run: String = "metal") -> URL {
        GoldenStore.fixturesDirectory.appendingPathComponent("goldens/quick-v1-\(run)-slice.json")
    }

    /// The committed slice, nil before the first capture is committed.
    static func load(run: String = "metal") throws -> GoldenSlice? {
        let u = url(run: run)
        guard FileManager.default.fileExists(atPath: u.path) else { return nil }
        return try JSONDecoder().decode(GoldenSlice.self, from: Data(contentsOf: u))
    }
}

/// Reads the real goldens (`MFERENCE_NATIVE_GOLDENS` = a run directory such
/// as `$PROGRAM/goldens/quick-v1/metal`); skipped when it is unset. No model
/// is loaded.
@Suite("GoldenStoreRealTests", .enabled(if: GoldenStore.root != nil))
struct GoldenStoreRealTests {
    @Test func everyPromptHasItsGoldens() throws {
        let set = try GoldenStore.promptSet()
        let config = GoldenStoreTests.quick
        for p in set.prompts {
            let g = try GoldenStore.open(p.id)
            #expect(g.manifest.promptSet == set.name, "\(p.id)")
            #expect(g.manifest.nVocab == config.vocabSize, "\(p.id)")
            let logits = try g.logits()
            #expect(logits.rows == g.generatedTokens.count, "\(p.id)")
            #expect(logits.positions.first == g.promptTokens.count - 1, "\(p.id)")
            guard g.manifest.mode == "golden" else { continue }
            // every tensor file has its declared size
            for t in g.tensors { _ = try g.tensor(named: t.name, occurrence: t.occurrence) }
            // every tap point is present at the last prompt position and the
            // first decode step
            let last = g.promptTokens.count - 1
            var points: [Native.TapPoint] = [.embedding, .finalNorm, .logits]
            for l in 0..<config.blockCount {
                points += [.mixerInput(layer: l), .mixerOutput(layer: l), .ffnOutput(layer: l), .layerOutput(layer: l)]
            }
            for point in points {
                let m = try g.tensor(for: point, config: config)
                #expect(m.rowIndex(ofPosition: last) != nil, "\(p.id) \(point) at \(last)")
                if g.manifest.capture.decodeSteps > 0 {
                    #expect(m.rowIndex(ofPosition: last + 1) != nil, "\(p.id) \(point) at \(last + 1)")
                }
            }
        }
    }

    @Test func goldensMatchTheCommittedSlice() throws {
        guard let slice = try GoldenSlice.load() else { return }
        let root = try #require(GoldenStore.root)
        for (id, s) in slice.prompts {
            let g = try GoldenStore.open(id)
            #expect(g.promptTokens.count == s.promptTokenCount, "\(id)")
            var tokens = g.promptTokens
            let tokenDigest = SHA256.hash(data: Data(bytes: &tokens, count: tokens.count * 4))
                .map { String(format: "%02x", $0) }.joined()
            #expect(tokenDigest == s.promptTokensSHA256, "\(id)")
            #expect(g.generatedTokens == s.generatedTokens, "\(id)")
            // the logits file is the one the slice was made from
            if let expected = s.files["logits.f32"] {
                let data = try Data(contentsOf: root.appendingPathComponent("\(id)/logits.f32"), options: .alwaysMapped)
                let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                #expect(digest == expected, "\(id) logits.f32")
            }
        }
    }
}
