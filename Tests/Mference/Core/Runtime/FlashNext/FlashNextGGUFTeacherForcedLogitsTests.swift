import Foundation
import Metal
import Testing
@testable import Mference

/// WS4 U5 layer 3: teacher-forced logits from the GGUF install for one prompt,
/// for comparison against llama.cpp (`mf-compare`, outside the repo). Env-gated
/// and a model run (one per `lock.sh run`):
/// - `MFERENCE_UNSLOTH_GTURBO`: the `qwen38flashnextudiq3xxs.gturbo` directory;
/// - `MFERENCE_U5_TOKENS`: llama.cpp's `p<i>.tokens.json` (a JSON array of ids),
///   so both sides see identical tokens and the tokenizer is out of the gate;
/// - `MFERENCE_U5_OUT`: the output directory;
/// - `MFERENCE_U5_PROMPT` (optional): the prompt text, for the tokenizer
///   context check.
///
/// Writes `mference.logits.f32`, row-major `[n_tokens][n_vocab]` little-endian
/// float32: position `p`'s logits after the production decode path has consumed
/// tokens `0...p` one at a time (the head writes FP16; widened here). Also
/// writes, as context only: `mference.prefill-last.f32` (the chunked-prefill
/// logits of the last position) and `mference.context.json` (whether
/// Mference's tokenizer reproduces llama.cpp's ids).
@Suite struct FlashNextGGUFTeacherForcedLogitsTests {
    static let env = ProcessInfo.processInfo.environment
    static var enabled: Bool {
        env["MFERENCE_UNSLOTH_GTURBO"] != nil && env["MFERENCE_U5_TOKENS"] != nil && env["MFERENCE_U5_OUT"] != nil
    }

    static func appendRow(_ buffer: MTLBuffer, vocab: Int, to handle: FileHandle) throws {
        let half = buffer.contents().bindMemory(to: Float16.self, capacity: vocab)
        var row = [Float](repeating: 0, count: vocab)
        for i in 0..<vocab { row[i] = Float(half[i]) }
        try row.withUnsafeBytes { try handle.write(contentsOf: Data($0)) }
    }

    @Test(.enabled(if: FlashNextGGUFTeacherForcedLogitsTests.enabled,
                   "set MFERENCE_UNSLOTH_GTURBO, MFERENCE_U5_TOKENS and MFERENCE_U5_OUT"))
    func writeTeacherForcedLogits() async throws {
        let dirPath = try #require(Self.env["MFERENCE_UNSLOTH_GTURBO"])
        let tokensPath = try #require(Self.env["MFERENCE_U5_TOKENS"])
        let outPath = try #require(Self.env["MFERENCE_U5_OUT"])
        let dir = URL(fileURLWithPath: dirPath)
        let parsed = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: tokensPath)))
        let tokens = try #require(parsed as? [Int]).map(Int32.init)
        let out = URL(fileURLWithPath: outPath)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        #expect(!tokens.isEmpty && tokens.count <= 1_024)

        let context = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: context.device, streamingMode: .pread(slotCount: 16))
        #expect(model.config.family == .qwen38flashnextudiq3xxs)
        let vocab = model.config.vocabSize
        let runner = try FlashNextForwardRunner(model: model, context: context, maxContext: tokens.count + 8)
        let logits = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))

        // Gated output: the decode path, one position at a time.
        let path = out.appendingPathComponent("mference.logits.f32")
        FileManager.default.createFile(atPath: path.path, contents: nil)
        let handle = try FileHandle(forWritingTo: path)
        for (p, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: p, into: logits)
            try Self.appendRow(logits, vocab: vocab, to: handle)
        }
        try handle.close()
        let written = try FileManager.default.attributesOfItem(atPath: path.path)[.size] as? Int
        #expect(written == tokens.count * vocab * 4)

        // Context: the chunked-prefill last row.
        runner.reset()
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0, outputMode: .logits,
                                            config: .production(chunkTokens: 128), into: logits) { _ in }
        let lastPath = out.appendingPathComponent("mference.prefill-last.f32")
        FileManager.default.createFile(atPath: lastPath.path, contents: nil)
        let lastHandle = try FileHandle(forWritingTo: lastPath)
        try Self.appendRow(logits, vocab: vocab, to: lastHandle)
        try lastHandle.close()

        // Context: does Mference's own tokenizer reproduce llama.cpp's ids?
        var report: [String: Any] = ["positions": tokens.count, "vocab": vocab]
        if let promptPath = Self.env["MFERENCE_U5_PROMPT"],
           let folder = MFTokenizer.tokenizerFolder(forModelDirectory: dir) {
            let tokenizer = try await MFTokenizer.load(from: folder, family: .qwen38flashnextudiq3xxs)
            let ours = tokenizer.encode(try String(contentsOfFile: promptPath, encoding: .utf8), addBOS: false)
            let firstDiff = zip(ours, tokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            report["tokenizer_match"] = ours == tokens
            report["tokenizer_count"] = ours.count
            report["tokenizer_first_diff"] = firstDiff ?? (ours.count == tokens.count ? -1 : min(ours.count, tokens.count))
        }
        try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
            .write(to: out.appendingPathComponent("mference.context.json"))
        print("[u5] positions=\(tokens.count) wrote \(path.lastPathComponent) and context; \(report)")
    }
}
