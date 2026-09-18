import Foundation
import Metal
import Testing
@testable import Mference

/// Opt-in real-checkpoint correctness gate, not a performance benchmark.
/// Enforce AGENTS.md's hardware/memory/install/single-process preflight first.
/// Uses one model and runner, resets between arms; never installs a checkpoint.
@Suite(.serialized) struct DSV4InstalledPrefillTests {
    @Test func sparseCutoverAndWarmAppendsMatchSequentialExactly() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MFERENCE_DEEPSEEK_GTURBO"] else { return }
        let ctx = try MetalContext()
        let url = URL(fileURLWithPath: path)
        let mode: ExpertStreamingMode
        let modeName = env["MFERENCE_DEEPSEEK_QUALIFICATION_EXPERTS"] ?? "16"
        if modeName == "resident" { mode = .resident }
        else {
            let slots = try #require(Int(modeName))
            try #require(slots >= 8)
            mode = .pread(slotCount: slots)
        }
        // Default full-SHA verification, not trusted-receipt mode.
        let model = try Model.load(directoryURL: url, device: ctx.device,
            expecting: .deepseekV4Flash_284B_A13B, streamingMode: mode)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: url)
        let text = String(repeating: "A small town measures river levels every morning. Compare the weekly averages, explain uncertainty, and preserve the original observations.\n", count: 200)
        let allTokens = tokenizer.encode(text, addBOS: true)
        let boundaries = [2047, 2051, 2052, 2083]
        try #require(allTokens.count >= 2083)
        let tokens = Array(allTokens.prefix(2083))
        let runtime = RuntimeConfiguration(prefillChunkTokens: 128, forceLogitsHead: true)
        let runner = try RealForwardRunner(model: model, context: ctx, maxContext: 2112,
                                          runtimeConfiguration: runtime)
        let out = try #require(ctx.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        func snapshot() -> [UInt16] {
            let bits = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: UInt16.self),
                                                count: model.config.vocabSize))
            #expect(bits.allSatisfy { Float16(bitPattern: $0).isFinite }, "finite full-vocabulary logits")
            return bits
        }
        func greedy() -> Int32 {
            let values = out.contents().assumingMemoryBound(to: Float16.self)
            var best = 0
            for i in 1..<model.config.vocabSize where values[i] > values[best] { best = i }
            return Int32(best)
        }
        func log(_ text: String) {
            FileHandle.standardError.write(Data(("[dsv4 installed prefill] " + text + "\n").utf8))
        }
        log("begin strict-verified experts=\(modeName), sequential=2083, chunk=128")
        var referenceHeads: [[UInt16]] = []
        for (i, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: i, into: out)
            if boundaries.contains(i + 1) { referenceHeads.append(snapshot()) }
            if (i + 1).isMultiple(of: 256) { log("sequential \(i + 1)/2083") }
        }
        var continuation: [Int32] = []
        var referenceTail: [[UInt16]] = []
        for i in 0..<8 {
            let token = greedy()
            continuation.append(token)
            try await runner.produce(token: token, position: tokens.count + i, into: out)
            referenceTail.append(snapshot())
        }
        runner.reset()
        var position = 0
        for (index, end) in boundaries.enumerated() {
            let result = try await runner.prefillChunked(tokens: tokens[position..<end],
                startPosition: position, outputMode: .logits, config: .production(chunkTokens: 128),
                into: out, onProgress: { _ in })
            #expect(result.execution?.batchedTokens == end - position)
            #expect(result.execution?.replayedTokens == 0)
            let mismatches = zip(snapshot(), referenceHeads[index]).filter { $0 != $1 }.count
            #expect(mismatches == 0, "full logits at \(end)")
            log("batched head \(end): mismatches=\(mismatches), replay=\(result.execution?.replayedTokens ?? -1)")
            position = end
            try runner.prepareForContinuation(expectedPosition: position)
        }
        for (i, token) in continuation.enumerated() {
            #expect(greedy() == token, "greedy continuation \(i)")
            try await runner.produce(token: token, position: position + i, into: out)
            let mismatches = zip(snapshot(), referenceTail[i]).filter { $0 != $1 }.count
            #expect(mismatches == 0, "continuation full logits \(i)")
        }
        log("completed four boundary heads and eight greedy/full-logit continuation checks")
    }
}
