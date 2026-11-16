import Accelerate
import Foundation
import Metal
import Testing
@testable import Mference

/// Class T gate for the grouped LUT4 GEMM on Flash-Next GGUF prefill tiles
/// (`FlashNextForwardRunner.usesSimdgroupLUT4RoutedExperts`). One process:
/// control is the runner with every prefill tile on the LUT4 row kernel
/// (`useRowKernelForGGUFPrefill`), candidate the default runner. Both prefill
/// the same prompts at `MFERENCE_FLASHNEXT_GATE_CHUNK` tokens per chunk (1024
/// or 2048; the grouped kernel engages only on tiles of about 8+ rows per
/// expert, so prompts of about 410+ tokens in one chunk) and are then
/// teacher-forced through the same saved answer on the decode path.
///
/// Items: the frozen long-synthesis prompt (2,940 tokens with Qwen's chat
/// template) and about 8K tokens of the repository's design docs, each with
/// 200 tokens of the saved Qwen answer. Thresholds are P3a's over every scored
/// position: fail on a mean shift > 0.005 nats that is distinguishable from
/// zero (batch-means 95 % interval), on a mean shift > 0.02, or on top-1
/// agreement < 0.98; max |shift| is a diagnostic. The candidate must have run
/// the grouped kernel on at least one tile.
///
/// Env: `MFERENCE_FLASHNEXT_GATE` (the `qwen38flashnextudiq3xxs.gturbo`
/// directory, read-only), `MFERENCE_FLASHNEXT_GATE_CHUNK`, optional
/// `MFERENCE_FLASHNEXT_GATE_OUT` (per-position rows as JSON). 16 expert slots.
/// `MFERENCE_FLASHNEXT_GATE_ITEM` (`long` or `docs`) runs one item, both arms,
/// so a long gate can be split into separate lock holds; the thresholds are
/// then applied offline to the pooled rows of all items, not per item.
@Suite(.serialized) struct FlashNextGroupedLUT4EquivalenceGateTests {
    static let env = ProcessInfo.processInfo.environment
    static var enabled: Bool { env["MFERENCE_FLASHNEXT_GATE"] != nil && env["MFERENCE_FLASHNEXT_GATE_CHUNK"] != nil }

    struct Item { let name: String; let tokens: [Int32]; let prefillCount: Int }
    struct Row { let nll: Double; let top1: Int }

    @Test(.enabled(if: FlashNextGroupedLUT4EquivalenceGateTests.enabled,
                   "set MFERENCE_FLASHNEXT_GATE and MFERENCE_FLASHNEXT_GATE_CHUNK"))
    func groupedLUT4PrefillKeepsTeacherForcedPerplexity() async throws {
        let install = URL(fileURLWithPath: try #require(Self.env["MFERENCE_FLASHNEXT_GATE"]))
        let chunk = try #require(Int(Self.env["MFERENCE_FLASHNEXT_GATE_CHUNK"] ?? ""))
        try #require(chunk == 1024 || chunk == 2048, "chunk must be 1024 or 2048 (host cap)")
        let root = GemmaPrefillGateFixture.repositoryRoot
        let folder = try #require(MFTokenizer.tokenizerFolder(forModelDirectory: install))
        let tokenizer = try await MFTokenizer.load(from: folder, family: .qwen38flashnextudiq3xxs)
        let answer = Array(tokenizer.encode(try Qwen36PrefillGateFixture.savedAnswer("long-synthesis.qwen36"),
                                            addBOS: false).prefix(200))
        try #require(answer.count == 200)

        let rows = try JSONDecoder().decode([[String: String]].self, from: Data(contentsOf:
            root.appendingPathComponent("docs/benchmark-prompts/real-generation-v1/long-synthesis.json")))
        let messages = try rows.map {
            MFTokenizer.Message(role: try #require(MFTokenizer.Role(rawValue: $0["role"] ?? "")),
                                content: $0["content"] ?? "")
        }
        let longHead = try tokenizer.encodeChat(messages: messages)
        var documents = ""
        for path in ["docs/SYSTEM_DESIGN.md", "docs/FAMILY_CONTRACT.md", "docs/RUNTIME_CONTROLS.md",
                     "docs/OPENAI_SERVER.md", "docs/BENCHMARKS.md"] {
            let text = try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
            documents += "<document path=\"\(path)\">\n\(text)\n</document>\n\n"
            if tokenizer.encode(documents, addBOS: false).count >= 7_800 { break }
        }
        let docTokens = Array(tokenizer.encode(documents, addBOS: false).prefix(7_800))
        try #require(docTokens.count == 7_800, "the design docs are shorter than the 8K item needs")
        let docsHead = try tokenizer.encodeChat(messages: [MFTokenizer.Message(
            role: .user, content: tokenizer.decode(docTokens)
                + "\n\nSummarize how the documents above keep memory bounded while serving a long prompt.")])
        let allItems = [Item(name: "long+qwen-answer", tokens: longHead + answer, prefillCount: longHead.count),
                        Item(name: "docs-8k+qwen-answer", tokens: docsHead + answer, prefillCount: docsHead.count)]
        let only = Self.env["MFERENCE_FLASHNEXT_GATE_ITEM"]
        let items = only == nil ? allItems : allItems.filter { $0.name.hasPrefix(only!) }
        try #require(!items.isEmpty, "MFERENCE_FLASHNEXT_GATE_ITEM must be long or docs")
        let longest = items.map(\.tokens.count).max() ?? 0

        let context = try MetalContext()
        let model = try Model.load(directoryURL: install, device: context.device, streamingMode: .pread(slotCount: 16))
        try #require(model.config.family == .qwen38flashnextudiq3xxs)
        try #require(FlashNextForwardRunner.usesSimdgroupLUT4RoutedExperts(
            supportsApple10TensorOps: context.device.supportsApple10TensorOps),
            "the grouped LUT4 kernel is selected only off Apple10")
        let vocab = model.config.vocabSize
        let logits = try #require(context.device.makeBuffer(length: vocab * 2, options: .storageModeShared))
        var values = [Float](repeating: 0, count: vocab)
        var exponent = [Float](repeating: 0, count: vocab)
        func analyze(target: Int32) -> Row {
            var source = vImage_Buffer(data: logits.contents(), height: 1,
                                       width: vImagePixelCount(vocab), rowBytes: vocab * 2)
            let count = vDSP_Length(vocab)
            var maximum: Float = 0, index: vDSP_Length = 0, sum: Float = 0
            values.withUnsafeMutableBufferPointer { v in
                var destination = vImage_Buffer(data: v.baseAddress, height: 1,
                                                width: vImagePixelCount(vocab), rowBytes: vocab * 4)
                vImageConvert_Planar16FtoPlanarF(&source, &destination, 0)
                var size = Int32(vocab)
                vDSP_maxvi(v.baseAddress!, 1, &maximum, &index, count)
                exponent.withUnsafeMutableBufferPointer { e in
                    var shift = -maximum
                    vDSP_vsadd(v.baseAddress!, 1, &shift, e.baseAddress!, 1, count)
                    vvexpf(e.baseAddress!, e.baseAddress!, &size)
                    vDSP_sve(e.baseAddress!, 1, &sum, count)
                }
            }
            return Row(nll: Double(maximum + log(sum) - values[Int(target)]), top1: Int(index))
        }

        func run(control: Bool) async throws -> (rows: [String: [Row]], seconds: [String: Double], tiles: Int) {
            let runner = try FlashNextForwardRunner(model: model, context: context, maxContext: longest + 64)
            if control { runner.useRowKernelForGGUFPrefill() }
            var rows: [String: [Row]] = [:], seconds: [String: Double] = [:]
            for item in items {
                runner.reset()
                let start = Date()
                _ = try await runner.prefillChunked(tokens: item.tokens[..<item.prefillCount], startPosition: 0,
                                                    outputMode: .logits, config: .production(chunkTokens: chunk),
                                                    into: logits) { _ in }
                seconds[item.name] = Date().timeIntervalSince(start)
                var itemRows = [analyze(target: item.tokens[item.prefillCount])]
                for position in item.prefillCount..<(item.tokens.count - 1) {
                    try await runner.produce(token: item.tokens[position], position: position, into: logits)
                    itemRows.append(analyze(target: item.tokens[position + 1]))
                }
                rows[item.name] = itemRows
            }
            return (rows, seconds, runner.prefillSimdLUT4Tiles)
        }

        let control = try await run(control: true)
        let candidate = try await run(control: false)
        #expect(control.tiles == 0, "control ran \(control.tiles) grouped tiles")
        #expect(candidate.tiles > 0, "candidate never ran the grouped LUT4 kernel")

        var differences: [Double] = [], baseSum = 0.0, sameTop = 0
        var saved: [[String: Any]] = []
        for item in items {
            let pairs = Array(zip(try #require(control.rows[item.name]), try #require(candidate.rows[item.name])))
            for (a, b) in pairs {
                differences.append(b.nll - a.nll)
                baseSum += a.nll
                sameTop += a.top1 == b.top1 ? 1 : 0
            }
            let itemMean = pairs.reduce(0.0) { $0 + $1.1.nll - $1.0.nll } / Double(pairs.count)
            let itemTop = Double(pairs.filter { $0.0.top1 == $0.1.top1 }.count) / Double(pairs.count)
            print(String(format: "[flashnext-lut4-gate] %@: prompt %d tok, chunk %d, prefill %.1f s -> %.1f s, dNLL=%+.5f top1=%.4f over %d",
                         item.name, item.prefillCount, chunk, control.seconds[item.name] ?? 0,
                         candidate.seconds[item.name] ?? 0, itemMean, itemTop, pairs.count))
            saved.append(["name": item.name, "base_nll": pairs.map(\.0.nll), "test_nll": pairs.map(\.1.nll),
                          "base_top1": pairs.map(\.0.top1), "test_top1": pairs.map(\.1.top1)])
        }
        let predictions = differences.count
        let meanDifference = differences.reduce(0, +) / Double(predictions)
        // Neighbouring positions share context: the interval uses 10-position batch means.
        let batches = stride(from: 0, to: predictions, by: 10).map { start -> Double in
            let batch = differences[start..<min(start + 10, predictions)]
            return batch.reduce(0, +) / Double(batch.count)
        }
        let batchMean = batches.reduce(0, +) / Double(batches.count)
        let batchVariance = batches.reduce(0) { $0 + ($1 - batchMean) * ($1 - batchMean) } / Double(batches.count - 1)
        let halfWidth = 1.96 * (batchVariance / Double(batches.count)).squareRoot()
        let agreement = Double(sameTop) / Double(predictions)
        let maxAbs = differences.map(abs).max() ?? 0
        print(String(format: "[flashnext-lut4-gate] row->grouped LUT4 chunk %d tiles %d predictions=%d base NLL=%.5f dNLL=%+.5f (95%% %+.5f..%+.5f) max|d|=%.4f top1 agreement=%.4f",
                     chunk, candidate.tiles, predictions, baseSum / Double(predictions), meanDifference,
                     meanDifference - halfWidth, meanDifference + halfWidth, maxAbs, agreement))
        if only == nil {
            let material = meanDifference > 0.005 && meanDifference - halfWidth > 0
            #expect(!material, "dNLL=\(meanDifference) +/- \(halfWidth)")
            #expect(meanDifference <= 0.02, "dNLL=\(meanDifference)")
            #expect(agreement >= 0.98, "agreement=\(agreement)")
        }
        if let path = Self.env["MFERENCE_FLASHNEXT_GATE_OUT"] {
            try JSONSerialization.data(withJSONObject: ["modelID": model.modelID, "chunk": chunk, "rows": saved])
                .write(to: URL(fileURLWithPath: path))
        }
    }
}
