import Accelerate
import Foundation
import Metal
import Testing
@testable import Mference

/// Quality gate for prefill kernels that reorder floating-point sums.
///
/// Set `MFERENCE_GEMMA_PREFILL_GATE` to an installed `gemma4.gturbo` or
/// `gemma4qat.gturbo`; the install is read-only. Control is the per-row,
/// source-exact prefill shipped before 2026-09-21; candidate is the default.
/// Both prefill the frozen community prompts and are teacher-forced through
/// the same saved answers, so every difference comes from the prefill kernels.
///
/// MoE routing makes tiny kernel differences flip near-tied experts, so two
/// correct schedules disagree position by position while agreeing on average.
/// The gate therefore fails only on a material mean shift (> 0.005 nats per
/// token) that is also distinguishable from zero (batch-means 95 % interval),
/// or on any shift above 0.02. `MFERENCE_GEMMA_PREFILL_GATE_OUT` saves rows.
enum GemmaPrefillGateFixture {
    static let answerNames = ["medium-review.qat", "medium-review.original",
                              "long-synthesis.qat", "long-synthesis.original"]

    static var repositoryRoot: URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { root.deleteLastPathComponent() }
        return root
    }

    /// Answers the two Gemma checkpoints gave to the frozen community prompts;
    /// see `Tests/Mference/Fixtures/gemma4-prefill-gate/README.md`.
    static func savedAnswer(_ name: String) throws -> String {
        let url = repositoryRoot
            .appendingPathComponent("Tests/Mference/Fixtures/gemma4-prefill-gate/answers.json")
        let answers = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: url))
        return try #require(answers[name], "missing saved answer \(name)")
    }
}

/// The gate below only runs against an installed checkpoint, so this ordinary
/// test is what notices a saved answer missing from the repository.
@Suite struct GemmaPrefillGateFixtureTests {
    @Test(arguments: GemmaPrefillGateFixture.answerNames)
    func savedAnswerShipsWithTheRepository(name: String) throws {
        let answer = try GemmaPrefillGateFixture.savedAnswer(name)
        // The gate teacher-forces up to 300 tokens of each answer.
        #expect(answer.count >= 2000, "\(name) has only \(answer.count) characters")
    }
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_PREFILL_GATE"] != nil))
struct GemmaPrefillEquivalenceGateTests {
    struct Item {
        let name: String
        let tokens: [Int32]
        let prefillCount: Int
    }

    struct Row {
        let nll: Double
        let top1: Int
    }

    struct Outcome {
        var rows: [String: [Row]] = [:]
        var prefillSeconds: [String: Double] = [:]
        var groupedTiles = 0
        var sharedPath: PrefillSharedExpert.BlockPath?
    }

    private static func messages(_ url: URL) throws -> [MFTokenizer.Message] {
        let rows = try JSONDecoder().decode([[String: String]].self, from: Data(contentsOf: url))
        return try rows.map {
            MFTokenizer.Message(role: try #require(MFTokenizer.Role(rawValue: $0["role"] ?? "")),
                                content: $0["content"] ?? "")
        }
    }

    @Test func batchedPrefillKeepsTeacherForcedPerplexity() async throws {
        let install = URL(fileURLWithPath: try #require(
            ProcessInfo.processInfo.environment["MFERENCE_GEMMA_PREFILL_GATE"]))
        let prompts = GemmaPrefillGateFixture.repositoryRoot
            .appendingPathComponent("docs/benchmark-prompts/real-generation-v1")
        let tokenizer = try await MFTokenizer.load(forModelDirectory: install)
        func chat(_ name: String, prompt: String, answer: String, continuation: Int) throws -> Item {
            let head = try tokenizer.encodeChat(messages: try Self.messages(prompts.appendingPathComponent(prompt)))
            let tail = tokenizer.encode(try GemmaPrefillGateFixture.savedAnswer(answer), addBOS: false)
            try #require(tail.count >= continuation)
            return Item(name: name, tokens: head + Array(tail.prefix(continuation)), prefillCount: head.count)
        }
        // Answers sampled from both checkpoints, so neither is only scored on its own text.
        let items = [
            try chat("medium+qat-answer", prompt: "medium-review.json",
                     answer: "medium-review.qat", continuation: 300),
            try chat("medium+original-answer", prompt: "medium-review.json",
                     answer: "medium-review.original", continuation: 300),
            try chat("long+qat-answer", prompt: "long-synthesis.json",
                     answer: "long-synthesis.qat", continuation: 200),
            try chat("long+original-answer", prompt: "long-synthesis.json",
                     answer: "long-synthesis.original", continuation: 200),
        ]

        let context = try MetalContext()
        let model = try Model.load(directoryURL: install, device: context.device, expecting: .gemma4_26B_A4B)
        let vocab = model.config.vocabSize
        let softcap = Float(model.config.finalLogitSoftcap)
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
                var divisor = softcap, size = Int32(vocab)
                vDSP_vsdiv(v.baseAddress!, 1, &divisor, v.baseAddress!, 1, count)
                vvtanhf(v.baseAddress!, v.baseAddress!, &size)
                vDSP_vsmul(v.baseAddress!, 1, &divisor, v.baseAddress!, 1, count)
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

        func run(_ environment: [String: String]) async throws -> Outcome {
            let runner = try RealForwardRunner(
                model: model, context: context, maxContext: 4096,
                runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 4096, forceLogitsHead: true),
                gemmaPrefillPolicy: GemmaPrefillPolicy(modelID: model.modelID, environment: environment))
            var outcome = Outcome()
            for item in items {
                runner.reset()
                let start = Date()
                _ = try await runner.prefillChunked(tokens: item.tokens[..<item.prefillCount],
                    startPosition: 0, outputMode: .logits, config: .production(chunkTokens: 4096),
                    into: logits, onProgress: { _ in })
                outcome.prefillSeconds[item.name] = Date().timeIntervalSince(start)
                var rows = [analyze(target: item.tokens[item.prefillCount])]
                for position in item.prefillCount..<(item.tokens.count - 1) {
                    try await runner.produce(token: item.tokens[position], position: position, into: logits)
                    rows.append(analyze(target: item.tokens[position + 1]))
                }
                outcome.rows[item.name] = rows
            }
            outcome.groupedTiles = runner.prefillGroupedExpertTiles
            outcome.sharedPath = runner.lastPrefillSharedExpertPath
            return outcome
        }

        let control = try await run(["MFERENCE_QAT_EXACT_PREFILL": "1", "MFERENCE_GEMMA_PREFILL_LEGACY": "1"])
        let candidate = try await run([:])

        #expect(control.groupedTiles == 0)
        #expect(control.sharedPath == .repeatedRows)
        #expect(candidate.groupedTiles > 0, "the 3,015-token prompt fills grouped-GEMM tiles")
        #expect(candidate.sharedPath == .tensorOpsInt4, "the INT4 shared expert must run batched")

        var differences: [Double] = [], controlSum = 0.0, sameTop = 0
        var saved: [[String: Any]] = []
        for item in items {
            let pairs = Array(zip(try #require(control.rows[item.name]), try #require(candidate.rows[item.name])))
            for (a, b) in pairs {
                differences.append(b.nll - a.nll)
                controlSum += a.nll
                sameTop += a.top1 == b.top1 ? 1 : 0
            }
            let itemMean = pairs.reduce(0.0) { $0 + $1.1.nll - $1.0.nll } / Double(pairs.count)
            print(String(format: "[prefill-gate] %@: prefill %.1f s -> %.1f s, dNLL=%+.5f over %d", item.name,
                         control.prefillSeconds[item.name] ?? 0, candidate.prefillSeconds[item.name] ?? 0,
                         itemMean, pairs.count))
            saved.append(["name": item.name, "control_nll": pairs.map(\.0.nll), "candidate_nll": pairs.map(\.1.nll),
                          "control_top1": pairs.map(\.0.top1), "candidate_top1": pairs.map(\.1.top1)])
        }
        if let path = ProcessInfo.processInfo.environment["MFERENCE_GEMMA_PREFILL_GATE_OUT"] {
            try JSONSerialization.data(withJSONObject: ["modelID": model.modelID, "items": saved])
                .write(to: URL(fileURLWithPath: path))
        }
        let predictions = differences.count
        let meanDifference = differences.reduce(0, +) / Double(predictions)
        // Neighbouring positions share context, so the interval uses means of
        // 10-position batches rather than treating positions as independent.
        let batches = stride(from: 0, to: predictions, by: 10).map { start -> Double in
            let batch = differences[start..<min(start + 10, predictions)]
            return batch.reduce(0, +) / Double(batch.count)
        }
        let batchMean = batches.reduce(0, +) / Double(batches.count)
        let batchVariance = batches.reduce(0) { $0 + ($1 - batchMean) * ($1 - batchMean) } / Double(batches.count - 1)
        let halfWidth = 1.96 * (batchVariance / Double(batches.count)).squareRoot()
        let agreement = Double(sameTop) / Double(predictions)
        print(String(format: "[prefill-gate] %@ predictions=%d control NLL=%.5f dNLL=%+.5f (95%% %+.5f..%+.5f) top1 agreement=%.4f grouped tiles=%d",
                     model.modelID, predictions, controlSum / Double(predictions), meanDifference,
                     meanDifference - halfWidth, meanDifference + halfWidth, agreement, candidate.groupedTiles))
        let material = meanDifference > 0.005 && meanDifference - halfWidth > 0
        #expect(!material, "dNLL=\(meanDifference) +/- \(halfWidth)")
        #expect(meanDifference <= 0.02, "dNLL=\(meanDifference)")
        #expect(agreement >= 0.98, "agreement=\(agreement)")
    }
}
