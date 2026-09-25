import Accelerate
import Foundation
import Metal
import Testing
@testable import Mference

/// Quality gate for prefill kernels that reorder floating-point sums.
///
/// Set `MFERENCE_GEMMA_PREFILL_GATE` to an installed `gemma4.gturbo` or
/// `gemma4qat.gturbo`; the install is read-only. Control is the per-row,
/// source-exact prefill with non-tensor attention shipped before 2026-09-21;
/// candidate is the default. A third run, the default with non-tensor
/// attention, isolates the tensor-ops full-attention kernel. All three prefill
/// the frozen community prompts and are teacher-forced through the same saved
/// answers, so every difference comes from the prefill kernels.
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
        var tensorOpsAttentionLayers = 0
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

        func run(_ environment: [String: String],
                 attention: RuntimePrefillAttentionPath) async throws -> Outcome {
            let runner = try RealForwardRunner(
                model: model, context: context, maxContext: 4096,
                runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 4096,
                                                           prefillAttentionPath: attention,
                                                           forceLogitsHead: true),
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
            outcome.tensorOpsAttentionLayers = runner.prefillTensorOpsAttentionLayers
            return outcome
        }

        let control = try await run(["MFERENCE_QAT_EXACT_PREFILL": "1", "MFERENCE_GEMMA_PREFILL_LEGACY": "1"],
                                    attention: .causalTiled)
        let previous = try await run([:], attention: .causalTiled)
        let candidate = try await run([:], attention: .fullTensorOps2DPreferred)

        #expect(control.groupedTiles == 0)
        #expect(control.sharedPath == .repeatedRows)
        #expect(control.tensorOpsAttentionLayers == 0)
        #expect(previous.tensorOpsAttentionLayers == 0)
        #expect(candidate.groupedTiles > 0, "the 3,015-token prompt fills grouped-GEMM tiles")
        #expect(candidate.sharedPath == .tensorOpsInt4, "the INT4 shared expert must run batched")
        #expect(candidate.tensorOpsAttentionLayers > 0, """
            full-attention prefill fell back from the tensor-ops kernel; it needs macOS 26 (MSL 4.0) \
            and a GPU where the pipeline builds
            """)

        func compare(_ label: String, _ base: Outcome, _ test: Outcome) throws -> [[String: Any]] {
            var differences: [Double] = [], baseSum = 0.0, sameTop = 0
            var saved: [[String: Any]] = []
            for item in items {
                let pairs = Array(zip(try #require(base.rows[item.name]), try #require(test.rows[item.name])))
                for (a, b) in pairs {
                    differences.append(b.nll - a.nll)
                    baseSum += a.nll
                    sameTop += a.top1 == b.top1 ? 1 : 0
                }
                let itemMean = pairs.reduce(0.0) { $0 + $1.1.nll - $1.0.nll } / Double(pairs.count)
                print(String(format: "[prefill-gate] %@ %@: prefill %.1f s -> %.1f s, dNLL=%+.5f over %d", label,
                             item.name, base.prefillSeconds[item.name] ?? 0, test.prefillSeconds[item.name] ?? 0,
                             itemMean, pairs.count))
                saved.append(["name": item.name, "base_nll": pairs.map(\.0.nll), "test_nll": pairs.map(\.1.nll),
                              "base_top1": pairs.map(\.0.top1), "test_top1": pairs.map(\.1.top1)])
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
            print(String(format: "[prefill-gate] %@ %@ predictions=%d base NLL=%.5f dNLL=%+.5f (95%% %+.5f..%+.5f) top1 agreement=%.4f",
                         label, model.modelID, predictions, baseSum / Double(predictions), meanDifference,
                         meanDifference - halfWidth, meanDifference + halfWidth, agreement))
            let material = meanDifference > 0.005 && meanDifference - halfWidth > 0
            #expect(!material, "\(label) dNLL=\(meanDifference) +/- \(halfWidth)")
            #expect(meanDifference <= 0.02, "\(label) dNLL=\(meanDifference)")
            #expect(agreement >= 0.98, "\(label) agreement=\(agreement)")
            return saved
        }

        print("[prefill-gate] grouped tiles=\(candidate.groupedTiles) tensor-ops attention layers=\(candidate.tensorOpsAttentionLayers)")
        let againstControl = try compare("control->candidate", control, candidate)
        // The previous default differs from the candidate only in full-attention prefill.
        let againstPrevious = try compare("previous->candidate", previous, candidate)
        if let path = ProcessInfo.processInfo.environment["MFERENCE_GEMMA_PREFILL_GATE_OUT"] {
            try JSONSerialization.data(withJSONObject: ["modelID": model.modelID,
                                                        "controlToCandidate": againstControl,
                                                        "previousToCandidate": againstPrevious])
                .write(to: URL(fileURLWithPath: path))
        }
    }
}
