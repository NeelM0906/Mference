import Accelerate
import Foundation
import Metal
import Testing
@testable import Mference

/// Quality gates for Qwen 3.6 kernels that reorder floating-point sums: the
/// GQA-shared prefill attention kernel (`attention_prefill_gqa8_d256_simd`),
/// the grouped-GEMM routed experts (`MPPGroupedRoutedMoE` and
/// `SimdgroupGroupedRoutedMoE`), and the batched decode full-attention kernel
/// (scored on the teacher-forced decode steps that follow each prefill).
///
/// Set `MFERENCE_QWEN36_PREFILL_GATE` to an installed `qwen36.gturbo`; the
/// install is read-only. In each test, control is the default runner with that
/// one kernel replaced by its predecessor (the tiled attention kernel; the row
/// kernel for every routed tile; MPP and the row kernel in place of the
/// simdgroup grouped kernel; or the per-component decode attention kernel);
/// candidate is the default runner, which selects
/// these kernels on GPUs without the Apple10 TensorOps path, with only the
/// kernel under test differing (the MPP gate's candidate turns the simdgroup
/// kernel off). Both
/// prefill the same prompts in 4,096-token chunks and are teacher-forced
/// through the same saved answers, so every difference comes from that kernel.
///
/// The items are the frozen community prompts with Qwen's and Gemma's answers,
/// and one prompt of about 8K tokens built from the repository's design docs,
/// whose second chunk attends to 4,096 cached keys.
///
/// The thresholds are the Gemma prefill gate's, all over every scored
/// position: fail on a material mean shift (> 0.005 nats per token) that is
/// also distinguishable from zero (batch-means 95 % interval), on a mean shift
/// above 0.02, or on top-1 agreement below 0.98. Per-item means and the
/// per-position max |shift| are printed as diagnostics, not asserted.
/// `MFERENCE_QWEN36_PREFILL_GATE_OUT` saves the rows, one key per gate in one
/// JSON document.
enum Qwen36PrefillGateFixture {
    static let answerNames = ["medium-review.qwen36", "long-synthesis.qwen36"]

    static func savedAnswer(_ name: String) throws -> String {
        let url = GemmaPrefillGateFixture.repositoryRoot
            .appendingPathComponent("Tests/Mference/Fixtures/qwen36-prefill-gate/answers.json")
        let answers = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: url))
        return try #require(answers[name], "missing saved answer \(name)")
    }
}

@Suite struct Qwen36PrefillGateFixtureTests {
    @Test(arguments: Qwen36PrefillGateFixture.answerNames)
    func savedAnswerShipsWithTheRepository(name: String) throws {
        let answer = try Qwen36PrefillGateFixture.savedAnswer(name)
        // The gate teacher-forces up to 300 tokens of each answer.
        #expect(answer.count >= 2000, "\(name) has only \(answer.count) characters")
    }
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MFERENCE_QWEN36_PREFILL_GATE"] != nil))
struct Qwen36PrefillEquivalenceGateTests {
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
        var dispatches = 0
    }

    private static func messages(_ url: URL) throws -> [MFTokenizer.Message] {
        let rows = try JSONDecoder().decode([[String: String]].self, from: Data(contentsOf: url))
        return try rows.map {
            MFTokenizer.Message(role: try #require(MFTokenizer.Role(rawValue: $0["role"] ?? "")),
                                content: $0["content"] ?? "")
        }
    }

    @Test func gqaPrefillAttentionKeepsTeacherForcedPerplexity() async throws {
        try await runGate(
            label: "tiled->gqa", outKey: "tiledToGQA",
            precondition: { model, context in
                let usesGQA = RealForwardRunner.usesQwenGQASimdAttention(
                    family: model.config.family, supportsApple10TensorOps: context.device.supportsApple10TensorOps)
                try #require(usesGQA, "the GQA kernel is the default only on GPUs without the Apple10 TensorOps path")
            },
            configureControl: { $0.prefillAttentionKernels.useTiledKernelForGQAShapes() },
            dispatches: { $0.prefillAttentionKernels.gqaSimdDispatches },
            checkDispatches: { control, candidate, itemCount in
                #expect(control == 0)
                // 10 full-attention layers per chunk: 1 chunk for each short item, 2 for the 8K item.
                #expect(candidate == 10 * (itemCount + 1), "candidate GQA dispatches \(candidate)")
            })
    }

    @Test func groupedRoutedExpertsKeepTeacherForcedPerplexity() async throws {
        try await runGate(
            label: "row->grouped", outKey: "rowToGrouped",
            precondition: { model, context in
                let selected = RealForwardRunner.usesGroupedRoutedExpertGEMM(
                    family: model.config.family, gemmaBatchedExperts: true,
                    supportsApple10TensorOps: context.device.supportsApple10TensorOps)
                try #require(selected, "grouped routed experts are the Qwen default only off Apple10")
                try #require(MPPGroupedRoutedMoE(context: context).isAvailable,
                             "the tensorops grouped kernel does not build on this host")
            },
            configureControl: { $0.useRowKernelForRoutedExperts() },
            // MPP alone: the simdgroup kernel, ahead of it by default, is off.
            configureCandidate: { $0.useMPPOrRowKernelForRoutedExperts() },
            dispatches: { $0.prefillGroupedExpertTiles },
            checkDispatches: { control, candidate, _ in
                #expect(control == 0)
                #expect(candidate > 0, "candidate grouped tiles \(candidate)")
            })
    }

    /// Decode full attention: the batched grouped kernel and its split
    /// (`groupedFull256Batched`) against the per-component grouped kernel at
    /// a fixed 16 chunks. Every teacher-forced decode step runs the new kernel:
    /// 16 chunks below 4,096 positions (the medium and long items), 32 from
    /// 4,096. This gate lengthens the docs item to 8,400 prompt tokens so its
    /// whole decode runs at 32 chunks past 8K.
    @Test func batchedDecodeAttentionKeepsTeacherForcedPerplexity() async throws {
        try await runGate(
            label: "per-component->batched decode attention", outKey: "decodeAttentionBatched",
            documentTokenCount: 8_400,
            precondition: { model, _ in
                try #require(model.config.family == .qwen36)
            },
            configureControl: { $0.usePerComponentDecodeAttention() },
            dispatches: { $0.decodeAttentionIsBatched ? 1 : 0 },
            checkDispatches: { control, candidate, _ in
                #expect(control == 0)
                #expect(candidate == 1)
            })
    }

    /// The simdgroup grouped kernel against the selection before it (MPP on
    /// tiles within 1.5x padding, the row kernel otherwise).
    @Test func simdgroupGroupedRoutedExpertsKeepTeacherForcedPerplexity() async throws {
        try await runGate(
            label: "mpp/row->simdgroup-grouped", outKey: "mppRowToSimdgroupGrouped",
            precondition: { model, context in
                let selected = RealForwardRunner.usesSimdgroupGroupedRoutedExperts(
                    family: model.config.family,
                    supportsApple10TensorOps: context.device.supportsApple10TensorOps)
                try #require(selected, "the simdgroup grouped kernel is the Qwen default only off Apple10")
            },
            configureControl: { $0.useMPPOrRowKernelForRoutedExperts() },
            dispatches: { $0.prefillSimdGroupedExpertTiles },
            checkDispatches: { control, candidate, _ in
                #expect(control == 0)
                #expect(candidate > 0, "candidate simdgroup grouped tiles \(candidate)")
            })
    }

    private func runGate(label: String, outKey: String, documentTokenCount: Int = 7_800,
                         precondition: (Model, MetalContext) throws -> Void,
                         configureControl: (RealForwardRunner) -> Void,
                         configureCandidate: (RealForwardRunner) -> Void = { _ in },
                         dispatches: (RealForwardRunner) -> Int,
                         checkDispatches: (_ control: Int, _ candidate: Int, _ itemCount: Int) -> Void)
        async throws {
        let install = URL(fileURLWithPath: try #require(
            ProcessInfo.processInfo.environment["MFERENCE_QWEN36_PREFILL_GATE"]))
        let root = GemmaPrefillGateFixture.repositoryRoot
        let prompts = root.appendingPathComponent("docs/benchmark-prompts/real-generation-v1")
        let tokenizer = try await MFTokenizer.load(forModelDirectory: install)
        func answerTokens(_ answer: String, count: Int) throws -> [Int32] {
            let text = answer.hasSuffix(".qwen36")
                ? try Qwen36PrefillGateFixture.savedAnswer(answer)
                : try GemmaPrefillGateFixture.savedAnswer(answer)
            let tail = tokenizer.encode(text, addBOS: false)
            try #require(tail.count >= count)
            return Array(tail.prefix(count))
        }
        func item(_ name: String, messages: [MFTokenizer.Message], answer: String, continuation: Int) throws -> Item {
            let head = try tokenizer.encodeChat(messages: messages)
            return Item(name: name, tokens: head + (try answerTokens(answer, count: continuation)),
                        prefillCount: head.count)
        }
        func chat(_ name: String, prompt: String, answer: String, continuation: Int) throws -> Item {
            try item(name, messages: try Self.messages(prompts.appendingPathComponent(prompt)),
                     answer: answer, continuation: continuation)
        }

        // About 8K prompt tokens of the repository's own design docs, so the
        // second 4,096-token chunk attends to a full first chunk of cached keys.
        var documents = ""
        for path in ["docs/SYSTEM_DESIGN.md", "docs/FAMILY_CONTRACT.md", "docs/RUNTIME_CONTROLS.md",
                     "docs/OPENAI_SERVER.md", "docs/BENCHMARKS.md"] {
            let text = try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
            documents += "<document path=\"\(path)\">\n\(text)\n</document>\n\n"
            if tokenizer.encode(documents, addBOS: false).count >= documentTokenCount { break }
        }
        var documentTokens = tokenizer.encode(documents, addBOS: false)
        try #require(documentTokens.count >= documentTokenCount, "the design docs are shorter than the 8K item needs")
        documentTokens = Array(documentTokens.prefix(documentTokenCount))
        let longQuestion = tokenizer.decode(documentTokens)
            + "\n\nSummarize how the documents above keep memory bounded while serving a long prompt."

        let items = [
            try chat("medium+qwen-answer", prompt: "medium-review.json",
                     answer: "medium-review.qwen36", continuation: 300),
            try chat("medium+gemma-answer", prompt: "medium-review.json",
                     answer: "medium-review.original", continuation: 300),
            try chat("long+qwen-answer", prompt: "long-synthesis.json",
                     answer: "long-synthesis.qwen36", continuation: 200),
            try chat("long+gemma-answer", prompt: "long-synthesis.json",
                     answer: "long-synthesis.original", continuation: 200),
            try item("docs-8k+qwen-answer", messages: [MFTokenizer.Message(role: .user, content: longQuestion)],
                     answer: "long-synthesis.qwen36", continuation: 200),
        ]
        let longest = items.map(\.tokens.count).max() ?? 0
        try #require(items.last!.prefillCount > 4096, "the 8K item must span two prefill chunks")

        let context = try MetalContext()
        let model = try Model.load(directoryURL: install, device: context.device, expecting: .qwen36_35B_A3B,
                                   streamingMode: .pread(slotCount: 96))
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
                var size = Int32(vocab)
                if softcap > 0 {
                    var divisor = softcap
                    vDSP_vsdiv(v.baseAddress!, 1, &divisor, v.baseAddress!, 1, count)
                    vvtanhf(v.baseAddress!, v.baseAddress!, &size)
                    vDSP_vsmul(v.baseAddress!, 1, &divisor, v.baseAddress!, 1, count)
                }
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

        func run(control: Bool) async throws -> Outcome {
            let runner = try RealForwardRunner(
                model: model, context: context, maxContext: longest + 64,
                runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 4096, forceLogitsHead: true))
            if control { configureControl(runner) } else { configureCandidate(runner) }
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
            outcome.dispatches = dispatches(runner)
            return outcome
        }

        try precondition(model, context)
        let control = try await run(control: true)
        let candidate = try await run(control: false)
        checkDispatches(control.dispatches, candidate.dispatches, items.count)

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
            print(String(format: "[qwen36-prefill-gate] %@: prompt %d tok, prefill %.1f s -> %.1f s, dNLL=%+.5f top1=%.4f over %d",
                         item.name, item.prefillCount, control.prefillSeconds[item.name] ?? 0,
                         candidate.prefillSeconds[item.name] ?? 0, itemMean, itemTop, pairs.count))
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
        let maxAbs = differences.map(abs).max() ?? 0
        print(String(format: "[qwen36-prefill-gate] %@ %@ predictions=%d base NLL=%.5f dNLL=%+.5f (95%% %+.5f..%+.5f) max|d|=%.4f top1 agreement=%.4f",
                     label, model.modelID, predictions, baseSum / Double(predictions), meanDifference,
                     meanDifference - halfWidth, meanDifference + halfWidth, maxAbs, agreement))
        let material = meanDifference > 0.005 && meanDifference - halfWidth > 0
        #expect(!material, "dNLL=\(meanDifference) +/- \(halfWidth)")
        #expect(meanDifference <= 0.02, "dNLL=\(meanDifference)")
        #expect(agreement >= 0.98, "agreement=\(agreement)")
        if let path = ProcessInfo.processInfo.environment["MFERENCE_QWEN36_PREFILL_GATE_OUT"] {
            // Both gates share this file: keep the other gate's rows for the
            // same model and replace only this gate's key.
            let url = URL(fileURLWithPath: path)
            var document: [String: Any] = [:]
            if let data = try? Data(contentsOf: url),
               let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               existing["modelID"] as? String == model.modelID {
                document = existing
            }
            document["modelID"] = model.modelID
            document[outKey] = saved
            try JSONSerialization.data(withJSONObject: document).write(to: url)
        }
    }
}
