import Testing
import Foundation
import Metal
@testable import Mference

/// W2.1b — the **model-level** quantizer quality gate (spec
/// `docs/superpowers/specs/2026-08-08-family-bringup-kit-design.md`, Workstream
/// 2 deliverable 1 gate (b)). Method and results:
/// `docs/QUANTIZER_QUALITY.md`.
///
/// The measurement compares two installs of the *same* Qwen 3.6 checkpoint:
///   * **trusted control** — `qwen36`, mlx-community's pre-quantized conversion,
///     re-laid-out by the repacker;
///   * **ours** — `qwen36original`, the vendor's BF16 upload put through
///     `StreamingInt4Quantizer` / `Int4AffineEncoder` at install time.
///
/// # Why this suite dumps to disk instead of comparing in-process
///
/// AGENTS.md forbids two model processes at once, and a 35B MoE will not have
/// two copies resident regardless. So each model is run **alone**, writes its
/// logits to disk, and the comparison happens offline
/// (`Scripts/quantizer-quality-compare.py`). Nothing here ever loads both.
///
/// # Why teacher forcing, not free-running rollouts
///
/// Two rollouts diverge after the first token they disagree on, and every
/// position after that is conditioned on different history — the resulting
/// distributions are not comparable, so a "KLD" over them measures divergence
/// of *contexts*, not damage from quantization. This suite therefore drives
/// both models over one **fixed** token sequence (prompt + the trusted model's
/// own greedy continuation) and captures the full-vocab distribution at every
/// position. Free-running rollouts are still captured, but only for the
/// separate token-exactness report.
///
/// # Dump format
///
/// Logits are written as **raw FP16, exactly as the runner produced them**, not
/// widened to FP32. That is lossless (FP16 is the runner's own output
/// precision), halves the file, and makes the noise-floor check exact: two
/// deterministic runs must produce byte-identical dumps. The comparator widens
/// to FP32 before the softmax, and keeps the **full** vocabulary rather than a
/// top-K slice, so the reported KL divergence is exact rather than a bound.
///
/// # Env gates (skips cleanly when absent)
///
///   * `MFERENCE_QUANT_QUALITY_GTURBO` — install to run. One per process.
///   * `MFERENCE_QUANT_QUALITY_DUMP`   — directory to write into.
///   * `MFERENCE_QUANT_QUALITY_LABEL`  — subdirectory name, e.g. `trusted`,
///     `ours`, `trusted-repeat`.
///   * `MFERENCE_QUANT_QUALITY_TEACHER` — optional path to a `tokens.json`
///     written by an earlier run. When set, this run teacher-forces **those**
///     sequences instead of its own; that is what makes the comparison valid.
///     The first (trusted) run leaves it unset and publishes the file.
///   * `MFERENCE_QUANT_QUALITY_NEW` — continuation length, default 64.
@Suite(.serialized) struct QuantizerQualityMeasurement {

    // MARK: - Corpus

    /// One prompt in the fixed corpus.
    private struct CorpusItem: Codable {
        let name: String
        /// Prompt token ids after the chat template (raw prompts skip it).
        var promptIDs: [Int32]
        /// Prompt + continuation. This is what gets teacher-forced.
        var sequence: [Int32]
        /// Where the continuation starts inside `sequence`.
        var continuationStart: Int
    }

    private struct TeacherCorpus: Codable {
        let label: String
        let items: [CorpusItem]
    }

    /// The frozen benchmark prompts plus a few short raw ones. The frozen
    /// prompts' content is truncated to `contentCap` characters before
    /// templating: the full long-synthesis prompt is ~900 tokens, and a
    /// full-vocab logit dump costs ~485 KB per position, so the untruncated
    /// corpus would write several GB per run for no extra signal.
    private static let contentCap = 900

    private static let rawPrompts: [(String, String)] = [
        ("raw-arith", "2 + 2 ="),
        ("raw-capital", "The capital of France is"),
        ("raw-list", "Three primary colors are red, blue, and"),
    ]

    private static func frozenPromptFiles() -> [URL] {
        // Tests run from the package root in CI and locally.
        let dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("docs/benchmark-prompts/real-generation-v1")
        let found = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        return found.filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func buildCorpus(tokenizer: MFTokenizer) throws -> [CorpusItem] {
        var items: [CorpusItem] = []
        for file in frozenPromptFiles() {
            let data = try Data(contentsOf: file)
            guard let rows = try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
                  let content = rows.first?["content"] as? String else { continue }
            let clipped = String(content.prefix(contentCap))
            let rendered = try tokenizer.applyChatTemplate(
                [MFTokenizer.Message(role: .user, content: clipped)])
            let ids = tokenizer.encode(rendered, addBOS: false).map { Int32($0) }
            items.append(CorpusItem(name: file.deletingPathExtension().lastPathComponent,
                                    promptIDs: ids, sequence: ids,
                                    continuationStart: ids.count))
        }
        for (name, text) in rawPrompts {
            let ids = tokenizer.encode(text, addBOS: false).map { Int32($0) }
            items.append(CorpusItem(name: name, promptIDs: ids, sequence: ids,
                                    continuationStart: ids.count))
        }
        return items
    }

    // MARK: - Harness

    private struct Harness {
        let context: MetalContext
        let model: Model
        let tokenizer: MFTokenizer
        let runner: any ContinuableLogitProducer
        let vocab: Int
        let logits: MTLBuffer
    }

    private static func env(_ key: String) -> String? {
        let value = ProcessInfo.processInfo.environment[key]
        return (value?.isEmpty ?? true) ? nil : value
    }

    private static func loadHarness() async throws -> Harness? {
        guard let path = env("MFERENCE_QUANT_QUALITY_GTURBO") else { return nil }
        let modelURL = URL(fileURLWithPath: path)
        let context = try MetalContext()
        // Production door: Qwen 3.6 has a runner, so nothing is bypassed here.
        let model = try Model.load(directoryURL: modelURL, device: context.device)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: modelURL)
        // `forceLogitsHead` is NOT optional here. The default fused head skips
        // the 512 KB logits write entirely and leaves only a greedy argmax in
        // `lastGreedyToken` — a KLD harness that forgot this would dump
        // never-written memory and report confident nonsense.
        let runtime = try ForwardRunnerFactory.make(
            model: model,
            context: context,
            maxContext: 4096,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let vocab = model.config.vocabSize
        guard let buffer = context.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.size, options: .storageModeShared) else {
            throw ModelError.indexCorrupt(detail: "could not allocate a logits buffer")
        }
        return Harness(context: context, model: model, tokenizer: tokenizer,
                       runner: runtime.producer, vocab: vocab, logits: buffer)
    }

    /// One forward step. Returns the runner's FP16 logits verbatim.
    private static func step(_ h: Harness, token: Int32, position: Int) async throws
        -> UnsafeBufferPointer<Float16> {
        try await h.runner.produce(token: token, position: position, into: h.logits)
        let base = h.logits.contents().assumingMemoryBound(to: Float16.self)
        return UnsafeBufferPointer(start: base, count: h.vocab)
    }

    private static func argmax(_ values: UnsafeBufferPointer<Float16>) -> Int32 {
        var best = 0
        var bestValue = values[0]
        for i in 1..<values.count where values[i] > bestValue {
            bestValue = values[i]
            best = i
        }
        return Int32(best)
    }

    /// Greedy continuation of `item.promptIDs`. Pure argmax over the raw FP16
    /// logits — Qwen 3.6 has no final-logit softcap, so no adjustment applies,
    /// and keeping it argmax-only means both installs are scored by exactly the
    /// same rule.
    private static func greedyRollout(_ h: Harness,
                                      item: CorpusItem,
                                      newTokens: Int) async throws
        -> (tokens: [Int32], margins: [Float]) {
        h.runner.reset()
        var position = 0
        var next: Int32 = 0
        var margins: [Float] = []
        for token in item.promptIDs {
            let logits = try await step(h, token: token, position: position)
            position += 1
            if position == item.promptIDs.count {
                next = argmax(logits)
                margins.append(top2Margin(logits))
            }
        }
        var produced: [Int32] = []
        for _ in 0..<newTokens {
            produced.append(next)
            let logits = try await step(h, token: next, position: position)
            position += 1
            next = argmax(logits)
            margins.append(top2Margin(logits))
        }
        return (produced, margins)
    }

    /// Gap between the best and second-best logit. METH-01: a top-1 flip at a
    /// margin below the measured noise floor is not evidence of damage.
    private static func top2Margin(_ values: UnsafeBufferPointer<Float16>) -> Float {
        var best = -Float.greatestFiniteMagnitude
        var second = -Float.greatestFiniteMagnitude
        for i in 0..<values.count {
            let v = Float(values[i])
            if v > best { second = best; best = v } else if v > second { second = v }
        }
        return best - second
    }

    /// Teacher-force `item.sequence` and append every position's full-vocab
    /// FP16 logits to `handle`. Position *p* holds the distribution that
    /// predicts `sequence[p + 1]`.
    private static func teacherForce(_ h: Harness,
                                     item: CorpusItem,
                                     handle: FileHandle) async throws {
        h.runner.reset()
        for (position, token) in item.sequence.enumerated() {
            let logits = try await step(h, token: token, position: position)
            handle.write(Data(buffer: logits))
        }
    }

    // MARK: - The measurement

    @Test("W2.1b: dump greedy rollouts and teacher-forced logits for one install")
    func dumpLogits() async throws {
        guard let dumpRoot = Self.env("MFERENCE_QUANT_QUALITY_DUMP"),
              let label = Self.env("MFERENCE_QUANT_QUALITY_LABEL"),
              let h = try await Self.loadHarness() else {
            // No env gate: nothing installed for this measurement, skip cleanly.
            return
        }
        let newTokens = Int(Self.env("MFERENCE_QUANT_QUALITY_NEW") ?? "") ?? 64
        let outDir = URL(fileURLWithPath: dumpRoot).appendingPathComponent(label)
        try FileManager.default.createDirectory(at: outDir,
                                                withIntermediateDirectories: true)

        var corpus = try Self.buildCorpus(tokenizer: h.tokenizer)
        #expect(!corpus.isEmpty, "corpus is empty; frozen prompts were not found")

        // 1. Greedy rollout, for the token-exactness report.
        var rollouts: [String: [Int32]] = [:]
        var rolloutMargins: [String: [Float]] = [:]
        for index in corpus.indices {
            let (tokens, margins) = try await Self.greedyRollout(h, item: corpus[index],
                                                            newTokens: newTokens)
            rollouts[corpus[index].name] = tokens
            rolloutMargins[corpus[index].name] = margins
        }
        try JSONEncoder().encode(rollouts)
            .write(to: outDir.appendingPathComponent("rollout.json"))
        try JSONEncoder().encode(rolloutMargins)
            .write(to: outDir.appendingPathComponent("rollout_margins.json"))

        // 2. Decide what to teacher-force. The first run publishes its own
        //    sequences; every later run must consume that file, or the two
        //    dumps would be conditioned on different histories and the KL
        //    numbers would be meaningless.
        if let teacherPath = Self.env("MFERENCE_QUANT_QUALITY_TEACHER") {
            let data = try Data(contentsOf: URL(fileURLWithPath: teacherPath))
            let published = try JSONDecoder().decode(TeacherCorpus.self, from: data)
            let byName = Dictionary(uniqueKeysWithValues:
                published.items.map { ($0.name, $0) })
            for index in corpus.indices {
                let name = corpus[index].name
                let item = try #require(byName[name],
                                        "teacher corpus is missing \(name)")
                #expect(item.promptIDs == corpus[index].promptIDs,
                        "\(name): prompt tokenized differently than the teacher run")
                corpus[index] = item
            }
        } else {
            for index in corpus.indices {
                corpus[index].continuationStart = corpus[index].promptIDs.count
                corpus[index].sequence = corpus[index].promptIDs
                    + (rollouts[corpus[index].name] ?? [])
            }
            try JSONEncoder().encode(TeacherCorpus(label: label, items: corpus))
                .write(to: outDir.appendingPathComponent("tokens.json"))
        }

        // 3. Teacher-force and dump.
        var manifest: [[String: Any]] = []
        for item in corpus {
            let file = outDir.appendingPathComponent("\(item.name).logits.f16")
            FileManager.default.createFile(atPath: file.path, contents: nil)
            let handle = try FileHandle(forWritingTo: file)
            try await Self.teacherForce(h, item: item, handle: handle)
            try handle.close()
            manifest.append([
                "name": item.name,
                "positions": item.sequence.count,
                "vocab": h.vocab,
                "continuationStart": item.continuationStart,
                "file": file.lastPathComponent,
            ])
        }
        let meta: [String: Any] = [
            "label": label,
            "install": Self.env("MFERENCE_QUANT_QUALITY_GTURBO") ?? "",
            "modelID": h.model.modelID,
            "vocab": h.vocab,
            "dtype": "float16",
            "newTokens": newTokens,
            "items": manifest,
        ]
        try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted])
            .write(to: outDir.appendingPathComponent("meta.json"))

        FileHandle.standardError.write(Data(
            "[quant-quality] wrote \(manifest.count) dumps for \(label) to \(outDir.path)\n".utf8))
    }
}
