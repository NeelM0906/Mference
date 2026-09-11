import Testing
import Foundation
import Metal
@testable import Mference

/// **Measurement** harness for the `qwen38flashnext` production runner
/// (`FlashNextForwardRunner`) against the REAL install: greedy continuation,
/// chat, and a needle-in-haystack probe past the indexer's 2,048 budget, each
/// reporting decode tok/s and peak RSS.
///
/// It is a measurement harness, not a gate: the capability gate was lifted for
/// this family on 2026-09-10 (`ManifestReader.familiesWithoutRunner` is empty),
/// so the CLI, the server and `Model.load`'s auto-detect overload all load this
/// install through the ordinary production funnel —
/// `productionDoorLoadsRealInstall` below asserts exactly that, and
/// `FlashNextCapabilityGateTests` covers the gate itself.
///
/// Two entry points here predate the lift and are kept deliberately, because
/// they pin the baseline and tokenizer this suite measures against rather than
/// whatever auto-detection resolves:
///   * `Model.load(directoryURL:device:expecting:)` — the overload that takes an
///     explicit `ArchConfig` baseline. The same internal door the FlashNext
///     reference/parity tie-back tests use, so a baseline drift shows up as a
///     validation failure here instead of a silent re-detect.
///   * `MFTokenizer.load(from:family:)` on the sidecar `tokenizer/` folder.
/// Everything downstream (`ForwardRunnerFactory.make`, `runRawCompletion`,
/// sampling, the timing footer) is the production generation machinery,
/// unmodified.
///
/// Env-gated, skipped without the gate:
///   * `MFERENCE_FLASHNEXT_GTURBO` — path to the verified install dir (~163 GiB).
///   * `MFERENCE_FLASHNEXT_VERIFY=trusted-receipt` — after one full-SHA run, skip
///     the ~163 GiB first-touch SHA-256 in favour of the install receipt's size
///     checks (`ModelIntegrityPolicy.sizeCheckTrustedReceipt`). Defaults to the
///     strict `.fullSha256`.
///
/// Quality caveat carried in every report: greedy token-exactness vs a
/// reference rollout cannot be checked at 180B scale (no reference rollout
/// exists), so coherent output is a *read* of the kernels at scale, not a
/// proof. W2.1b — the quantitative quality gate — closed on both halves on
/// 2026-09-02 (docs/QUANTIZER_QUALITY.md) against Qwen 3.6, which this family
/// inherits through the shared quantizer nucleus; the two caveats it leaves
/// specific to Flash-Next (uniform INT4 routers, no norm-bias fold, neither
/// checked against an independent conversion because none exists) are recorded
/// as under measurement in docs/families/QWEN38_FLASH_NEXT.md.
@Suite(.serialized) struct FlashNextRealGenerationMeasurement {

    private struct Harness {
        let context: MetalContext
        let model: Model
        let tokenizer: MFTokenizer
        let forwardRuntime: ForwardRuntime
        let runner: any ContinuableLogitProducer
        let firstLoadSeconds: Double
        let verifyMode: String
    }

    private static func installPath() -> String? {
        ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_GTURBO"]
    }

    /// Load the real install against an explicitly pinned baseline and build the
    /// production `FlashNextForwardRunner` via the real factory. Returns nil when
    /// the env gate is absent so the suite skips cleanly.
    private static func loadHarness(maxContext: Int) async throws -> Harness? {
        guard let path = installPath() else { return nil }
        let modelURL = URL(fileURLWithPath: path)
        let context = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.qwen38flashnext],
                               "the baseline the runner is built against is missing")

        let verifyMode = ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_VERIFY"]
            ?? "full-sha256"
        let integrity: ModelIntegrityPolicy = verifyMode == "trusted-receipt"
            ? .sizeCheckTrustedReceipt : .fullSha256

        // Production expert-streaming defaults; INT4 experts stream from SSD.
        let runtime = RuntimeConfiguration(prefillChunkTokens: 128,
                                           forceLogitsHead: false)

        let loadStart = Date()
        // The `expecting:` overload pins the baseline this suite measures
        // against instead of letting auto-detect resolve it, so a baseline
        // drift fails validation here rather than silently re-detecting.
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: cfg,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: integrity)
        let firstLoadSeconds = Date().timeIntervalSince(loadStart)

        // Resolve the sidecar folder directly and load with the family hint,
        // matching the pinned baseline above rather than re-peeking.
        let tokenizerFolder = try #require(
            MFTokenizer.tokenizerFolder(forModelDirectory: modelURL),
            "install has no tokenizer/ sidecar with tokenizer.json")
        let tokenizer = try await MFTokenizer.load(from: tokenizerFolder,
                                                   family: .qwen38flashnext)

        let forwardRuntime = try ForwardRunnerFactory.make(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtime)

        FileHandle.standardError.write(Data(
            "[flashnext-firstlight] loading qwen38flashnext against the pinned baseline; the production gate was lifted 2026-09-10.\n".utf8))

        return Harness(context: context, model: model, tokenizer: tokenizer,
                       forwardRuntime: forwardRuntime,
                       runner: forwardRuntime.producer,
                       firstLoadSeconds: firstLoadSeconds,
                       verifyMode: verifyMode)
    }

    /// Peak resident set size of THIS (test) process, in MiB. `resident_size_max`
    /// is a high-water mark, so it survives across probes within the run.
    private static func peakRSSMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO),
                          $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.resident_size_max) / 1_048_576 : -1
    }

    /// CLI-identical footer string (matches `MferenceCLI/Run.swift`).
    private static func footer(_ s: RawDecodeResult) -> String {
        let tps = s.decodeSeconds > 0 ? Double(s.newTokens) / s.decodeSeconds : 0
        return "[stop=\(String(describing: s.reason)) "
            + "prefill=\(s.prefillTokens)tok/\(String(format: "%.2f", s.prefillSeconds))s "
            + "new=\(s.newTokens)tok "
            + "decode=\(String(format: "%.2f", s.decodeSeconds))s "
            + "tok/s=\(String(format: "%.3f", tps))]"
    }

    private static func shortBenchmarkPrompt(_ h: Harness) throws -> [Int32] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { root.deleteLastPathComponent() }
        let benchURL = root
            .appendingPathComponent("docs/benchmark-prompts/real-generation-v1")
            .appendingPathComponent("short-explanation.json")
        struct Turn: Decodable { let role: String; let content: String }
        let turns = try JSONDecoder().decode(
            [Turn].self, from: try Data(contentsOf: benchURL))
        let messages = turns.map {
            MFTokenizer.Message(role: $0.role == "system" ? .system : .user,
                                content: $0.content)
        }
        return h.tokenizer.encode(
            try h.tokenizer.applyChatTemplate(messages), addBOS: false)
    }

    private static func readLogits(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let values = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(values[$0]) }
    }

    private static func argmax(_ values: [Float]) -> Int {
        values.indices.max { values[$0] < values[$1] } ?? 0
    }

    /// Drive one generation through the production loop, capturing raw deltas
    /// verbatim (no structured-decoder filtering, so any think/markup tokens are
    /// visible), and print the text + CLI footer + peak RSS.
    @discardableResult
    private static func generate(_ h: Harness,
                                 label: String,
                                 promptIds: [Int32],
                                 maxNew: Int,
                                 temperature: Float) async throws
        -> (text: String, stats: RawDecodeResult)
    {
        let config = GenerationConfig(maxNewTokens: maxNew, temperature: temperature)
        let scratch = try RawCompletionScratch(
            context: h.context,
            vocab: h.model.config.vocabSize,
            logitSoftcap: Float(h.model.config.finalLogitSoftcap))
        var text = ""
        let stats = try await runRawCompletion(
            producer: h.runner,
            tokenizer: h.tokenizer,
            promptIds: promptIds,
            config: config,
            context: h.context,
            scratch: scratch,
            prefillConfig: h.forwardRuntime.prefillConfig) { progress in
                switch progress {
                case .prefill: break
                case .token(_, _, let delta): text += delta
                case .tail(let tail): text += tail
                }
            }
        var out = "\n===== FLASHNEXT FIRST LIGHT: \(label) =====\n"
        out += "prompt_tokens=\(promptIds.count) max_new=\(maxNew) "
        out += "temperature=\(temperature) verify=\(h.verifyMode)\n"
        out += "--- GENERATED TEXT (verbatim) ---\n\(text)\n--- END ---\n"
        out += footer(stats) + "\n"
        out += String(format: "peak_rss_mb=%.1f first_load_s=%.1f\n",
                      peakRSSMB(), h.firstLoadSeconds)
        FileHandle.standardError.write(Data(out.utf8))
        return (text, stats)
    }

    /// Build a >indexer-budget prompt: filler sentences with a passkey planted
    /// near the middle, then a retrieval question. Returns (promptText, passkey).
    private static func passkeyPrompt(targetTokens: Int,
                                      tokenizer: MFTokenizer) -> (String, String) {
        let passkey = "739215"
        let filler = "The garden was quiet in the long afternoon and the light "
            + "moved slowly across the grass while the birds settled in the hedges. "
        var body = ""
        // Grow until we exceed the target; the passkey is injected mid-body below.
        while tokenizer.encode(body, addBOS: false).count < targetTokens {
            body += filler
        }
        // Inject the passkey sentence near the middle by splitting the filler.
        let sentences = body.split(separator: ".", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        let mid = sentences.count / 2
        var rebuilt = ""
        for (i, s) in sentences.enumerated() {
            rebuilt += s + ". "
            if i == mid {
                rebuilt += "Important: the passkey is \(passkey). Remember it. "
            }
        }
        let prompt = rebuilt
            + "\n\nQuestion: What is the passkey mentioned above? "
            + "Answer with only the number."
        return (prompt, passkey)
    }

    /// The whole first light in ONE test: load the 163 GiB install once (first
    /// touch SHA-verifies unless the trusted-receipt mode is set), then run the
    /// three probes sequentially, resetting the runner between them
    /// (`runRawCompletion(.reset)` calls `producer.reset()`). Serialized so no
    /// second model process is ever spawned.
    ///
    /// Skips silently without `MFERENCE_FLASHNEXT_GTURBO`.
    @Test func firstLightMeasurement() async throws {
        guard Self.installPath() != nil else { return }
        // maxContext covers the long-context probe (~3-4k) plus its generation.
        guard let h = try await Self.loadHarness(maxContext: 8192) else { return }

        // Probe 1 — greedy short completion.
        let p1 = h.tokenizer.encode("The capital of France is", addBOS: true)
        let (t1, _) = try await Self.generate(
            h, label: "greedy-capital", promptIds: p1, maxNew: 24, temperature: 0)
        #expect(!t1.isEmpty, "probe 1 produced no text")

        // Probe 2 — greedy chat-template explanation (short-explanation benchmark).
        let p2 = try Self.shortBenchmarkPrompt(h)
        let (t2, _) = try await Self.generate(
            h, label: "greedy-short-explanation", promptIds: p2,
            maxNew: 128, temperature: 0)
        #expect(!t2.isEmpty, "probe 2 produced no text")

        // Probe 3 — long-context passkey retrieval (exercises the sparse indexer
        // beyond its 2048 budget). Wrapped through the chat template so it is a
        // real query, and given enough new tokens to finish any think preamble
        // and actually state the answer (this is a Qwen-style think-first model).
        let (promptText, passkey) = Self.passkeyPrompt(targetTokens: 3200,
                                                       tokenizer: h.tokenizer)
        let p3 = h.tokenizer.encode(
            try h.tokenizer.applyChatTemplate(
                [MFTokenizer.Message(role: .user, content: promptText)]),
            addBOS: false)
        let ctxNote = "[flashnext-firstlight] long-context prompt tokens=\(p3.count) "
            + "(indexer budget 2048; passkey=\(passkey))\n"
        FileHandle.standardError.write(Data(ctxNote.utf8))
        #expect(p3.count > 2048,
                "long-context probe must exceed the indexer budget; got \(p3.count)")
        let (t3, s3) = try await Self.generate(
            h, label: "long-context-passkey", promptIds: p3,
            maxNew: 320, temperature: 0)
        let retrieved = t3.contains(passkey)
        let tps3 = s3.decodeSeconds > 0 ? Double(s3.newTokens) / s3.decodeSeconds : 0
        FileHandle.standardError.write(Data(String(
            format: "[flashnext-firstlight] passkey retrieved=%@ decode_tok_s=%.3f at ctx=%d\n",
            retrieved ? "YES" : "NO", tps3, p3.count).utf8))
        // Not asserted hard: retrieval quality is a read, not a proven gate at
        // 180B scale. The captured YES/NO and tok/s are the measurement.
    }

    /// Current-build A/B against scalar replay on the real INT8-router install.
    /// This does not need an external reference: both sides share the exact
    /// weights and decode path, isolating the prefill implementation itself.
    @Test func chunkedPrefillMatchesSequentialOnRealInstall() async throws {
        guard Self.installPath() != nil else { return }
        guard let h = try await Self.loadHarness(maxContext: 256),
              let runner = h.runner as? FlashNextForwardRunner else { return }
        let prompt = try Self.shortBenchmarkPrompt(h)
        let vocab = h.model.config.vocabSize
        let logits = try #require(h.context.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared))

        for (position, token) in prompt.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        let sequentialPrompt = Self.readLogits(logits, count: vocab)
        let continuation = Int32(Self.argmax(sequentialPrompt))
        try await runner.produce(token: continuation, position: prompt.count,
                                 into: logits)
        let sequentialNext = Self.readLogits(logits, count: vocab)
        var sequentialRollout = [continuation]
        var sequentialCursor = sequentialNext
        for step in 1..<16 {
            let token = Int32(Self.argmax(sequentialCursor))
            sequentialRollout.append(token)
            try await runner.produce(token: token, position: prompt.count + step,
                                     into: logits)
            sequentialCursor = Self.readLogits(logits, count: vocab)
        }

        runner.reset()
        let result = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 128), into: logits,
            onProgress: { _ in })
        let chunkedPrompt = Self.readLogits(logits, count: vocab)
        let chunkedContinuation = Int32(Self.argmax(chunkedPrompt))
        try await runner.produce(token: chunkedContinuation, position: prompt.count,
                                 into: logits)
        let chunkedNext = Self.readLogits(logits, count: vocab)
        var chunkedRollout = [chunkedContinuation]
        var chunkedCursor = chunkedNext
        for step in 1..<16 {
            let token = Int32(Self.argmax(chunkedCursor))
            chunkedRollout.append(token)
            try await runner.produce(token: token, position: prompt.count + step,
                                     into: logits)
            chunkedCursor = Self.readLogits(logits, count: vocab)
        }

        let promptMaxAbs = zip(sequentialPrompt, chunkedPrompt)
            .map { abs($0 - $1) }.max() ?? 0
        let nextMaxAbs = zip(sequentialNext, chunkedNext)
            .map { abs($0 - $1) }.max() ?? 0
        let promptScale = sequentialPrompt.map(abs).max() ?? 1
        let nextScale = sequentialNext.map(abs).max() ?? 1
        let promptRelative = promptMaxAbs / max(promptScale, 1e-6)
        let nextRelative = nextMaxAbs / max(nextScale, 1e-6)
        let promptArgmax = Self.argmax(sequentialPrompt)
        let chunkedPromptArgmax = Self.argmax(chunkedPrompt)
        let nextArgmax = Self.argmax(sequentialNext)
        let chunkedNextArgmax = Self.argmax(chunkedNext)
        let rolloutStatus = sequentialRollout == chunkedRollout ? "exact" : "DIFF"
        let report = "[flashnext-prefill-ab] prompt=\(prompt.count) "
            + "prompt_max_abs=\(promptMaxAbs) relative=\(promptRelative) "
            + "prompt_argmax=\(promptArgmax)/\(chunkedPromptArgmax) "
            + "next_max_abs=\(nextMaxAbs) relative=\(nextRelative) "
            + "next_argmax=\(nextArgmax)/\(chunkedNextArgmax) "
            + "greedy16=\(rolloutStatus)\n"
        FileHandle.standardError.write(Data(report.utf8))

        #expect(result == PrefillResult(newPosition: prompt.count,
                                        seed: .logitsWritten))
        #expect(chunkedPromptArgmax == promptArgmax)
        #expect(chunkedNextArgmax == nextArgmax)
        #expect(sequentialRollout == chunkedRollout)
        #expect(promptRelative < 0.05)
        #expect(nextRelative < 0.05)
    }

    /// The gate is UP: the production door now resolves the REAL installed
    /// directory instead of refusing it by name. Env-gated on the install path
    /// and deliberately cheap — it reads `manifest.json` only, no weights.
    ///
    /// Both steps `Model.load(directoryURL:device:)` takes before it touches a
    /// byte of weights are checked here: `peekFamily` (the funnel every
    /// CLI/server/app entry point goes through) and the baseline lookup that
    /// follows it. The load itself is not run — with the default integrity
    /// policy it would SHA-256 the whole ~163 GiB install — and the
    /// measurements in this suite exercise the real load anyway.
    @Test func productionDoorLoadsRealInstall() throws {
        guard let path = Self.installPath() else { return }
        let modelURL = URL(fileURLWithPath: path)
        let family = try ManifestReader.peekFamily(directoryURL: modelURL)
        #expect(family == .qwen38flashnext)
        #expect(ArchConfig.knownArchitectures[family] != nil,
                "auto-detect resolved \(family.rawValue) with no baseline to load it")
        #expect(ManifestReader.familiesWithoutRunner[family.rawValue] == nil)
    }
}
