import Testing
import Foundation
import Metal
@testable import Mference

/// First-light **measurement** harness for the `qwen38flashnext` production
/// runner (`FlashNextForwardRunner`) against the REAL install. This is a
/// measurement-only entry point, not a gate lift: it never touches
/// `ManifestReader.familiesWithoutRunner`, and the production capability gate
/// (`peekFamily`) still refuses this family for the CLI, server and app — see
/// `productionDoorStillRefusesRealInstall` below and
/// `FlashNextCapabilityGateTests`.
///
/// How the gate is bypassed *only here* (option (b): a test-target, env-gated
/// measurement, exactly like the `MFERENCE_INKLING_GTURBO` real-model suites):
///   * `Model.load(directoryURL:device:expecting:)` — the overload that takes an
///     explicit `ArchConfig` baseline and does NOT funnel through
///     `ManifestReader.peekFamily`. This is the same internal door the FlashNext
///     reference/parity tie-back tests already use.
///   * `MFTokenizer.load(from:family:)` on the sidecar `tokenizer/` folder —
///     the non-gating tokenizer path (`load(forModelDirectory:)` would call
///     `peekFamily` and be refused).
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
/// proof. W2.1b (KLD vs a known-good conversion) remains the missing
/// quantitative quality gate.
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

    /// Load the real install through the internal (ungated) door and build the
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
        // BYPASS DOOR: the `expecting:` overload skips `peekFamily`, so the gate
        // that refuses this family everywhere else is not consulted here.
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: cfg,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: integrity)
        let firstLoadSeconds = Date().timeIntervalSince(loadStart)

        // NON-GATING tokenizer path: resolve the sidecar folder directly and
        // load with the family hint. `load(forModelDirectory:)` would peek.
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
            "[flashnext-firstlight] WARNING: loading gated family qwen38flashnext through the internal ungated door for MEASUREMENT ONLY; the production gate is unchanged.\n".utf8))

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
        // This file is at Tests/Mference/Core/Runtime/FlashNext/<file>; six
        // levels up is the repo root that holds docs/.
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
        let rendered = try h.tokenizer.applyChatTemplate(messages)
        let p2 = h.tokenizer.encode(rendered, addBOS: false)
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

    /// The gate stays DOWN: the production door (`peekFamily`, reached by
    /// `Model.load(directoryURL:device:)` and every CLI/server/app entry) still
    /// refuses the REAL installed directory by name. Env-gated on the install
    /// path but cheap — it only reads `manifest.json`, no weights.
    @Test func productionDoorStillRefusesRealInstall() throws {
        guard let path = Self.installPath() else { return }
        let modelURL = URL(fileURLWithPath: path)
        // peekFamily — the funnel.
        #expect(throws: ModelError.familyRunnerNotImplemented(
            family: "qwen38flashnext",
            missingAxes: ["hyperConnectionsLowRank",
                          "attentionIndexer",
                          "pleNgramEmbedding"])) {
            _ = try ManifestReader.peekFamily(directoryURL: modelURL)
        }
        // The auto-detect Model.load overload must refuse it too, before any
        // baseline resolution — the production load path the CLI uses.
        let ctx = try MetalContext()
        #expect(throws: (any Error).self) {
            _ = try Model.load(directoryURL: modelURL, device: ctx.device)
        }
    }
}
