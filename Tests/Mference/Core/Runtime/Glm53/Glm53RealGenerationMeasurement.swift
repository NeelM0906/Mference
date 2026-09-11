import Testing
import Foundation
import Metal
@testable import Mference

/// **Measurement** harness for the `glm53Flash` production runner
/// (`Glm53ForwardRunner`) against the REAL install: greedy raw continuation,
/// the frozen short-explanation chat case, a dense-vs-indexer A/B at short
/// context, and a passkey probe past the pooled indexer's `index_topk`
/// (2,048 tokens) so the sparse selection is actually applied, each reporting
/// decode tok/s and peak RSS.
///
/// It is the family's first-light vehicle: the capability gate
/// (`ManifestReader.familiesWithoutRunner`) refuses the family at every
/// ordinary door until first light is green, so this suite loads through
/// `Model.load(directoryURL:device:expecting:)` — the explicit-baseline door
/// the toy parity fixture also uses — and the sidecar tokenizer with the
/// family hint. Everything downstream (`ForwardRunnerFactory.make`,
/// `runRawCompletion`, sampling, the timing footer) is the production
/// generation machinery, unmodified.
///
/// Env-gated, skipped without the gate:
///   * `MFERENCE_GLM53_GTURBO` — path to the verified install dir (~181 GB).
///   * `MFERENCE_GLM53_VERIFY=trusted-receipt` — skip the first-touch SHA-256
///     in favour of the install receipt's size checks. Default `full-sha256`.
///   * `MFERENCE_GLM53_PROBES` — comma list of `raw`, `chat`, `dense-ab`,
///     `needle`; default `raw,chat,dense-ab`.
///   * `MFERENCE_GLM53_MAX_CONTEXT` — runner context; default 2048, or the
///     needle length plus 512 when `needle` is requested.
///   * `MFERENCE_GLM53_NEEDLE_TOKENS` — needle prompt target; default 2,600
///     (must exceed `index_topk` 2,048 so the pooled selection is applied).
///   * `MFERENCE_GLM53_SLOTS` — expert cache slots; default 16.
///   * `MFERENCE_GLM53_RESIDENT=1` — map every expert file once (the mode
///     `auto` picks on a 256 GB host) instead of the slot cache; the runner
///     then routes on the GPU with no per-layer round trip.
///   * `MFERENCE_GLM53_CHAT_MAX_NEW` — chat probe token budget; default 160.
///
/// Quality caveat carried in every report: greedy token-exactness against a
/// reference rollout cannot be checked at 320B scale on this host inside the
/// runtime's memory budget, so coherent output is a *read* of the kernels at
/// scale, not a proof. The dense A/B is the one exact check that survives:
/// below `index_topk` cached tokens the model's own selection is exhaustive,
/// so the indexer arm and the dense arm must generate the same tokens.
@Suite(.serialized) struct Glm53RealGenerationMeasurement {

    private struct Harness {
        let context: MetalContext
        let model: Model
        let tokenizer: MFTokenizer
        let forwardRuntime: ForwardRuntime
        let runner: Glm53ForwardRunner
        let firstLoadSeconds: Double
        let verifyMode: String
        let maxContext: Int
    }

    private static var env: [String: String] { ProcessInfo.processInfo.environment }
    private static func installPath() -> String? { env["MFERENCE_GLM53_GTURBO"] }

    private static func probes() -> Set<String> {
        let raw = env["MFERENCE_GLM53_PROBES"] ?? "raw,chat,dense-ab"
        return Set(raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
    }

    private static func intEnv(_ name: String, default value: Int) -> Int {
        env[name].flatMap { Int($0) } ?? value
    }

    private static func needleTokens() -> Int { intEnv("MFERENCE_GLM53_NEEDLE_TOKENS", default: 2_600) }

    private static func maxContext() -> Int {
        let fallback = probes().contains("needle") ? needleTokens() + 512 : 2048
        return intEnv("MFERENCE_GLM53_MAX_CONTEXT", default: fallback)
    }

    private static func log(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }

    /// Load the real install against the pinned production baseline and build
    /// `Glm53ForwardRunner` through the real factory. Nil without the gate.
    private static func loadHarness(maxContext: Int) async throws -> Harness? {
        guard let path = installPath() else { return nil }
        let modelURL = URL(fileURLWithPath: path)
        let context = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.glm53Flash],
                               "the baseline the runner is built against is missing")

        let verifyMode = env["MFERENCE_GLM53_VERIFY"] ?? "full-sha256"
        let integrity: ModelIntegrityPolicy = verifyMode == "trusted-receipt"
            ? .sizeCheckTrustedReceipt : .fullSha256
        let resident = env["MFERENCE_GLM53_RESIDENT"] == "1"
        let slots = resident
            ? RuntimeConfiguration.allowedExpertCacheSlots.max()!
            : intEnv("MFERENCE_GLM53_SLOTS", default: 16)
        let runtime = RuntimeConfiguration(expertCacheSlots: slots,
                                           prefillChunkTokens: 128,
                                           forceLogitsHead: false)

        let loadStart = Date()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: cfg,
            streamingMode: resident ? .resident : .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: integrity)
        let firstLoadSeconds = Date().timeIntervalSince(loadStart)

        let tokenizerFolder = try #require(
            MFTokenizer.tokenizerFolder(forModelDirectory: modelURL),
            "install has no tokenizer/ sidecar with tokenizer.json")
        let tokenizer = try await MFTokenizer.load(from: tokenizerFolder, family: .glm53Flash)
        #expect(tokenizer.dialect == .glm5)

        let forwardRuntime = try ForwardRunnerFactory.make(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtime)
        let runner = try #require(forwardRuntime.producer as? Glm53ForwardRunner,
                                  "the factory did not dispatch Glm53ForwardRunner")

        #expect(runner.expertsResident == resident)
        log("[glm53-firstlight] loaded glm53Flash against the pinned baseline "
            + "(verify=\(verifyMode), experts=\(resident ? "resident" : "\(slots) slots"), maxContext=\(maxContext), "
            + String(format: "first_load_s=%.1f", firstLoadSeconds) + ")")
        return Harness(context: context, model: model, tokenizer: tokenizer,
                       forwardRuntime: forwardRuntime, runner: runner,
                       firstLoadSeconds: firstLoadSeconds, verifyMode: verifyMode,
                       maxContext: maxContext)
    }

    /// Peak resident set size of THIS process, in MiB (a high-water mark).
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

    private struct Generation {
        let text: String
        let tokens: [Int32]
        let stats: RawDecodeResult
    }

    /// One generation through the production loop, raw deltas verbatim (no
    /// structured-decoder filtering, so the think block stays visible).
    @discardableResult
    private static func generate(_ h: Harness, label: String, promptIds: [Int32],
                                 maxNew: Int, temperature: Float) async throws -> Generation {
        let config = GenerationConfig(maxNewTokens: maxNew, temperature: temperature)
        let scratch = try RawCompletionScratch(
            context: h.context,
            vocab: h.model.config.vocabSize,
            logitSoftcap: Float(h.model.config.finalLogitSoftcap))
        var text = ""
        var tokens: [Int32] = []
        h.runner.beginDecodePhaseWindow()
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
                case .token(_, let id, let delta): tokens.append(id); text += delta
                case .tail(let tail): text += tail
                }
            }
        var out = "\n===== GLM53 FIRST LIGHT: \(label) =====\n"
        out += "prompt_tokens=\(promptIds.count) max_new=\(maxNew) "
        out += "temperature=\(temperature) verify=\(h.verifyMode) dense_ab=\(h.runner.denseSelectionForAB)\n"
        out += "--- GENERATED TEXT (verbatim) ---\n\(text)\n--- END ---\n"
        out += "token_ids=\(tokens)\n"
        out += footer(stats) + "\n"
        let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
        out += "phases: expert io \(ms(h.runner.totalIoNanos)) ms, indexer top-k \(ms(h.runner.totalIndexerTopKNanos)) ms, "
        out += "router \(ms(h.runner.totalRouterNanos)) ms, gpu busy \(ms(h.runner.totalGpuBusyNanos)) ms, "
        out += "gpu span \(ms(h.runner.totalGpuSpanNanos)) ms, command buffers \(h.runner.totalCommandBuffers)\n"
        out += String(format: "peak_rss_mb=%.1f first_load_s=%.1f\n",
                      peakRSSMB(), h.firstLoadSeconds)
        FileHandle.standardError.write(Data(out.utf8))
        return Generation(text: text, tokens: tokens, stats: stats)
    }

    /// The frozen short-explanation case, rendered through the glm5 template.
    private static func shortExplanationPrompt(_ tokenizer: MFTokenizer) throws -> [Int32] {
        // Tests/Mference/Core/Runtime/Glm53/<file>: five levels up is the repo root.
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let benchURL = root
            .appendingPathComponent("docs/benchmark-prompts/real-generation-v1")
            .appendingPathComponent("short-explanation.json")
        struct Turn: Decodable { let role: String; let content: String }
        let turns = try JSONDecoder().decode([Turn].self, from: try Data(contentsOf: benchURL))
        let messages = turns.map {
            MFTokenizer.Message(role: $0.role == "system" ? .system : .user, content: $0.content)
        }
        return tokenizer.encode(try tokenizer.applyChatTemplate(messages), addBOS: false)
    }

    /// A >index_topk prompt: filler with a passkey planted mid-body, then the
    /// retrieval question. Returns (promptText, passkey).
    private static func passkeyPrompt(targetTokens: Int,
                                      tokenizer: MFTokenizer) -> (String, String) {
        let passkey = "739215"
        let filler = "The garden was quiet in the long afternoon and the light "
            + "moved slowly across the grass while the birds settled in the hedges. "
        var body = ""
        let fillerTokens = max(1, tokenizer.encode(filler, addBOS: false).count)
        let repeats = targetTokens / fillerTokens + 1
        for _ in 0..<repeats { body += filler }
        let sentences = body.split(separator: ".", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        let mid = sentences.count / 2
        var rebuilt = ""
        for (i, s) in sentences.enumerated() {
            rebuilt += s + ". "
            if i == mid { rebuilt += "Important: the passkey is \(passkey). Remember it. " }
        }
        let prompt = rebuilt
            + "\n\nQuestion: What is the passkey mentioned above? Answer with only the number."
        return (prompt, passkey)
    }

    /// The whole first light in ONE test: load the install once, then run the
    /// requested probes sequentially (`runRawCompletion(.reset)` resets the
    /// runner between them). Serialized so no second model process is spawned.
    @Test func firstLightMeasurement() async throws {
        guard Self.installPath() != nil else { return }
        let probes = Self.probes()
        let maxContext = Self.maxContext()
        guard let h = try await Self.loadHarness(maxContext: maxContext) else { return }
        let indexTopK = h.model.config.compressedAttention.indexTopK

        if probes.contains("raw") {
            let p1 = h.tokenizer.encode("The capital of France is", addBOS: true)
            let g = try await Self.generate(h, label: "greedy-capital", promptIds: p1,
                                            maxNew: 24, temperature: 0)
            #expect(!g.text.isEmpty, "raw probe produced no text")
        }

        var chatPrompt: [Int32] = []
        if probes.contains("chat") || probes.contains("dense-ab") {
            chatPrompt = try Self.shortExplanationPrompt(h.tokenizer)
        }

        if probes.contains("chat") {
            let g = try await Self.generate(h, label: "greedy-short-explanation",
                                            promptIds: chatPrompt,
                                            maxNew: Self.intEnv("MFERENCE_GLM53_CHAT_MAX_NEW", default: 160),
                                            temperature: 0)
            #expect(!g.text.isEmpty, "chat probe produced no text")
        }

        if probes.contains("dense-ab") {
            // Below index_topk cached tokens the selection is exhaustive, so
            // the two arms are the same computation and must produce the same
            // tokens. Both arms stay under the budget: prompt plus 32 tokens.
            let maxNew = 32
            #expect(chatPrompt.count + maxNew <= indexTopK,
                    "dense A/B must stay at or below index_topk \(indexTopK); prompt is \(chatPrompt.count)")
            h.runner.denseSelectionForAB = false
            let sparse = try await Self.generate(h, label: "dense-ab/indexer", promptIds: chatPrompt,
                                                 maxNew: maxNew, temperature: 0)
            h.runner.denseSelectionForAB = true
            let dense = try await Self.generate(h, label: "dense-ab/dense", promptIds: chatPrompt,
                                                maxNew: maxNew, temperature: 0)
            h.runner.denseSelectionForAB = false
            let firstDiff = zip(sparse.tokens, dense.tokens).enumerated()
                .first { $0.element.0 != $0.element.1 }?.offset
            Self.log("[glm53-firstlight] dense A/B: \(sparse.tokens.count)/\(dense.tokens.count) tokens, "
                     + "identical=\(sparse.tokens == dense.tokens)"
                     + (firstDiff.map { ", first divergence at token \($0)" } ?? ""))
            #expect(sparse.tokens == dense.tokens,
                    "indexer arm and dense arm diverged at \(firstDiff.map(String.init) ?? "length")")
        }

        if probes.contains("needle") {
            let target = Self.needleTokens()
            let (promptText, passkey) = Self.passkeyPrompt(targetTokens: target, tokenizer: h.tokenizer)
            let p3 = h.tokenizer.encode(
                try h.tokenizer.applyChatTemplate(
                    [MFTokenizer.Message(role: .user, content: promptText)]),
                addBOS: false)
            Self.log("[glm53-firstlight] needle prompt tokens=\(p3.count) "
                     + "(index_topk \(indexTopK); passkey=\(passkey))")
            #expect(p3.count > indexTopK,
                    "needle probe must exceed index_topk; got \(p3.count)")
            #expect(p3.count + 64 <= h.maxContext,
                    "needle prompt \(p3.count) does not fit maxContext \(h.maxContext)")
            let g = try await Self.generate(h, label: "long-context-passkey", promptIds: p3,
                                            maxNew: 48, temperature: 0)
            let retrieved = g.text.contains(passkey)
            let tps = g.stats.decodeSeconds > 0 ? Double(g.stats.newTokens) / g.stats.decodeSeconds : 0
            Self.log(String(format: "[glm53-firstlight] passkey retrieved=%@ decode_tok_s=%.3f at ctx=%d",
                            retrieved ? "YES" : "NO", tps, p3.count))
            // Not asserted hard: retrieval quality is a read, not a proven gate.
        }
    }

    /// The production door and the gate agree about this install: while the
    /// family is listed in `familiesWithoutRunner`, `peekFamily` refuses it by
    /// axis name; once the entry is gone it resolves the family. Env-gated on
    /// the install path and cheap — it reads `manifest.json` only.
    @Test func productionDoorAgreesWithTheGate() throws {
        guard let path = Self.installPath() else { return }
        let modelURL = URL(fileURLWithPath: path)
        let gated = ManifestReader.familiesWithoutRunner[ModelFamily.glm53Flash.rawValue]
        if let axes = gated {
            var thrown: Error?
            #expect(throws: (any Error).self) {
                do { _ = try ManifestReader.peekFamily(directoryURL: modelURL) }
                catch { thrown = error; throw error }
            }
            #expect(thrown as? ModelError == .familyRunnerNotImplemented(
                family: ModelFamily.glm53Flash.rawValue, missingAxes: axes))
        } else {
            let family = try ManifestReader.peekFamily(directoryURL: modelURL)
            #expect(family == .glm53Flash)
            #expect(ArchConfig.knownArchitectures[family] != nil)
        }
    }
}
