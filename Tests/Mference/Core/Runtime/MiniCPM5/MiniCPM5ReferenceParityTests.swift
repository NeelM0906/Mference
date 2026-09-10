import Foundation
import Metal
import Testing
@testable import Mference

/// The `minicpm5` reference-parity gates: the Metal runner, loaded from a real
/// install of the toy checkpoint, against goldens captured from `transformers`
/// 5.6.2 on the same INT4-reconstructed weights (`Fixtures/minicpm5/`).
///
/// # Tolerance tiers
///
/// The reference runs in float32. The runner keeps its residual stream,
/// projections and attention in FP16 with FP32 accumulation, so a bit-exact
/// match is not the claim. The gates are:
///
/// 1. **per-module** (attention-branch, MLP-branch and residual output of
///    every layer at every prompt position) and **full forward** (logits at
///    every position): `numpy.allclose` at `atol = rtol = 1e-2` — the
///    FP16-activation tier. First measured 2026-09-10: worst max-abs 3.4e-3
///    on `hidden_out`, 3.2e-3 on the logits (`reportsObservedParityMargins`),
///    so the gate sits ~3x above the observed worst; recorded on the family
///    page.
/// 2. **argmax at every prefill position**: exact.
/// 3. **greedy rollouts**: token-exact for 16 steps on both prompts, through
///    chunked prefill + the fused greedy head, and again through the exact
///    logits head. The goldens assert a top-1/top-2 margin ≥ 5e-3 at every
///    step, so a flip here would be a real defect, not arithmetic noise.
/// 4. **cached decode equals recompute**: the runner's rollout equals the
///    reference's *uncached* re-prefill rollout, and prefill-then-decode
///    equals sequential decode bit for bit (`MiniCPM5ForwardRunnerTests`).
@Suite(.serialized) struct MiniCPM5ReferenceParityTests {

    static let atol: Float = 1e-2
    static let rtol: Float = 1e-2

    private struct Harness {
        let dir: URL
        let ctx: MetalContext
        let runner: MiniCPM5ForwardRunner
        let logits: MTLBuffer
        let capture: MTLBuffer
        let config: ArchConfig

        func cleanup() { try? FileManager.default.removeItem(at: dir) }

        func captureRow(_ elementOffset: Int) -> [Float] {
            let base = capture.contents().bindMemory(to: Float16.self,
                                                     capacity: MiniCPM5ForwardRunner.captureElements(config: config))
            return (0..<config.hiddenSize).map { Float(base[elementOffset + $0]) }
        }

        func logitsRow() -> [Float] {
            let base = logits.contents().bindMemory(to: Float16.self, capacity: config.vocabSize)
            return (0..<config.vocabSize).map { Float(base[$0]) }
        }
    }

    private static func makeHarness(forceLogits: Bool = true,
                                    maxContext: Int = 128) throws -> Harness {
        let dir = try MiniCPM5Parity.installToyCheckpoint()
        let ctx = try MetalContext()
        let config = MiniCPM5Parity.archConfig()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: config)
        let runner = try MiniCPM5ForwardRunner(
            model: model, context: ctx, maxContext: maxContext,
            runtimeConfiguration: RuntimeConfiguration(prefillEnabled: true,
                                                       forceLogitsHead: forceLogits))
        guard let logits = ctx.device.makeBuffer(
                length: config.vocabSize * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let capture = ctx.device.makeBuffer(
                length: MiniCPM5ForwardRunner.captureElements(config: config)
                    * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        runner.parityCapture = capture
        return Harness(dir: dir, ctx: ctx, runner: runner, logits: logits,
                       capture: capture, config: config)
    }

    private static func argmax(_ row: [Float]) -> Int {
        var best = 0
        for i in 1..<row.count where row[i] > row[best] { best = i }
        return best
    }

    @Test func toyConfigMatchesTheGoldensManifest() throws {
        #expect(try MiniCPM5Parity.manifestConfigMatchesArch() == [])
    }

    // MARK: - Gates 1 and 2: per-module and full forward, every position

    @Test(arguments: MiniCPM5Goldens.Prompt.allCases)
    func gate1and2_perLayerTensorsAndLogitsMatchAtEveryPosition(
        prompt: MiniCPM5Goldens.Prompt) async throws {
        let h = try Self.makeHarness()
        defer { h.cleanup() }
        let goldens = try MiniCPM5Goldens(prompt: prompt, phase: .prefill)
        let tokens = try MiniCPM5Goldens.promptTokens(prompt)
        let expectedArgmax = try goldens.ints("argmax_all_positions")
        let cfg = h.config

        for (position, token) in tokens.enumerated() {
            try await h.runner.produceExactPrefill(token: token, position: position,
                                                   into: h.logits)
            // Embedding row: exact up to the FP16 store.
            let embed = FlashNextDelta.compare(
                h.captureRow(MiniCPM5ForwardRunner.captureEmbedSlot(config: cfg)),
                try goldens.row("embed_out", position), atol: Self.atol, rtol: Self.rtol)
            #expect(embed.passes, "\(prompt) pos \(position) embed_out \(embed.description)")
            for layer in 0..<cfg.numLayers {
                let key = String(format: "layer%02d", layer)
                for (kind, name) in ["attn_out", "mlp_out", "hidden_out"].enumerated() {
                    let actual = h.captureRow(
                        MiniCPM5ForwardRunner.captureSlot(layer: layer, kind: kind, config: cfg))
                    let expected = try goldens.row("\(key).\(name)", position)
                    let delta = FlashNextDelta.compare(actual, expected,
                                                       atol: Self.atol, rtol: Self.rtol)
                    #expect(delta.passes,
                            "\(prompt) pos \(position) \(key).\(name) \(delta.description)")
                }
            }
            let finalNorm = FlashNextDelta.compare(
                h.captureRow(MiniCPM5ForwardRunner.captureFinalNormSlot(config: cfg)),
                try goldens.row("final_norm_out", position), atol: Self.atol, rtol: Self.rtol)
            #expect(finalNorm.passes, "\(prompt) pos \(position) final_norm_out \(finalNorm.description)")
            let logits = h.logitsRow()
            let logitDelta = FlashNextDelta.compare(logits, try goldens.row("logits", position),
                                                    atol: Self.atol, rtol: Self.rtol)
            #expect(logitDelta.passes, "\(prompt) pos \(position) logits \(logitDelta.description)")
            #expect(Self.argmax(logits) == expectedArgmax[position],
                    "\(prompt) pos \(position): argmax \(Self.argmax(logits)) vs \(expectedArgmax[position])")
        }
    }

    // MARK: - Gate 3: greedy rollouts token-exact

    @Test(arguments: MiniCPM5Goldens.Prompt.allCases)
    func gate3_greedyRolloutIsTokenExact_fusedHead(prompt: MiniCPM5Goldens.Prompt) async throws {
        let h = try Self.makeHarness(forceLogits: false)
        defer { h.cleanup() }
        h.runner.parityCapture = nil
        let goldens = try MiniCPM5Goldens(prompt: prompt, phase: .decode)
        let expected = try goldens.ints("generated_token_ids")
        let tokens = try MiniCPM5Goldens.promptTokens(prompt)

        let result = try await h.runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32), into: h.logits, onProgress: { _ in })
        guard case .greedyToken(let seed) = result.seed else {
            Issue.record("expected a greedy seed from the fused head")
            return
        }
        var generated = [Int(seed)]
        var position = tokens.count
        while generated.count < expected.count {
            try await h.runner.produce(token: Int32(generated.last!), position: position,
                                       into: h.logits)
            generated.append(Int(h.runner.lastGreedyToken))
            position += 1
        }
        #expect(generated == expected, "\(prompt): \(generated) vs \(expected)")
    }

    @Test(arguments: MiniCPM5Goldens.Prompt.allCases)
    func gate3_greedyRolloutIsTokenExact_logitsHead(prompt: MiniCPM5Goldens.Prompt) async throws {
        let h = try Self.makeHarness()
        defer { h.cleanup() }
        h.runner.parityCapture = nil
        let goldens = try MiniCPM5Goldens(prompt: prompt, phase: .decode)
        let expected = try goldens.ints("generated_token_ids")
        let tokens = try MiniCPM5Goldens.promptTokens(prompt)

        _ = try await h.runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: h.logits, onProgress: { _ in })
        var generated = [Self.argmax(h.logitsRow())]
        var position = tokens.count
        while generated.count < expected.count {
            try await h.runner.produceExactPrefill(token: Int32(generated.last!),
                                                   position: position, into: h.logits)
            // Step logits within the same tier as the prefill logits.
            let delta = FlashNextDelta.compare(h.logitsRow(),
                                               try goldens.row("step_logits", generated.count),
                                               atol: Self.atol, rtol: Self.rtol)
            #expect(delta.passes, "\(prompt) step \(generated.count) logits \(delta.description)")
            generated.append(Self.argmax(h.logitsRow()))
            position += 1
        }
        #expect(generated == expected, "\(prompt): \(generated) vs \(expected)")
    }

    // MARK: - Gate 4: cached decode equals recompute

    @Test(arguments: MiniCPM5Goldens.Prompt.allCases)
    func gate4_cachedDecodeEqualsUncachedRecompute(prompt: MiniCPM5Goldens.Prompt) async throws {
        let h = try Self.makeHarness(forceLogits: false)
        defer { h.cleanup() }
        h.runner.parityCapture = nil
        let goldens = try MiniCPM5Goldens(prompt: prompt, phase: .decode)
        let uncachedReference = try goldens.ints("uncached_rollout_token_ids")
        let cachedReference = try goldens.ints("generated_token_ids")
        #expect(uncachedReference == cachedReference)   // the reference's own proof
        let tokens = try MiniCPM5Goldens.promptTokens(prompt)

        // Uncached: re-prefill the whole sequence for every step.
        var sequence = tokens
        var uncached: [Int] = []
        for _ in 0..<uncachedReference.count {
            h.runner.reset()
            let result = try await h.runner.prefillChunked(
                tokens: sequence[...], startPosition: 0, outputMode: .greedyIfAvailable,
                config: .production(chunkTokens: 32), into: h.logits, onProgress: { _ in })
            guard case .greedyToken(let next) = result.seed else {
                Issue.record("expected a greedy seed"); return
            }
            uncached.append(Int(next))
            sequence.append(Int32(next))
        }
        #expect(uncached == uncachedReference, "\(prompt) uncached: \(uncached)")

        // Cached: one prefill, then decode steps.
        h.runner.reset()
        let result = try await h.runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32), into: h.logits, onProgress: { _ in })
        guard case .greedyToken(let seed) = result.seed else {
            Issue.record("expected a greedy seed"); return
        }
        var cached = [Int(seed)]
        var position = tokens.count
        while cached.count < cachedReference.count {
            try await h.runner.produce(token: Int32(cached.last!), position: position,
                                       into: h.logits)
            cached.append(Int(h.runner.lastGreedyToken))
            position += 1
        }
        #expect(cached == uncached, "\(prompt): cached \(cached) vs uncached \(uncached)")
    }

    // MARK: - Margins

    /// Reports the observed worst deltas so the tier above is visible rather
    /// than inferred; recorded on docs/families/MINICPM5.md.
    @Test func reportsObservedParityMargins() async throws {
        var report: [String] = []
        for prompt in MiniCPM5Goldens.Prompt.allCases {
            let h = try Self.makeHarness()
            defer { h.cleanup() }
            let goldens = try MiniCPM5Goldens(prompt: prompt, phase: .prefill)
            let tokens = try MiniCPM5Goldens.promptTokens(prompt)
            let cfg = h.config
            var worst: [String: (abs: Float, rel: Float)] = [:]
            func note(_ key: String, _ actual: [Float], _ expected: [Float]) {
                let delta = FlashNextDelta.compare(actual, expected, atol: 0, rtol: 0)
                let previous = worst[key] ?? (0, 0)
                worst[key] = (max(previous.abs, delta.maxAbs), max(previous.rel, delta.maxRel))
            }
            for (position, token) in tokens.enumerated() {
                try await h.runner.produceExactPrefill(token: token, position: position,
                                                       into: h.logits)
                for layer in 0..<cfg.numLayers {
                    for (kind, name) in ["attn_out", "mlp_out", "hidden_out"].enumerated() {
                        let key = String(format: "layer%02d.%@", layer, name)
                        note(name, h.captureRow(MiniCPM5ForwardRunner.captureSlot(
                            layer: layer, kind: kind, config: cfg)), try goldens.row(key, position))
                    }
                }
                note("logits", h.logitsRow(), try goldens.row("logits", position))
            }
            for key in ["attn_out", "mlp_out", "hidden_out", "logits"] {
                let w = worst[key]!
                report.append(String(format: "%@ %@: maxAbs %.3e", prompt.rawValue, key, w.abs))
            }
        }
        print("[minicpm5 parity margins] " + report.joined(separator: " | "))
        #expect(!report.isEmpty)
    }
}
