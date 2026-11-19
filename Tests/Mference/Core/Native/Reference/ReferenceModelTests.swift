import Foundation
import Metal
import Testing
@testable import Mference

/// The whole reference path on the tiny synthetic model: GGUF → pack →
/// store → runner → scheduler, against a pure-CPU forward pass, plus the
/// inside-Mference byte-identity properties (layout, chunking, verify, taps,
/// rollback, prefix reuse).
@Suite("ReferenceModelTests", .serialized)
struct ReferenceModelTests {
    /// Only Sendable state, so models built from it can be handed off with
    /// `sending`; each model opens its own store over the pack.
    struct Fixture {
        let ctx: MetalContext
        let dir: URL
        var pack: URL { dir.appendingPathComponent("tiny.mfpack") }

        func store(verifyHashes: Bool = false) throws -> Native.ReferenceWeightStore {
            try Native.ReferenceWeightStore(directory: pack, device: ctx.device, verifyHashes: verifyHashes)
        }

        func model(context: Int = 256, chunk: Int = 16, verify: Int = 4, precision: Native.KVPrecision = .f16,
                   prefixBudget: UInt64 = 1 << 24) throws -> Native.ReferenceForwardModel {
            let store = try store()
            let factory = Native.ReferenceForwardModelFactory(context: ctx)
            let c = store.config
            let plan = try Native.ReferenceMemoryPlanner().resolve(
                config: c, weightsMapped: store.mappedByteCount, contextLength: context, precision: precision,
                maxVerifyRows: verify, maxPrefillChunk: chunk,
                activationBytes: factory.activationBytes(config: c, maxPrefillChunk: chunk, maxVerifyRows: verify),
                prefixCacheRAMBudget: prefixBudget, budget: 1 << 32)
            let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
            let components = Native.ModelComponents(
                device: ctx.device, weights: store, gemm: Native.ReferenceQuantGEMM(context: ctx),
                attention: Native.ReferenceAttentionOp(context: ctx), gdn: Native.ReferenceGDNOp(context: ctx),
                glue: Native.ReferenceGlueOps(context: ctx), kv: kv)
            return try factory.make(components, plan: plan) as! Native.ReferenceForwardModel
        }
    }

    static func fixture(layout: @escaping (Native.GGMLType, Int, Int) -> Native.TileLayout = {
        Native.PackLayoutTable.layout(for: $0, rows: $1, blocksPerRow: $2)
    }) throws -> Fixture {
        let ctx = try MetalContext()
        let dir = try RefSynthetic.temporaryDirectory()
        let ggufURL = dir.appendingPathComponent("tiny.gguf")
        try RefSynthetic.gguf().write(to: ggufURL)
        try Native.ReferencePackWriter.write(gguf: ggufURL, to: dir.appendingPathComponent("tiny.mfpack"),
                                             label: "tiny", layout: layout)
        let fx = Fixture(ctx: ctx, dir: dir)
        _ = try fx.store(verifyHashes: true)
        return fx
    }

    static func tiled(_ type: Native.GGMLType, _ rows: Int, _ blocks: Int) -> Native.TileLayout {
        RefSynthetic.tiled(type, rows, blocks)
    }

    /// Run one step on the GPU and return a copy of its logits.
    static func step(_ m: Native.ReferenceForwardModel, _ ctx: MetalContext, kind: Native.StepKind,
                     tokens: [Int32], logits: Native.LogitRows) throws -> [Float] {
        let start = m.kv.length
        try m.kv.beginStep(rows: tokens.count, kind: kind)
        var out: Native.TensorRef?
        try RefTest.run(ctx) { enc in
            out = try m.encodeStep(Native.StepPlan(kind: kind, startPosition: start, rows: tokens.count,
                                                   input: .host(tokens), logits: logits), encoder: enc)
        }
        return out.map(RefTest.floats) ?? []
    }

    static func reset(_ m: Native.ReferenceForwardModel, _ ctx: MetalContext) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        try m.kv.reset(commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// The CPU twin of the whole forward pass; returns the last row's logits.
    static func cpuForward(_ store: Native.ReferenceWeightStore, tokens: [Int32]) throws -> [Float] {
        let c = store.config
        func t(_ n: String) throws -> Native.QuantTensor { try store.tensor(n) }
        func vals(_ n: String) throws -> [Float] { Native.ReferenceDequant.values(of: try t(n)) }
        let rows = tokens.count, hidden = c.hiddenSize
        var x = Native.ReferenceCPU.getRows(weight: try t("token_embd.weight"), tokens: tokens)
        func addInto(_ y: [Float]) { for i in x.indices { x[i] = x[i] + y[i] } }
        for i in 0..<c.blockCount {
            let p = "blk.\(i)."
            var h = Native.ReferenceCPU.rmsNorm(x, rows: rows, cols: hidden, weight: try vals(p + "attn_norm.weight"),
                                                eps: c.rmsNormEpsilon)
            if c.layerKind(i) == .fullAttention {
                let attn = Native.ReferenceCPU.Attention(
                    headCount: c.headCount, kvHeadCount: c.kvHeadCount, headDim: c.keyLength,
                    ropeDims: c.ropeDimensionCount, eps: c.rmsNormEpsilon, freqBase: c.ropeFreqBase, precision: .f16)
                let qg = Native.ReferenceCPU.gemm(weight: try t(p + "attn_q.weight"), input: h, rows: rows)
                let k = Native.ReferenceCPU.gemm(weight: try t(p + "attn_k.weight"), input: h, rows: rows)
                let v = Native.ReferenceCPU.gemm(weight: try t(p + "attn_v.weight"), input: h, rows: rows)
                let a = attn.step(qGate: qg, k: k, v: v, rows: rows, qNorm: try vals(p + "attn_q_norm.weight"),
                                  kNorm: try vals(p + "attn_k_norm.weight"))
                addInto(Native.ReferenceCPU.gemm(weight: try t(p + "attn_output.weight"), input: a, rows: rows))
            } else {
                let shape = Native.ReferenceCPU.GDNShape(convKernel: c.ssmConvKernel, convChannels: c.ssmConvChannels,
                                                         valueHeads: c.ssmTimeStepRank, groups: c.ssmGroupCount,
                                                         dim: c.ssmStateSize, eps: c.rmsNormEpsilon)
                let zero = Native.ReferenceCPU.GDNState(
                    conv: [Float](repeating: 0, count: (c.ssmConvKernel - 1) * c.ssmConvChannels),
                    ssm: [Float](repeating: 0, count: c.ssmTimeStepRank * c.ssmStateSize * c.ssmStateSize))
                let r = Native.ReferenceCPU.gdn(
                    shape: shape,
                    qkv: Native.ReferenceCPU.gemm(weight: try t(p + "attn_qkv.weight"), input: h, rows: rows),
                    z: Native.ReferenceCPU.gemm(weight: try t(p + "attn_gate.weight"), input: h, rows: rows),
                    alpha: Native.ReferenceCPU.gemm(weight: try t(p + "ssm_alpha.weight"), input: h, rows: rows),
                    beta: Native.ReferenceCPU.gemm(weight: try t(p + "ssm_beta.weight"), input: h, rows: rows),
                    rows: rows, conv1d: try vals(p + "ssm_conv1d.weight"), dtBias: try vals(p + "ssm_dt.bias"),
                    a: try vals(p + "ssm_a"), norm: try vals(p + "ssm_norm.weight"), state: zero)
                addInto(Native.ReferenceCPU.gemm(weight: try t(p + "ssm_out.weight"), input: r.output, rows: rows))
            }
            h = Native.ReferenceCPU.rmsNorm(x, rows: rows, cols: hidden,
                                            weight: try vals(p + "post_attention_norm.weight"), eps: c.rmsNormEpsilon)
            let g = Native.ReferenceCPU.gemm(weight: try t(p + "ffn_gate.weight"), input: h, rows: rows)
            let u = Native.ReferenceCPU.gemm(weight: try t(p + "ffn_up.weight"), input: h, rows: rows)
            addInto(Native.ReferenceCPU.gemm(weight: try t(p + "ffn_down.weight"),
                                             input: Native.ReferenceCPU.swiglu(gate: g, up: u), rows: rows))
        }
        let last = Array(x[(rows - 1) * hidden ..< rows * hidden])
        let n = Native.ReferenceCPU.rmsNorm(last, rows: 1, cols: hidden, weight: try vals("output_norm.weight"),
                                            eps: c.rmsNormEpsilon)
        return Native.ReferenceCPU.gemm(weight: try t("token_embd.weight"), input: n, rows: 1)
    }

    static func argmax(_ v: [Float]) -> Int { v.indices.max { v[$0] < v[$1] || (v[$0] == v[$1] && $0 > $1) }! }

    @Test func forwardMatchesCPUTwinAndIsLayoutIndependent() throws {
        let plain = try Self.fixture()
        let tiled = try Self.fixture(layout: Self.tiled)
        defer {
            try? FileManager.default.removeItem(at: plain.dir)
            try? FileManager.default.removeItem(at: tiled.dir)
        }
        let tokens: [Int32] = [262, 104, 101, 108, 108, 111, 263, 5]
        let m1 = try plain.model()
        let gpu = try Self.step(m1, plain.ctx, kind: .prefill, tokens: tokens, logits: .last)
        let cpu = try Self.cpuForward(try plain.store(), tokens: tokens)
        let err = RefTest.maxRelativeError(gpu, cpu)
        #expect(err < 1e-3, "GPU vs CPU twin relative error \(err)")
        #expect(Self.argmax(gpu) == Self.argmax(cpu))
        let m2 = try tiled.model()
        let gpuTiled = try Self.step(m2, tiled.ctx, kind: .prefill, tokens: tokens, logits: .last)
        #expect(gpuTiled.map(\.bitPattern) == gpu.map(\.bitPattern))
    }

    /// Chunked prefill, decode and verify give byte-identical logits.
    @Test func chunkingDecodeAndVerifyAreByteIdentical() throws {
        let fx = try Self.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let m = try fx.model()
        let tokens: [Int32] = [262, 104, 101, 108, 108, 111, 263, 262, 20, 33]
        let whole = try Self.step(m, fx.ctx, kind: .prefill, tokens: tokens, logits: .last)
        try Self.reset(m, fx.ctx)
        _ = try Self.step(m, fx.ctx, kind: .prefill, tokens: Array(tokens[0..<3]), logits: .none)
        _ = try Self.step(m, fx.ctx, kind: .prefill, tokens: Array(tokens[3..<6]), logits: .none)
        var perRow: [[Float]] = []
        for t in tokens[6...] { perRow.append(try Self.step(m, fx.ctx, kind: .decode, tokens: [t], logits: .last)) }
        #expect(perRow.last!.map(\.bitPattern) == whole.map(\.bitPattern))
        // A verify step over the last 4 rows reproduces each decode row.
        try Self.reset(m, fx.ctx)
        _ = try Self.step(m, fx.ctx, kind: .prefill, tokens: Array(tokens[0..<6]), logits: .none)
        let all = try Self.step(m, fx.ctx, kind: .verify, tokens: Array(tokens[6...]), logits: .all)
        let vocab = m.config.vocabSize
        for r in 0..<4 {
            #expect(Array(all[r * vocab ..< (r + 1) * vocab]).map(\.bitPattern) == perRow[r].map(\.bitPattern))
        }
        // Keep two verify rows, then decode the third again: same logits as before.
        let cb = fx.ctx.queue.makeCommandBuffer()!
        try m.kv.rollback(to: 8, commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
        let again = try Self.step(m, fx.ctx, kind: .decode, tokens: [tokens[8]], logits: .last)
        #expect(again.map(\.bitPattern) == perRow[2].map(\.bitPattern))
    }

    final class CollectingTap: Native.TensorTap {
        let device: MTLDevice
        var captured: [Native.TapPoint: Native.TensorRef] = [:]
        init(device: MTLDevice) { self.device = device }
        func destination(for point: Native.TapPoint, like t: Native.TensorRef) -> Native.TensorRef? {
            let b = device.makeBuffer(length: t.rows * t.cols * t.dtype.byteSize, options: .storageModeShared)!
            let d = Native.TensorRef(buffer: b, dtype: t.dtype, rows: t.rows, cols: t.cols)
            captured[point] = d
            return d
        }
    }

    @Test func tapsCaptureEveryPointWithoutChangingLogits() throws {
        let fx = try Self.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let m = try fx.model()
        let tokens: [Int32] = [262, 104, 101, 108]
        let plain = try Self.step(m, fx.ctx, kind: .prefill, tokens: tokens, logits: .last)
        try Self.reset(m, fx.ctx)
        let tap = CollectingTap(device: fx.ctx.device)
        m.tap = tap
        let tapped = try Self.step(m, fx.ctx, kind: .prefill, tokens: tokens, logits: .last)
        #expect(tapped.map(\.bitPattern) == plain.map(\.bitPattern))
        let c = m.config
        #expect(tap.captured.count == 3 + 4 * c.blockCount)
        let logits = try #require(tap.captured[.logits])
        #expect(RefTest.floats(logits) == tapped)
        // layerOutput(i) == layerOutput(i-1) + mixerOutput(i) + ffnOutput(i), row by row.
        let l0 = RefTest.floats(try #require(tap.captured[.layerOutput(layer: 0)]))
        let l1 = RefTest.floats(try #require(tap.captured[.layerOutput(layer: 1)]))
        let mix = RefTest.floats(try #require(tap.captured[.mixerOutput(layer: 1)]))
        let ffn = RefTest.floats(try #require(tap.captured[.ffnOutput(layer: 1)]))
        for i in l1.indices { #expect(l1[i] == (l0[i] + mix[i]) + ffn[i]) }
        let embedding = try #require(tap.captured[.embedding])
        #expect(embedding.rows == 4 && embedding.cols == c.hiddenSize)
        #expect(try #require(tap.captured[.finalNorm]).rows == 1)
    }

    actor Events {
        var tokens: [Int32] = []
        var progress: [Int] = []
        func add(_ e: Native.StepEvent) {
            switch e {
            case .token(let t): tokens.append(t)
            case .prefillProgress(let done, _): progress.append(done)
            }
        }
    }

    @Test func schedulerGeneratesGreedilyAndReusesPrefixesExactly() async throws {
        let fx = try Self.fixture()
        defer { try? FileManager.default.removeItem(at: fx.dir) }
        let model = try fx.model(chunk: 4)
        let sampler = try Native.ReferenceSampler(context: fx.ctx)
        let scheduler = try Native.ReferenceStepSchedulerFactory().make(
            model: model, sampler: sampler, strategy: Native.ReferencePlainDecoding(), queue: fx.ctx.queue)
        let params = Native.SamplingParams(temperature: 0, topK: nil, topP: nil, minP: 0, repetitionPenalty: 1,
                                           presencePenalty: 0, frequencyPenalty: 0, repeatLastN: 0, seed: nil)
        let prompt: [Int32] = [262, 104, 101, 108, 108, 111, 263, 262, 30]
        let events = Events()
        let job = Native.GenerationJob(promptTokens: prompt, params: params, maxNewTokens: 6, stopTokens: [],
                                       reusePrefix: true, prefillChunk: 4)
        let cold = try await scheduler.run(job) { e in Task { await events.add(e) } }
        #expect(cold.tokens.count == 6 && cold.stopReason == .maxTokens && cold.reusedPrefixTokens == 0)

        // The same continuation computed by hand, one decode step at a time.
        let manual = try Self.fixture()
        defer { try? FileManager.default.removeItem(at: manual.dir) }
        let m = try manual.model(chunk: 16)
        var logits = try Self.step(m, manual.ctx, kind: .prefill, tokens: prompt, logits: .last)
        var expected: [Int32] = []
        for _ in 0..<6 {
            let t = Int32(Self.argmax(logits))
            expected.append(t)
            logits = try Self.step(m, manual.ctx, kind: .decode, tokens: [t], logits: .last)
        }
        #expect(cold.tokens == expected)

        // A follow-up sharing the prompt reuses it and generates the same tokens
        // as a cold run of the follow-up.
        let followUp = prompt + cold.tokens.prefix(3) + [263]
        let warm = try await scheduler.run(Native.GenerationJob(
            promptTokens: followUp, params: params, maxNewTokens: 5, stopTokens: [], reusePrefix: true,
            prefillChunk: 4)) { _ in }
        #expect(warm.reusedPrefixTokens == prompt.count)
        let coldFollowUp = try await scheduler.run(Native.GenerationJob(
            promptTokens: followUp, params: params, maxNewTokens: 5, stopTokens: [], reusePrefix: false,
            prefillChunk: 4)) { _ in }
        #expect(coldFollowUp.reusedPrefixTokens == 0)
        #expect(warm.tokens == coldFollowUp.tokens)

        // Stop tokens end generation without being emitted.
        let stop = try await scheduler.run(Native.GenerationJob(
            promptTokens: prompt, params: params, maxNewTokens: 6, stopTokens: [expected[2]], reusePrefix: false,
            prefillChunk: 4)) { _ in }
        #expect(stop.stopReason == .stopToken && stop.tokens == Array(expected.prefix(2)))

        // Cancelled before the first step throws.
        let s2 = try Native.ReferenceStepSchedulerFactory().make(
            model: try fx.model(chunk: 4), sampler: try Native.ReferenceSampler(context: nil),
            strategy: Native.ReferencePlainDecoding(), queue: fx.ctx.queue) as! Native.ReferenceStepScheduler
        let runTask = Task { try await s2.run(job) { _ in s2.cancel() } }
        let cancelled = try await runTask.value
        #expect(cancelled.stopReason == .cancelled)
    }
}
