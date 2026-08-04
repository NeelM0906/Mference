import Testing
import Foundation
import Metal
@testable import Mference

/// CPU cross-check of the Inkling GPU forward pass against the real
/// installed weights, one layer at a time. Scalar reference implements the
/// contract in docs/INKLING_SMALL.md directly from the resident tensors;
/// the GPU side runs `produce()` with MFERENCE_INKLING_STOP_LAYER so the
/// scratch buffers stay inspectable at the boundary.
///
/// Env-gated on MFERENCE_INKLING_GTURBO (and the stop layer set by the test
/// runner command line); skipped otherwise.
@Suite struct InklingLayerParityTests {

    // MARK: - CPU reference primitives

    private static func bf16(_ bits: UInt16) -> Float { Quantization.bf16ToFloat(bits) }

    /// Dequantized row `r` of an INT4 affine [m, n] tensor.
    private static func dequantRow(_ v: TensorView, row: Int, n: Int) -> [Float] {
        let base = v.buffer.contents()
        let wRow = base.advanced(by: Int(v.offset) + row * n / 2)
            .assumingMemoryBound(to: UInt8.self)
        let sRow = base.advanced(by: Int(v.scaleOffset) + row * (n / 64) * 2)
            .assumingMemoryBound(to: UInt16.self)
        let bRow = base.advanced(by: Int(v.biasOffset) + row * (n / 64) * 2)
            .assumingMemoryBound(to: UInt16.self)
        var out = [Float](repeating: 0, count: n)
        for g in 0..<(n / 64) {
            let scale = bf16(sRow[g]), bias = bf16(bRow[g])
            for k in 0..<64 {
                let b = wRow[g * 32 + k / 2]
                let nib = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[g * 64 + k] = Float(nib) * scale + bias
            }
        }
        return out
    }

    private static func gemv(_ v: TensorView, x: [Float], m: Int, n: Int) -> [Float] {
        var y = [Float](repeating: 0, count: m)
        for r in 0..<m {
            let w = dequantRow(v, row: r, n: n)
            var acc: Float = 0
            for i in 0..<n { acc += w[i] * x[i] }
            y[r] = acc
        }
        return y
    }

    private static func rms(_ x: [Float], _ w: TensorView, eps: Float = 1e-6) -> [Float] {
        let wp = w.buffer.contents().advanced(by: Int(w.offset))
            .assumingMemoryBound(to: UInt16.self)
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1.0 / (ss / Float(x.count) + eps).squareRoot()
        return (0..<x.count).map { x[$0] * inv * bf16(wp[$0]) }
    }

    private static func perHeadRMS(_ x: [Float], heads: Int, hd: Int,
                                   _ w: TensorView, eps: Float = 1e-6) -> [Float] {
        let wp = w.buffer.contents().advanced(by: Int(w.offset))
            .assumingMemoryBound(to: UInt16.self)
        var out = x
        for h in 0..<heads {
            var ss: Float = 0
            for i in 0..<hd { ss += x[h * hd + i] * x[h * hd + i] }
            let inv = 1.0 / (ss / Float(hd) + eps).squareRoot()
            for i in 0..<hd { out[h * hd + i] = x[h * hd + i] * inv * bf16(wp[i]) }
        }
        return out
    }

    /// Zero-state sconv step: out = taps[K-1]*x + x (history is zero).
    private static func sconvZeroState(_ x: [Float], _ w: TensorView, k: Int) -> [Float] {
        let wp = w.buffer.contents().advanced(by: Int(w.offset))
            .assumingMemoryBound(to: UInt16.self)
        return (0..<x.count).map { c in
            x[c] * bf16(wp[c * k + (k - 1)]) + x[c]
        }
    }

    private static func readF16(_ b: MTLBuffer, _ n: Int) -> [Float] {
        let p = b.contents().bindMemory(to: Float16.self, capacity: n)
        return (0..<n).map { Float(p[$0]) }
    }
    private static func readF32(_ b: MTLBuffer, _ n: Int) -> [Float] {
        let p = b.contents().bindMemory(to: Float.self, capacity: n)
        return (0..<n).map { p[$0] }
    }

    private static func compare(_ label: String, gpu: [Float], cpu: [Float],
                                tol: Float) -> Bool {
        var dot: Float = 0, ng: Float = 0, nc: Float = 0
        var maxAbsDiff: Float = 0
        for i in 0..<gpu.count {
            dot += gpu[i] * cpu[i]; ng += gpu[i] * gpu[i]; nc += cpu[i] * cpu[i]
            maxAbsDiff = max(maxAbsDiff, abs(gpu[i] - cpu[i]))
        }
        let cos = dot / (ng.squareRoot() * nc.squareRoot() + 1e-20)
        print("[parity] \(label): cos=\(cos) maxAbsDiff=\(maxAbsDiff) "
            + "gpuNorm=\(ng.squareRoot()) cpuNorm=\(nc.squareRoot())")
        return cos > tol
    }

    // MARK: - Layer 0 (dense) parity, token 0

    @Test func layer0DenseParity() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            RealForwardRunner.inklingStopLayer == 0 else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        let token: Int32 = 3575   // "The"
        try await runner.produce(token: token, position: 0, into: logits)

        let D = cfg.hiddenSize
        let HD = cfg.headDim, NQ = cfg.numHeads, NKV = cfg.numKVHeads
        let K = cfg.sconvKernelSize

        // CPU reference, layer 0.
        let embedRow = Self.dequantRow(model.embedding, row: Int(token), n: D)
        let h0 = Self.rms(embedRow, model.embedNorm)
        let normed = Self.rms(h0, try model.inputNorm(layer: 0))
        #expect(Self.compare("L0 normed", gpu: Self.readF16(runner.normed, D),
                             cpu: normed, tol: 0.999))

        let qRaw = Self.gemv(try model.inklingWqDu(layer: 0), x: normed,
                             m: NQ * HD, n: D)
        let kRaw = Self.gemv(try model.inklingWkDv(layer: 0), x: normed,
                             m: NKV * HD, n: D)
        let vRaw = Self.gemv(try model.inklingWvDv(layer: 0), x: normed,
                             m: NKV * HD, n: D)
        let kConv = Self.sconvZeroState(kRaw, try model.inklingKSconv(layer: 0), k: K)
        let vConv = Self.sconvZeroState(vRaw, try model.inklingVSconv(layer: 0), k: K)
        let q = Self.perHeadRMS(qRaw, heads: NQ, hd: HD,
                                try model.inklingQNorm(layer: 0))
        _ = Self.perHeadRMS(kConv, heads: NKV, hd: HD,
                            try model.inklingKNorm(layer: 0))
        #expect(Self.compare("L0 q(post-norm)",
                             gpu: Self.readF16(runner.qScratch, NQ * HD),
                             cpu: q, tol: 0.999))

        // Position 0 attention: softmax over one key == that key's V row
        // (per kv head, replicated over its 4 query heads).
        var attnOut = [Float](repeating: 0, count: NQ * HD)
        for h in 0..<NQ {
            let kvh = h / (NQ / NKV)
            for i in 0..<HD { attnOut[h * HD + i] = vConv[kvh * HD + i] }
        }
        #expect(Self.compare("L0 attnOut",
                             gpu: Self.readF16(runner.attnOut, NQ * HD),
                             cpu: attnOut, tol: 0.999))

        let oOut = Self.gemv(try model.inklingWoUd(layer: 0), x: attnOut,
                             m: D, n: NQ * HD)
        #expect(Self.compare("L0 oOut", gpu: Self.readF16(runner.oOut, D),
                             cpu: oOut, tol: 0.999))

        let attnDelta = Self.sconvZeroState(oOut, try model.inklingAttnSconv(layer: 0), k: K)
        var hidden1 = (0..<D).map { h0[$0] + attnDelta[$0] }

        // Dense MLP.
        let routedX = Self.rms(hidden1, try model.postAttnNorm(layer: 0))
        #expect(Self.compare("L0 routedX",
                             gpu: Self.readF16(runner.routedX, D),
                             cpu: routedX, tol: 0.999))
        let FD = cfg.denseIntermediateSize
        let g = Self.gemv(try model.inklingDenseGate(layer: 0), x: routedX,
                          m: FD, n: D)
        let u = Self.gemv(try model.inklingDenseUp(layer: 0), x: routedX,
                          m: FD, n: D)
        var act = [Float](repeating: 0, count: FD)
        for i in 0..<FD { act[i] = g[i] / (1 + exp(-g[i])) * u[i] }
        var mlp = Self.gemv(try model.inklingDenseDown(layer: 0), x: act,
                            m: D, n: FD)
        let gain = model.inklingScalar(try model.inklingDenseGlobalScale(layer: 0))
        for i in 0..<D { mlp[i] *= gain }
        let mlpDelta = Self.sconvZeroState(mlp, try model.inklingMlpSconv(layer: 0), k: K)
        for i in 0..<D { hidden1[i] += mlpDelta[i] }

        #expect(Self.compare("L0 hidden(final)",
                             gpu: Self.readF32(runner.inklingHiddenF32!, D),
                             cpu: hidden1, tol: 0.999))
    }
}

extension InklingLayerParityTests {
    /// Layer-2 (first MoE) parity for token 0: router selection + weights,
    /// shared-expert combine, routed experts from the streamed blob, and the
    /// final residual. Requires MFERENCE_INKLING_STOP_LAYER=2.
    @Test func layer2MoEParity() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            RealForwardRunner.inklingStopLayer == 2 else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        try await runner.produce(token: 3575, position: 0, into: logits)

        let D = cfg.hiddenSize
        let F = cfg.moeIntermediateSize

        // GPU state at the layer-2 boundary.
        let routedX = Self.readF16(runner.routedX, D)
        let gpuIdx = runner.outIndices.contents()
            .bindMemory(to: UInt32.self, capacity: 6)
        let gpuW = Self.readF16(runner.outWeights, 8)
        let gammas = Self.readF32(runner.inkling!.sharedGammas, 2)

        // --- CPU router over the same routedX (fp32).
        let routerW = try model.router(layer: 2)
        let wp = routerW.buffer.contents().advanced(by: Int(routerW.offset))
            .assumingMemoryBound(to: UInt16.self)
        var logitsR = [Float](repeating: 0, count: 258)
        for e in 0..<258 {
            var acc: Float = 0
            for i in 0..<D { acc += Self.bf16(wp[e * D + i]) * routedX[i] }
            logitsR[e] = acc
        }
        let biasV = try model.inklingGateBias(layer: 2)
        let bp = biasV.buffer.contents().advanced(by: Int(biasV.offset))
            .assumingMemoryBound(to: Float.self)
        let scores = (0..<256).map { 1 / (1 + exp(-logitsR[$0])) + bp[$0] }
        let refIdx = scores.enumerated().sorted { $0.element > $1.element }
            .prefix(6).map(\.offset)
        func logsigmoid(_ x: Float) -> Float { min(x, 0) - log(1 + exp(-abs(x))) }
        let sel = refIdx.map { logsigmoid(logitsR[$0]) }
            + [logsigmoid(logitsR[256]), logsigmoid(logitsR[257])]
        let mx = sel.max()!
        let exps = sel.map { expf($0 - mx) }
        let denom = exps.reduce(0, +)
        let gScale = model.inklingScalar(try model.inklingGateGlobalScale(layer: 2))
        let refW = exps.map { $0 / denom * 8.0 * gScale }
        print("[parity] L2 router idx gpu=\((0..<6).map { Int(gpuIdx[$0]) }) cpu=\(refIdx)")
        print("[parity] L2 router w gpu=\(gpuW) cpu=\(refW) gammas gpu=\(gammas)")
        #expect(Set((0..<6).map { Int(gpuIdx[$0]) }) == Set(refIdx))

        // --- CPU shared experts + routed experts.
        func switchGLU(_ gate: TensorView, _ up: TensorView, _ down: TensorView,
                       x: [Float]) -> [Float] {
            let g = Self.gemv(gate, x: x, m: F, n: D)
            let u = Self.gemv(up, x: x, m: F, n: D)
            var act = [Float](repeating: 0, count: F)
            for i in 0..<F { act[i] = g[i] / (1 + exp(-g[i])) * u[i] }
            return Self.gemv(down, x: act, m: D, n: F)
        }
        var cpuOut = [Float](repeating: 0, count: D)
        for s in 0..<2 {
            let y = switchGLU(
                try model.inklingSharedExpert("gate_proj", layer: 2, expert: s, of: 2),
                try model.inklingSharedExpert("up_proj", layer: 2, expert: s, of: 2),
                try model.inklingSharedExpert("down_proj", layer: 2, expert: s, of: 2),
                x: routedX)
            for i in 0..<D { cpuOut[i] += refW[6 + s] * y[i] }
        }
        #expect(Self.compare("L2 shared(h1)", gpu: Self.readF16(runner.h1Buf, D),
                             cpu: cpuOut, tol: 0.995))

        let offs = model.routedExpertOffsets(layer: 2)
        let blobs = try await model.fetchRoutedExperts(
            layer: 2, experts: (0..<6).map { Int(gpuIdx[$0]) })
        func blobView(_ blob: TensorView, w: UInt32, s: UInt32, b: UInt32,
                      m: Int, n: Int) -> TensorView {
            TensorView(buffer: blob.buffer,
                       offset: blob.offset + UInt64(w), length: UInt64(m * n / 2),
                       scaleOffset: blob.offset + UInt64(s),
                       scaleLength: UInt64(m * n / 64 * 2),
                       biasOffset: blob.offset + UInt64(b),
                       biasLength: UInt64(m * n / 64 * 2),
                       shape: (UInt32(m), UInt32(n), 1, 1), dtype: 0)
        }
        for (slot, blob) in blobs.enumerated() {
            let y = switchGLU(
                blobView(blob, w: offs.gateWOff, s: offs.gateSOff, b: offs.gateBOff, m: F, n: D),
                blobView(blob, w: offs.upWOff, s: offs.upSOff, b: offs.upBOff, m: F, n: D),
                blobView(blob, w: offs.downWOff, s: offs.downSOff, b: offs.downBOff, m: D, n: F),
                x: routedX)
            for i in 0..<D { cpuOut[i] += Float(gpuW[slot]) * y[i] }
        }
        #expect(Self.compare("L2 moe out(h2)", gpu: Self.readF16(runner.h2Buf, D),
                             cpu: cpuOut, tol: 0.995))
    }
}

extension InklingLayerParityTests {
    /// Head parity: full 42-layer pass, then CPU logits for probe ids from
    /// the final normed row. No stop layer.
    @Test func headParity() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            RealForwardRunner.inklingStopLayer == -1 else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        try await runner.produce(token: 3575, position: 0, into: logits)

        let D = cfg.hiddenSize
        let normed = Self.readF16(runner.normed, D)
        // CPU: recompute final norm from the FP32 stream and cross-check.
        let hiddenF32 = Self.readF32(runner.inklingHiddenF32!, D)
        let cpuNormed = Self.rms(hiddenF32, model.finalNorm)
        #expect(Self.compare("final normed", gpu: normed, cpu: cpuNormed,
                             tol: 0.999))

        let lp = logits.contents().bindMemory(to: Float16.self,
                                              capacity: cfg.vocabSize)
        let probeIDs = [0, 2, 410, 877, 12194, 100000, 199999]
        for id in probeIDs {
            let row = Self.dequantRow(model.lmHead, row: id, n: D)
            var acc: Float = 0
            for i in 0..<D { acc += row[i] * cpuNormed[i] }
            let cpuLogit = acc / 16.0
            print("[parity] logit[\(id)]: gpu=\(Float(lp[id])) cpu=\(cpuLogit)")
        }
        // Padding tail must be -inf.
        print("[parity] pad logit[200100]=\(Float(lp[200100])) [201000]=\(Float(lp[201000]))")
    }
}

extension InklingLayerParityTests {
    /// Token-1 parity at layer 0: exercises everything position 0 cannot —
    /// conv-state carry across tokens, the KV cache, 2-position softmax, and
    /// (critically) the relative-position bias. Requires
    /// MFERENCE_INKLING_STOP_LAYER=0.
    @Test func token1AttentionParity() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            RealForwardRunner.inklingStopLayer == 0 else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        let t0: Int32 = 3575, t1: Int32 = 6746
        try await runner.produce(token: t0, position: 0, into: logits)
        // Stop-layer returns early WITHOUT kv.advance(); advance manually so
        // position 1 lands in the next slot exactly as a full pass would.
        runner.inklingParityAdvanceKV()
        try await runner.produce(token: t1, position: 1, into: logits)

        let D = cfg.hiddenSize, HD = cfg.headDim
        let NQ = cfg.numHeads, NKV = cfg.numKVHeads, K = cfg.sconvKernelSize
        let dRel = cfg.relativePosition.dRel
        let scale = Float(cfg.attentionScale)

        func layer0Proj(_ token: Int32) -> (normed: [Float], kRaw: [Float],
                                            vRaw: [Float], qRaw: [Float],
                                            rel: [Float], h0: [Float]) {
            let e = Self.dequantRow(model.embedding, row: Int(token), n: D)
            let h0 = Self.rms(e, model.embedNorm)
            let nrm = Self.rms(h0, try! model.inputNorm(layer: 0))
            return (nrm,
                    Self.gemv(try! model.inklingWkDv(layer: 0), x: nrm, m: NKV * HD, n: D),
                    Self.gemv(try! model.inklingWvDv(layer: 0), x: nrm, m: NKV * HD, n: D),
                    Self.gemv(try! model.inklingWqDu(layer: 0), x: nrm, m: NQ * HD, n: D),
                    Self.gemv(try! model.inklingWrDu(layer: 0), x: nrm, m: NQ * dRel, n: D),
                    h0)
        }
        let p0 = layer0Proj(t0)
        let p1 = layer0Proj(t1)

        let kSconv = try model.inklingKSconv(layer: 0)
        let vSconv = try model.inklingVSconv(layer: 0)
        let kTaps = kSconv.buffer.contents().advanced(by: Int(kSconv.offset))
            .assumingMemoryBound(to: UInt16.self)
        let vTaps = vSconv.buffer.contents().advanced(by: Int(vSconv.offset))
            .assumingMemoryBound(to: UInt16.self)
        func conv(_ taps: UnsafePointer<UInt16>, _ prev: [Float]?, _ x: [Float]) -> [Float] {
            (0..<x.count).map { c in
                let t2 = Self.bf16(taps[c * K + (K - 2)])
                let t3 = Self.bf16(taps[c * K + (K - 1)])
                return (prev.map { t2 * $0[c] } ?? 0) + t3 * x[c] + x[c]
            }
        }
        let k0 = Self.perHeadRMS(conv(kTaps, nil, p0.kRaw), heads: NKV, hd: HD,
                                 try model.inklingKNorm(layer: 0))
        let v0 = conv(vTaps, nil, p0.vRaw)
        let k1 = Self.perHeadRMS(conv(kTaps, p0.kRaw, p1.kRaw), heads: NKV, hd: HD,
                                 try model.inklingKNorm(layer: 0))
        let v1 = conv(vTaps, p0.vRaw, p1.vRaw)
        let q1 = Self.perHeadRMS(p1.qRaw, heads: NQ, hd: HD,
                                 try model.inklingQNorm(layer: 0))
        #expect(Self.compare("t1 q(post-norm)",
                             gpu: Self.readF16(runner.qScratch, NQ * HD),
                             cpu: q1, tol: 0.999))

        // Attention at q_pos=1 over kv 0..1 with the relative bias.
        let relProj = try model.inklingRelProj(layer: 0)
        let projPtr = relProj.buffer.contents().advanced(by: Int(relProj.offset))
            .assumingMemoryBound(to: UInt16.self)
        let extent = cfg.slidingWindow   // local layer bias width
        var attnOut = [Float](repeating: 0, count: NQ * HD)
        for h in 0..<NQ {
            let kvh = h / (NQ / NKV)
            var lg = [Float](repeating: 0, count: 2)
            for (idx, kv) in [(0, k0), (1, k1)].map({ ($0.0, $0.1) }) {
                var qk: Float = 0
                for i in 0..<HD { qk += q1[h * HD + i] * kv[kvh * HD + i] }
                let dist = 1 - idx
                var bias: Float = 0
                for i in 0..<dRel {
                    bias += p1.rel[h * dRel + i]
                        * Self.bf16(projPtr[i * extent + dist])
                }
                lg[idx] = qk * scale + bias
            }
            let m = max(lg[0], lg[1])
            let e0 = expf(lg[0] - m), e1 = expf(lg[1] - m)
            let w0 = e0 / (e0 + e1), w1 = e1 / (e0 + e1)
            for i in 0..<HD {
                attnOut[h * HD + i] = w0 * v0[kvh * HD + i] + w1 * v1[kvh * HD + i]
            }
        }
        #expect(Self.compare("t1 attnOut",
                             gpu: Self.readF16(runner.attnOut, NQ * HD),
                             cpu: attnOut, tol: 0.999))
    }
}

extension InklingLayerParityTests {
    /// Global-layer (L5) attention isolation: takes the GPU's own q/k/v/rel
    /// at the stop boundary and CPU-computes the expected attention output
    /// with global semantics (extent 1024, no window). Requires
    /// MFERENCE_INKLING_STOP_LAYER=5.
    @Test func layer5GlobalAttentionParity() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            RealForwardRunner.inklingStopLayer == 5 else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        try await runner.produce(token: 976, position: 0, into: logits)
        runner.inklingParityAdvanceKV()
        try await runner.produce(token: 9029, position: 1, into: logits)

        let HD = cfg.headDim, NQ = cfg.numHeads, NKV = cfg.numKVHeads
        let dRel = cfg.relativePosition.dRel
        let extent = cfg.relativePosition.extent   // 1024 on global layers
        let scale = Float(cfg.attentionScale)
        let L = 5

        let q = Self.readF16(runner.qScratch, NQ * HD)
        let rel = Self.readF16(runner.inklingRelScratch!, NQ * dRel)
        func slotRow(_ s: (buffer: MTLBuffer, offset: Int)) -> [Float] {
            let p = s.buffer.contents().advanced(by: s.offset)
                .bindMemory(to: Float16.self, capacity: NKV * HD)
            return (0..<NKV * HD).map { Float(p[$0]) }
        }
        let k0 = slotRow(runner.inklingParityKSlot(layer: L, position: 0))
        let k1 = slotRow(runner.inklingParityKSlot(layer: L, position: 1))
        let v0 = slotRow(runner.inklingParityVSlot(layer: L, position: 0))
        let v1 = slotRow(runner.inklingParityVSlot(layer: L, position: 1))

        let relProj = try model.inklingRelProj(layer: L)
        // Sanity: global proj bank is [dRel, 1024] BF16.
        #expect(Int(relProj.length) == dRel * extent * 2,
                "L5 proj length \(relProj.length) != \(dRel * extent * 2)")
        let projPtr = relProj.buffer.contents().advanced(by: Int(relProj.offset))
            .assumingMemoryBound(to: UInt16.self)

        var attnOut = [Float](repeating: 0, count: NQ * HD)
        for h in 0..<NQ {
            let kvh = h / (NQ / NKV)
            var lg = [Float](repeating: 0, count: 2)
            for (idx, kk) in [k0, k1].enumerated() {
                var qk: Float = 0
                for i in 0..<HD { qk += q[h * HD + i] * kk[kvh * HD + i] }
                let dist = 1 - idx
                var bias: Float = 0
                for i in 0..<dRel {
                    bias += rel[h * dRel + i] * Self.bf16(projPtr[i * extent + dist])
                }
                lg[idx] = qk * scale + bias
            }
            let m = max(lg[0], lg[1])
            let e0 = expf(lg[0] - m), e1 = expf(lg[1] - m)
            let w0 = e0 / (e0 + e1), w1 = e1 / (e0 + e1)
            for i in 0..<HD {
                attnOut[h * HD + i] = w0 * v0[kvh * HD + i] + w1 * v1[kvh * HD + i]
            }
        }
        #expect(Self.compare("L5 attnOut",
                             gpu: Self.readF16(runner.attnOut, NQ * HD),
                             cpu: attnOut, tol: 0.999))
    }
}

extension InklingLayerParityTests {
    /// Composition parity: 5-token prompt, tokens 0-3 run all 42 layers, the
    /// last token stops after L1. The CPU then computes token 4's ENTIRE
    /// layer 2 (attention with 5-position history + deep conv states + MoE)
    /// from the captured GPU state. Requires MFERENCE_INKLING_STOP_LAYER=1
    /// MFERENCE_INKLING_STOP_POSITION=4. Prints the CPU layer output for the
    /// companion run (stop layer 2) to diff externally.
    @Test func token4Layer2CompositionParity() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            RealForwardRunner.inklingStopLayer == 1,
            RealForwardRunner.inklingStopPosition == 4 else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        let prompt: [Int32] = [976, 9029, 328, 10128, 382]
        for (p, t) in prompt.enumerated() {
            try await runner.produce(token: t, position: p, into: logits)
            if p < prompt.count - 1 { /* full pass advanced internally */ }
        }
        // Token 4 stopped after L1 (no kv.advance).

        let D = cfg.hiddenSize, HD = cfg.headDim
        let NQ = cfg.numHeads, NKV = cfg.numKVHeads, K = cfg.sconvKernelSize
        let F = cfg.moeIntermediateSize
        let dRel = cfg.relativePosition.dRel
        let scale = Float(cfg.attentionScale)
        let L = 2

        let hin = Self.readF32(runner.inklingHiddenF32!, D)
        let normed = Self.rms(hin, try model.inputNorm(layer: L))
        let qRaw = Self.gemv(try model.inklingWqDu(layer: L), x: normed, m: NQ * HD, n: D)
        let kRaw = Self.gemv(try model.inklingWkDv(layer: L), x: normed, m: NKV * HD, n: D)
        let vRaw = Self.gemv(try model.inklingWvDv(layer: L), x: normed, m: NKV * HD, n: D)
        let rel = Self.gemv(try model.inklingWrDu(layer: L), x: normed, m: NQ * dRel, n: D)

        // Conv states for L2 hold t1-t3 history (t0 shifted out for K-1=3).
        let st = runner.inklingConvStates[L]
        func state(_ b: MTLBuffer, c: Int) -> [[Float]] {
            let p = b.contents().bindMemory(to: Float.self, capacity: c * (K - 1))
            return (0..<c).map { ch in (0..<(K - 1)).map { p[ch * (K - 1) + $0] } }
        }
        let kSt = state(st.k, c: NKV * HD)
        let vSt = state(st.v, c: NKV * HD)
        func convStep(_ w: TensorView, _ stt: [[Float]], _ x: [Float]) -> [Float] {
            let wp = w.buffer.contents().advanced(by: Int(w.offset))
                .assumingMemoryBound(to: UInt16.self)
            return (0..<x.count).map { c in
                var acc = x[c] * Self.bf16(wp[c * K + (K - 1)])
                for j in 0..<(K - 1) { acc += stt[c][j] * Self.bf16(wp[c * K + j]) }
                return acc + x[c]
            }
        }
        let k4 = Self.perHeadRMS(convStep(try model.inklingKSconv(layer: L), kSt, kRaw),
                                 heads: NKV, hd: HD, try model.inklingKNorm(layer: L))
        let v4 = convStep(try model.inklingVSconv(layer: L), vSt, vRaw)
        let q4 = Self.perHeadRMS(qRaw, heads: NQ, hd: HD, try model.inklingQNorm(layer: L))

        // Cached slots 0-3 (written by the full passes) + this token.
        var ks: [[Float]] = []
        var vs: [[Float]] = []
        for p in 0..<4 {
            let kp = runner.inklingParityKSlot(layer: L, position: p)
            let vp = runner.inklingParityVSlot(layer: L, position: p)
            let kptr = kp.buffer.contents().advanced(by: kp.offset)
                .bindMemory(to: Float16.self, capacity: NKV * HD)
            let vptr = vp.buffer.contents().advanced(by: vp.offset)
                .bindMemory(to: Float16.self, capacity: NKV * HD)
            ks.append((0..<NKV * HD).map { Float(kptr[$0]) })
            vs.append((0..<NKV * HD).map { Float(vptr[$0]) })
        }
        ks.append(k4)
        vs.append(v4)

        let relProj = try model.inklingRelProj(layer: L)
        let projPtr = relProj.buffer.contents().advanced(by: Int(relProj.offset))
            .assumingMemoryBound(to: UInt16.self)
        let extent = cfg.slidingWindow
        var attnOut = [Float](repeating: 0, count: NQ * HD)
        for h in 0..<NQ {
            let kvh = h / (NQ / NKV)
            var lgs = [Float]()
            for p in 0...4 {
                var qk: Float = 0
                for i in 0..<HD { qk += q4[h * HD + i] * ks[p][kvh * HD + i] }
                let dist = 4 - p
                var bias: Float = 0
                for i in 0..<dRel {
                    bias += rel[h * dRel + i] * Self.bf16(projPtr[i * extent + dist])
                }
                lgs.append(qk * scale + bias)
            }
            let m = lgs.max()!
            let es = lgs.map { expf($0 - m) }
            let den = es.reduce(0, +)
            for p in 0...4 {
                let w = es[p] / den
                for i in 0..<HD { attnOut[h * HD + i] += w * vs[p][kvh * HD + i] }
            }
        }
        let oOut = Self.gemv(try model.inklingWoUd(layer: L), x: attnOut, m: D, n: NQ * HD)
        let aSt = state(st.attn, c: D)
        let attnDelta = convStep(try model.inklingAttnSconv(layer: L), aSt, oOut)
        var hidden = (0..<D).map { hin[$0] + attnDelta[$0] }

        let routedX = Self.rms(hidden, try model.postAttnNorm(layer: L))
        // Router.
        let routerW = try model.router(layer: L)
        let wp = routerW.buffer.contents().advanced(by: Int(routerW.offset))
            .assumingMemoryBound(to: UInt16.self)
        var lgr = [Float](repeating: 0, count: 258)
        for e in 0..<258 {
            var acc: Float = 0
            for i in 0..<D { acc += Self.bf16(wp[e * D + i]) * routedX[i] }
            lgr[e] = acc
        }
        let biasV = try model.inklingGateBias(layer: L)
        let bp = biasV.buffer.contents().advanced(by: Int(biasV.offset))
            .assumingMemoryBound(to: Float.self)
        let scores = (0..<256).map { 1 / (1 + exp(-lgr[$0])) + bp[$0] }
        let refIdx = scores.enumerated().sorted { $0.element > $1.element }
            .prefix(6).map(\.offset)
        func logsigmoid(_ x: Float) -> Float { min(x, 0) - log(1 + exp(-abs(x))) }
        let sel = refIdx.map { logsigmoid(lgr[$0]) } + [logsigmoid(lgr[256]), logsigmoid(lgr[257])]
        let mxx = sel.max()!
        let exps = sel.map { expf($0 - mxx) }
        let den2 = exps.reduce(0, +)
        let gScale = model.inklingScalar(try model.inklingGateGlobalScale(layer: L))
        let refW = exps.map { $0 / den2 * 8.0 * gScale }

        func switchGLU(_ g: TensorView, _ u: TensorView, _ d: TensorView, x: [Float]) -> [Float] {
            let gv = Self.gemv(g, x: x, m: F, n: D)
            let uv = Self.gemv(u, x: x, m: F, n: D)
            var act = [Float](repeating: 0, count: F)
            for i in 0..<F { act[i] = gv[i] / (1 + exp(-gv[i])) * uv[i] }
            return Self.gemv(d, x: act, m: D, n: F)
        }
        var ffn = [Float](repeating: 0, count: D)
        for s2 in 0..<2 {
            let y = switchGLU(
                try model.inklingSharedExpert("gate_proj", layer: L, expert: s2, of: 2),
                try model.inklingSharedExpert("up_proj", layer: L, expert: s2, of: 2),
                try model.inklingSharedExpert("down_proj", layer: L, expert: s2, of: 2),
                x: routedX)
            for i in 0..<D { ffn[i] += refW[6 + s2] * y[i] }
        }
        let offs = model.routedExpertOffsets(layer: L)
        let blobs = try await model.fetchRoutedExperts(layer: L, experts: refIdx)
        func bv(_ blob: TensorView, w: UInt32, s: UInt32, b: UInt32, m: Int, n: Int) -> TensorView {
            TensorView(buffer: blob.buffer, offset: blob.offset + UInt64(w),
                       length: UInt64(m * n / 2),
                       scaleOffset: blob.offset + UInt64(s), scaleLength: UInt64(m * n / 32),
                       biasOffset: blob.offset + UInt64(b), biasLength: UInt64(m * n / 32),
                       shape: (UInt32(m), UInt32(n), 1, 1), dtype: 0)
        }
        for (i, blob) in blobs.enumerated() {
            let y = switchGLU(
                bv(blob, w: offs.gateWOff, s: offs.gateSOff, b: offs.gateBOff, m: F, n: D),
                bv(blob, w: offs.upWOff, s: offs.upSOff, b: offs.upBOff, m: F, n: D),
                bv(blob, w: offs.downWOff, s: offs.downSOff, b: offs.downBOff, m: D, n: F),
                x: routedX)
            for j in 0..<D { ffn[j] += refW[i] * y[j] }
        }
        // The GPU stores the mlp conv state pre-scaled by 1/32 (the FFN
        // prescale); un-scale it to compare against this unscaled reference.
        let mSt = state(st.mlp, c: D).map { $0.map { $0 * RealForwardRunner.inklingFFNPrescale } }
        let mlpDelta = convStep(try model.inklingMlpSconv(layer: L), mSt, ffn)
        for i in 0..<D { hidden[i] += mlpDelta[i] }

        // Persist for the companion GPU run to diff against.
        func dump(_ name: String, _ v: [Float]) throws {
            try v.map { String($0) }.joined(separator: "\n")
                .write(toFile: NSTemporaryDirectory() + "inkling_t4L2_\(name).txt",
                       atomically: true, encoding: .utf8)
        }
        try dump("cpu", hidden)
        try dump("normed", normed)
        try dump("q", q4)
        try dump("attn", attnOut)
        try dump("routedX", routedX)
        print("[parity] CPU t4/L2 intermediates written")
        print("[parity] CPU router idx=\(refIdx)")
    }

    /// Companion: same prompt, stop layer 2 at position 4; diffs GPU hidden
    /// against the CPU file from token4Layer2CompositionParity.
    @Test func token4Layer2CompositionGPU() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            RealForwardRunner.inklingStopLayer == 2,
            RealForwardRunner.inklingStopPosition == 4 else { return }
        let cpuPath = NSTemporaryDirectory() + "inkling_t4L2_cpu.txt"
        guard let txt = try? String(contentsOfFile: cpuPath, encoding: .utf8) else {
            Issue.record("run token4Layer2CompositionParity first")
            return
        }
        let cpu = txt.split(separator: "\n").map { Float($0)! }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        for (p, t) in [Int32(976), 9029, 328, 10128, 382].enumerated() {
            try await runner.produce(token: t, position: p, into: logits)
        }
        func load(_ name: String) -> [Float] {
            let t = try! String(contentsOfFile: NSTemporaryDirectory()
                + "inkling_t4L2_\(name).txt", encoding: .utf8)
            return t.split(separator: "\n").map { Float($0)! }
        }
        let D = cfg.hiddenSize
        _ = Self.compare("t4/L2 normed", gpu: Self.readF16(runner.normed, D),
                         cpu: load("normed"), tol: 0.999)
        _ = Self.compare("t4/L2 q", gpu: Self.readF16(runner.qScratch,
                                                      cfg.numHeads * cfg.headDim),
                         cpu: load("q"), tol: 0.999)
        _ = Self.compare("t4/L2 attnOut", gpu: Self.readF16(runner.attnOut,
                                                            cfg.numHeads * cfg.headDim),
                         cpu: load("attn"), tol: 0.999)
        _ = Self.compare("t4/L2 routedX", gpu: Self.readF16(runner.routedX, D),
                         cpu: load("routedX"), tol: 0.999)
        let idxPtr = runner.outIndices.contents()
            .bindMemory(to: UInt32.self, capacity: 6)
        print("[parity] GPU router idx=\((0..<6).map { Int(idxPtr[$0]) })")
        let gpu = Self.readF32(runner.inklingHiddenF32!, cfg.hiddenSize)
        #expect(Self.compare("t4/L2 hidden (composition)",
                             gpu: gpu, cpu: cpu, tol: 0.999))
    }
}

extension InklingLayerParityTests {
    /// Dumps the engine's layer-0 sub-stage buffers for token 4 of the fixed
    /// 5-token prompt, for diffing against the reference oracle. Requires
    /// MFERENCE_INKLING_STOP_LAYER=0 MFERENCE_INKLING_STOP_POSITION=4.
    @Test func token4Layer0StageDump() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            let outDir = ProcessInfo.processInfo.environment["MFERENCE_STAGE_OUT"],
            RealForwardRunner.inklingStopLayer == 0,
            RealForwardRunner.inklingStopPosition == 4 else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        for (p, t) in [Int32(976), 9029, 328, 10128, 382].enumerated() {
            try await runner.produce(token: t, position: p, into: logits)
        }
        try FileManager.default.createDirectory(
            atPath: outDir, withIntermediateDirectories: true)
        func dumpF16(_ name: String, _ b: MTLBuffer, _ n: Int, offset: Int = 0) throws {
            let p = b.contents().advanced(by: offset)
                .bindMemory(to: Float16.self, capacity: n)
            let v = (0..<n).map { Float(p[$0]) }
            try v.withUnsafeBytes { raw in
                try Data(raw).write(to: URL(fileURLWithPath: "\(outDir)/\(name).f32raw"))
            }
        }
        let D = cfg.hiddenSize
        let kvDim = cfg.numKVHeads * cfg.headDim
        try dumpF16("normed", runner.normed, D)
        try dumpF16("q_normed", runner.qScratch, cfg.numHeads * cfg.headDim)
        try dumpF16("rel", runner.inklingRelScratch!, cfg.relativePosition.projDim)
        try dumpF16("attn_out", runner.attnOut, cfg.numHeads * cfg.headDim)
        try dumpF16("o_out", runner.oOut, D)
        for p in 0...4 {
            let ks = runner.inklingParityKSlot(layer: 0, position: p)
            let vs = runner.inklingParityVSlot(layer: 0, position: p)
            try dumpF16("k_slot\(p)", ks.buffer, kvDim, offset: ks.offset)
            try dumpF16("v_slot\(p)", vs.buffer, kvDim, offset: vs.offset)
        }
        // FP32 dumps: the mlp delta (deltaF32 holds the last sconv output at
        // the stop point) and the final layer output.
        func dumpF32(_ name: String, _ b: MTLBuffer, _ n: Int) throws {
            try Data(bytes: b.contents(), count: n * 4)
                .write(to: URL(fileURLWithPath: "\(outDir)/\(name).f32raw"))
        }
        try dumpF32("mlp_delta", runner.inklingDeltaF32!, D)
        try dumpF32("layer_out", runner.inklingHiddenF32!, D)
        print("[stage] dumped to \(outDir)")
    }
}

extension InklingLayerParityTests {
    /// Teacher-forced logits dump for the KLD eval: feeds the token list in
    /// MFERENCE_KLD_IDS through produce() and appends each position's logits
    /// (unpadded vocab, f32) to MFERENCE_KLD_OUT.
    @Test func teacherForcedLogitsDump() async throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"],
            let idsPath = ProcessInfo.processInfo.environment["MFERENCE_KLD_IDS"],
            let outPath = ProcessInfo.processInfo.environment["MFERENCE_KLD_OUT"]
        else { return }
        let idsData = try Data(contentsOf: URL(fileURLWithPath: idsPath))
        let ids = try JSONDecoder().decode([Int32].self, from: idsData)
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 4096)
        let logits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared)!
        let valid = cfg.unpaddedVocabSize
        FileManager.default.createFile(atPath: outPath, contents: nil)
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: outPath))
        for (p, t) in ids.enumerated() {
            try await runner.produce(token: t, position: p, into: logits)
            let lp = logits.contents().bindMemory(to: Float16.self, capacity: valid)
            let row = (0..<valid).map { Float(lp[$0]) }
            row.withUnsafeBytes { handle.write(Data($0)) }
        }
        try handle.close()
        print("[kld] wrote \(ids.count) x \(valid) logits to \(outPath)")
    }
}

extension InklingLayerParityTests {
    /// Conv-state lifecycle: `reset()` clears the short-conv history (it dies
    /// with the KV cache), but `prepareForContinuation` must PRESERVE it —
    /// the retained KV prefix and the conv history describe the same cached
    /// tokens (PR #4 review, Codex P1).
    @Test func convStateSurvivesContinuationButNotReset() throws {
        guard let path = ProcessInfo.processInfo
            .environment["MFERENCE_INKLING_GTURBO"] else { return }
        let ctx = try MetalContext()
        let cfg = try #require(ArchConfig.knownArchitectures[.inklingSmall])
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
                                   device: ctx.device,
                                   expecting: cfg)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        func fillSentinel() {
            for st in runner.inklingConvStates {
                for b in [st.k, st.v, st.attn, st.mlp] {
                    memset(b.contents(), 0x3C, b.length)
                }
            }
        }
        func firstWord() -> UInt32 {
            runner.inklingConvStates[0].k.contents().load(as: UInt32.self)
        }
        // reset() zeroes.
        fillSentinel()
        runner.reset()
        #expect(firstWord() == 0, "reset() must clear conv history")
        // continuation preserves.
        fillSentinel()
        runner.inklingParityAdvanceKV()   // kv.position = 1
        try runner.prepareForContinuation(expectedPosition: 1)
        #expect(firstWord() == 0x3C3C_3C3C,
                "prepareForContinuation must keep conv history")
    }
}
