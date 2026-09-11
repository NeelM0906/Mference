import Foundation
import Metal
import Testing
@testable import Mference

/// The GLM-5.3 kernels at the **production** shapes the toy cannot reach —
/// 32 KDA heads of 128, 64 attention heads over a 512-wide latent, 288
/// experts — on random data against scalar transcriptions of the same
/// formulas. The toy parity suites prove the semantics at head dim 64; these
/// prove the lane / register / threadgroup arithmetic at the real widths.
@Suite(.serialized) struct Glm53KernelShapeTests {

    private static func rand(_ n: Int, _ scale: Float, _ rng: inout SystemRandomNumberGenerator) -> [Float] {
        (0..<n).map { _ in Float.random(in: -1...1, using: &rng) * scale }
    }
    private static func halfBuffer(_ device: MTLDevice, _ v: [Float]) -> MTLBuffer {
        let b = device.makeBuffer(length: max(1, v.count) * 2, options: .storageModeShared)!
        let p = b.contents().bindMemory(to: Float16.self, capacity: v.count)
        for i in 0..<v.count { p[i] = Float16(v[i]) }
        return b
    }
    private static func floatBuffer(_ device: MTLDevice, _ v: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: v, length: max(1, v.count) * 4, options: .storageModeShared)!
    }
    private static func bf16View(_ device: MTLDevice, _ v: [Float]) -> TensorView {
        let bits = v.map { Quantization.bf16Bits($0) }
        let b = device.makeBuffer(bytes: bits, length: bits.count * 2, options: .storageModeShared)!
        return TensorView(buffer: b, offset: 0, length: UInt64(bits.count * 2),
                          scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                          shape: (UInt32(v.count), 0, 0, 0), dtype: 1)
    }
    private static func f32View(_ device: MTLDevice, _ v: [Float]) -> TensorView {
        let b = floatBuffer(device, v)
        return TensorView(buffer: b, offset: 0, length: UInt64(v.count * 4),
                          scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                          shape: (UInt32(v.count), 0, 0, 0), dtype: 3)
    }
    private static func f16(_ v: [Float]) -> [Float] { v.map { Float(Float16($0)) } }
    private static func bf16(_ v: [Float]) -> [Float] { v.map { Quantization.bf16ToFloat(Quantization.bf16Bits($0)) } }
    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
    private static func run(_ ctx: MetalContext, _ body: (MTLCommandBuffer) -> Void) throws {
        let cb = try #require(ctx.queue.makeCommandBuffer())
        body(cb)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.error == nil)
    }

    @Test func kdaDecodeAtThirtyTwoHeadsOf128MatchesTheRecurrence() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53Kernels(context: ctx)
        var rng = SystemRandomNumberGenerator()
        let H = 32, D = 128, qkv = H * D
        let lowerBound: Float = -5, eps: Float = 1e-5
        let aLog = Self.rand(H, 1, &rng), dtBias = Self.rand(qkv, 0.5, &rng)
        let oNorm = Self.bf16(Self.rand(D, 1, &rng).map { 1 + 0.2 * $0 })
        let aLogV = Self.f32View(ctx.device, aLog), dtBiasV = Self.f32View(ctx.device, dtBias)
        let oNormV = Self.bf16View(ctx.device, oNorm)
        let stateBuf = ctx.device.makeBuffer(length: H * D * D * 4, options: .storageModeShared)!
        memset(stateBuf.contents(), 0, stateBuf.length)
        var state = [Float](repeating: 0, count: H * D * D)
        let outBuf = ctx.device.makeBuffer(length: qkv * 2, options: .storageModeShared)!
        let yBuf = ctx.device.makeBuffer(length: qkv * 2, options: .storageModeShared)!

        var worstOut: Float = 0, worstState: Float = 0
        for _ in 0..<3 {
            let conv = Self.f16(Self.rand(3 * qkv, 1, &rng))
            let a = Self.f16(Self.rand(qkv, 1, &rng)), b = Self.f16(Self.rand(H, 2, &rng))
            let gate = Self.f16(Self.rand(qkv, 2, &rng))
            try Self.run(ctx) { cb in
                kernels.encodeKDADecode(commandBuffer: cb, convOut: Self.halfBuffer(ctx.device, conv),
                                        a: Self.halfBuffer(ctx.device, a), b: Self.halfBuffer(ctx.device, b),
                                        gate: Self.halfBuffer(ctx.device, gate),
                                        aLog: aLogV, dtBias: dtBiasV, oNorm: oNormV, state: stateBuf,
                                        out: outBuf, yOut: yBuf, heads: H, headDim: D,
                                        lowerBound: lowerBound, eps: eps)
            }
            // Reference: the oracle's arithmetic, per head.
            var expected = [Float](repeating: 0, count: qkv)
            for h in 0..<H {
                let base = h * D
                var q = Array(conv[base..<(base + D)])
                var k = Array(conv[(qkv + base)..<(qkv + base + D)])
                let v = Array(conv[(2 * qkv + base)..<(2 * qkv + base + D)])
                var qq: Float = 0, kk: Float = 0
                for d in 0..<D { qq += q[d] * q[d]; kk += k[d] * k[d] }
                let invQ = 1 / (qq + 1e-6).squareRoot() / Float(D).squareRoot()
                let invK = 1 / (kk + 1e-6).squareRoot()
                for d in 0..<D { q[d] *= invQ; k[d] *= invK }
                let beta = Self.sigmoid(b[h])
                var decay = [Float](repeating: 0, count: D)
                for d in 0..<D {
                    let g = expf(aLog[h]) * (a[base + d] + dtBias[base + d])
                    decay[d] = expf(lowerBound * Self.sigmoid(g))
                }
                var y = [Float](repeating: 0, count: D)
                for dv in 0..<D {
                    let row = (h * D + dv) * D
                    var kv: Float = 0
                    for dk in 0..<D { state[row + dk] *= decay[dk]; kv += state[row + dk] * k[dk] }
                    let delta = (v[dv] - kv) * beta
                    var yv: Float = 0
                    for dk in 0..<D { state[row + dk] += k[dk] * delta; yv += state[row + dk] * q[dk] }
                    y[dv] = yv
                }
                var ss: Float = 0
                for d in 0..<D { ss += y[d] * y[d] }
                let inv = 1 / (ss / Float(D) + eps).squareRoot()
                for d in 0..<D { expected[base + d] = y[d] * inv * oNorm[d] * Self.sigmoid(gate[base + d]) }
            }
            let got = Glm53ForwardRunner.readFP16(outBuf, count: qkv)
            for i in 0..<qkv {
                worstOut = max(worstOut, abs(got[i] - expected[i]) / max(1, abs(expected[i])))
            }
            let gpuState = stateBuf.contents().bindMemory(to: Float.self, capacity: H * D * D)
            for i in stride(from: 0, to: H * D * D, by: 97) {
                worstState = max(worstState, abs(gpuState[i] - state[i]) / max(1, abs(state[i])))
            }
        }
        #expect(worstOut < 5e-3, "KDA out worst rel delta \(worstOut)")
        #expect(worstState < 1e-4, "KDA state worst rel delta \(worstState)")
    }

    @Test func latentAttentionAt64HeadsOver512MatchesSoftmaxDenseAndSelected() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53Kernels(context: ctx)
        var rng = SystemRandomNumberGenerator()
        let H = 64, kv = 512, T = 300
        let scale: Float = 1 / Float(256).squareRoot()
        let q = Self.f16(Self.rand(H * kv, 0.3, &rng))
        let lat = Self.f16(Self.rand(T * kv, 1, &rng))
        let qBuf = Self.halfBuffer(ctx.device, q), latBuf = Self.halfBuffer(ctx.device, lat)
        let outBuf = ctx.device.makeBuffer(length: H * kv * 2, options: .storageModeShared)!
        func expected(_ rows: [Int]) -> [Float] {
            var out = [Float](repeating: 0, count: H * kv)
            for h in 0..<H {
                var s = rows.map { t -> Float in
                    var dot: Float = 0
                    for d in 0..<kv { dot += q[h * kv + d] * lat[t * kv + d] }
                    return dot * scale
                }
                let mx = s.max()!
                var sum: Float = 0
                for i in s.indices { s[i] = expf(s[i] - mx); sum += s[i] }
                for (i, t) in rows.enumerated() {
                    for d in 0..<kv { out[h * kv + d] += s[i] / sum * lat[t * kv + d] }
                }
            }
            return out
        }
        func check(_ rows: [Int]?, _ label: String) throws {
            let sel = ctx.device.makeBuffer(length: max(1, rows?.count ?? 0) * 4, options: .storageModeShared)!
            if let rows {
                let p = sel.contents().bindMemory(to: UInt32.self, capacity: rows.count)
                for (i, t) in rows.enumerated() { p[i] = UInt32(t) }
            }
            try Self.run(ctx) { cb in
                kernels.encodeLatentAttention(commandBuffer: cb, qLatent: qBuf, latents: latBuf, selected: sel,
                                              out: outBuf, heads: H, latentDim: kv, cachedRows: T,
                                              selectedCount: rows.map { UInt32($0.count) } ?? Glm53Kernels.attendAll,
                                              scale: scale)
            }
            let got = Glm53ForwardRunner.readFP16(outBuf, count: H * kv)
            let want = expected(rows ?? Array(0..<T))
            var worst: Float = 0
            for (a, b) in zip(got, want) { worst = max(worst, abs(a - b)) }
            #expect(worst < 3e-3, "\(label): worst abs delta \(worst)")
        }
        try check(nil, "dense over \(T)")
        try check(Array(stride(from: 0, to: T, by: 3)) + [T - 2, T - 1], "selected")
    }

    @Test func headedInt8GemvAt64HeadsMatchesTheDequantizedProduct() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53Kernels(context: ctx)
        var rng = SystemRandomNumberGenerator()
        let H = 64, M = 512, N = 256, groups = N / 64
        let weights = (0..<(H * M * N)).map { _ in UInt8.random(in: 0...255, using: &rng) }
        let scales = Self.bf16(Self.rand(H * M * groups, 0.02, &rng))
        let biases = Self.bf16(Self.rand(H * M * groups, 0.5, &rng))
        let x = Self.f16(Self.rand(H * N, 1, &rng))
        let wBytes = weights.count, sBytes = scales.count * 2
        let buf = ctx.device.makeBuffer(length: wBytes + 2 * sBytes, options: .storageModeShared)!
        buf.contents().copyMemory(from: weights, byteCount: wBytes)
        let sp = buf.contents().advanced(by: wBytes).bindMemory(to: UInt16.self, capacity: scales.count)
        let bp = buf.contents().advanced(by: wBytes + sBytes).bindMemory(to: UInt16.self, capacity: biases.count)
        for i in 0..<scales.count { sp[i] = Quantization.bf16Bits(scales[i]); bp[i] = Quantization.bf16Bits(biases[i]) }
        let view = TensorView(buffer: buf, offset: 0, length: UInt64(wBytes),
                              scaleOffset: UInt64(wBytes), scaleLength: UInt64(sBytes),
                              biasOffset: UInt64(wBytes + sBytes), biasLength: UInt64(sBytes),
                              shape: (UInt32(H), UInt32(M), UInt32(N), 0), dtype: 0)
        let yBuf = ctx.device.makeBuffer(length: H * M * 2, options: .storageModeShared)!
        try Self.run(ctx) { cb in
            kernels.encodeHeadedInt8GEMV(commandBuffer: cb, weights: view, x: Self.halfBuffer(ctx.device, x),
                                         y: yBuf, heads: H, m: M, n: N)
        }
        let got = Glm53ForwardRunner.readFP16(yBuf, count: H * M)
        var worst: Float = 0
        for h in 0..<H {
            for m in 0..<M {
                var acc: Float = 0
                for n in 0..<N {
                    let g = (h * M + m) * groups + n / 64
                    acc += (Float(weights[(h * M + m) * N + n]) * scales[g] + biases[g]) * x[h * N + n]
                }
                worst = max(worst, abs(got[h * M + m] - acc) / max(1, abs(acc)))
            }
        }
        #expect(worst < 5e-3, "headed INT8 GEMV worst rel delta \(worst)")
    }

    @Test func poolKeysAtFourPer128ChannelsMatchesTheSoftmaxPool() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53Kernels(context: ctx)
        var rng = SystemRandomNumberGenerator()
        let kp = 4, dim = 128, T = 12, pool = 2
        let keys = Self.f16(Self.rand(T * dim, 1, &rng)), gates = Self.f16(Self.rand(T * dim, 2, &rng))
        let ape = Self.bf16(Self.rand(kp * dim, 1, &rng))
        let pooled = ctx.device.makeBuffer(length: (T / kp) * dim * 2, options: .storageModeShared)!
        try Self.run(ctx) { cb in
            kernels.encodePoolKeys(commandBuffer: cb, keys: Self.halfBuffer(ctx.device, keys),
                                   gates: Self.halfBuffer(ctx.device, gates), ape: Self.bf16View(ctx.device, ape),
                                   pooled: pooled, pool: pool, kPool: kp, dim: dim)
        }
        let got = Glm53ForwardRunner.readFP16(pooled, offset: pool * dim * 2, count: dim)
        var worst: Float = 0
        for d in 0..<dim {
            var logits = (0..<kp).map { gates[(pool * kp + $0) * dim + d] + ape[$0 * dim + d] }
            let mx = logits.max()!
            var sum: Float = 0
            for c in 0..<kp { logits[c] = expf(logits[c] - mx); sum += logits[c] }
            var acc: Float = 0
            for c in 0..<kp { acc += logits[c] / sum * keys[(pool * kp + c) * dim + d] }
            worst = max(worst, abs(got[d] - acc))
        }
        #expect(worst < 2e-3, "pooled key worst abs delta \(worst)")
    }

    @Test func batchedInt8GemmMatchesTheDequantizedProduct() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53PrefillKernels(context: ctx, swigluLimit: 10)
        var rng = SystemRandomNumberGenerator()
        let M = 520, N = 1024, T = 37, groups = N / 64
        let weights = (0..<(M * N)).map { _ in UInt8.random(in: 0...255, using: &rng) }
        let scales = Self.bf16(Self.rand(M * groups, 0.02, &rng))
        let biases = Self.bf16(Self.rand(M * groups, 0.5, &rng))
        let x = Self.f16(Self.rand(T * N, 1, &rng))
        let wBytes = weights.count, sBytes = scales.count * 2
        let buf = ctx.device.makeBuffer(length: wBytes + 2 * sBytes, options: .storageModeShared)!
        buf.contents().copyMemory(from: weights, byteCount: wBytes)
        let sp = buf.contents().advanced(by: wBytes).bindMemory(to: UInt16.self, capacity: scales.count)
        let bp = buf.contents().advanced(by: wBytes + sBytes).bindMemory(to: UInt16.self, capacity: biases.count)
        for i in 0..<scales.count { sp[i] = Quantization.bf16Bits(scales[i]); bp[i] = Quantization.bf16Bits(biases[i]) }
        let view = TensorView(buffer: buf, offset: 0, length: UInt64(wBytes),
                              scaleOffset: UInt64(wBytes), scaleLength: UInt64(sBytes),
                              biasOffset: UInt64(wBytes + sBytes), biasLength: UInt64(sBytes),
                              shape: (UInt32(M), UInt32(N), 0, 0), dtype: 0)
        let yBuf = ctx.device.makeBuffer(length: T * M * 2, options: .storageModeShared)!
        let yScalar = ctx.device.makeBuffer(length: T * M * 2, options: .storageModeShared)!
        let xBuf = Self.halfBuffer(ctx.device, x)
        try Self.run(ctx) { cb in
            kernels.encodeInt8GEMM(commandBuffer: cb, weights: view, x: xBuf, xStride: N,
                                   y: yBuf, yStride: M, m: M, n: N, tokens: T, matrixUnits: true)
            kernels.encodeInt8GEMM(commandBuffer: cb, weights: view, x: xBuf, xStride: N,
                                   y: yScalar, yStride: M, m: M, n: N, tokens: T, matrixUnits: false)
        }
        let got = Glm53ForwardRunner.readFP16(yBuf, count: T * M)
        let gotScalar = Glm53ForwardRunner.readFP16(yScalar, count: T * M)
        var worst: Float = 0, worstScalar: Float = 0, worstAbs: Float = 0, worstAbsAt = (0, 0), magnitude: Float = 0
        for t in 0..<T {
            for m in 0..<M {
                var acc: Float = 0
                for n in 0..<N {
                    let g = m * groups + n / 64
                    acc += (Float(weights[m * N + n]) * scales[g] + biases[g]) * x[t * N + n]
                }
                magnitude = max(magnitude, abs(acc))
                let d = abs(got[t * M + m] - acc)
                if d > worstAbs { worstAbs = d; worstAbsAt = (t, m) }
                worst = max(worst, d / max(1, abs(acc)))
                worstScalar = max(worstScalar, abs(gotScalar[t * M + m] - acc) / max(1, abs(acc)))
            }
        }
        print(String(format: "  [glm53 shape] GEMM: scalar worst rel %.3e; mma worst rel %.3e, worst abs %.4f at (t %d, m %d), output magnitude %.2f",
                     worstScalar, worst, worstAbs, worstAbsAt.0, worstAbsAt.1, magnitude))
        #expect(worstScalar < 5e-3, "scalar batched INT8 GEMM worst rel delta \(worstScalar)")
        // The matrix-unit path rounds each dequantized weight to fp16 (about
        // 1/16 of an INT8 step) and, like the scalar path, the output to fp16
        // (ulp 0.0625 at magnitude 100), so judge it against the output scale.
        #expect(worstAbs <= 2e-3 * magnitude, "matrix-unit INT8 GEMM worst abs delta \(worstAbs) at magnitude \(magnitude)")
        _ = worst

        // BF16 GEMM (router / pooling gate), fp32 and fp16 outputs.
        let wb = Self.bf16(Self.rand(M * N, 0.05, &rng))
        let wbView = Self.bf16View(ctx.device, wb)
        let y32 = ctx.device.makeBuffer(length: T * M * 4, options: .storageModeShared)!
        try Self.run(ctx) { cb in
            kernels.encodeBF16GEMM(commandBuffer: cb, weights: wbView, x: Self.halfBuffer(ctx.device, x), xStride: N,
                                   y: y32, yStride: M, outputFloat32: true, m: M, n: N, tokens: T)
        }
        let g32 = y32.contents().bindMemory(to: Float.self, capacity: T * M)
        var worstB: Float = 0
        for t in stride(from: 0, to: T, by: 5) {
            for m in stride(from: 0, to: M, by: 7) {
                var acc: Float = 0
                for n in 0..<N { acc += wb[m * N + n] * x[t * N + n] }
                worstB = max(worstB, abs(g32[t * M + m] - acc) / max(1, abs(acc)))
            }
        }
        #expect(worstB < 2e-3, "batched BF16 GEMM worst rel delta \(worstB)")
    }

    /// The chunk kernel walks T tokens with the state in registers; per token
    /// it performs the decode kernel's arithmetic in the same order, so the
    /// outputs and the final state must be bit-identical to T decode steps.
    @Test func kdaChunkEqualsSequentialDecodeStepsBitForBit() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53Kernels(context: ctx)
        let prefill = try Glm53PrefillKernels(context: ctx, swigluLimit: 10)
        var rng = SystemRandomNumberGenerator()
        let H = 8, D = 128, qkv = H * D, T = 5
        let aLog = Self.f32View(ctx.device, Self.rand(H, 1, &rng))
        let dtBias = Self.f32View(ctx.device, Self.rand(qkv, 0.5, &rng))
        let oNorm = Self.bf16View(ctx.device, Self.rand(D, 1, &rng).map { 1 + 0.2 * $0 })
        let conv = Self.rand(T * 3 * qkv, 1, &rng), a = Self.rand(T * qkv, 1, &rng)
        let b = Self.rand(T * H, 2, &rng), gate = Self.rand(T * qkv, 2, &rng)
        let stateSeq = ctx.device.makeBuffer(length: H * D * D * 4, options: .storageModeShared)!
        let stateChunk = ctx.device.makeBuffer(length: H * D * D * 4, options: .storageModeShared)!
        let seed = Self.rand(H * D * D, 0.1, &rng)
        stateSeq.contents().copyMemory(from: seed, byteCount: seed.count * 4)
        stateChunk.contents().copyMemory(from: seed, byteCount: seed.count * 4)
        var stepOuts: [MTLBuffer] = []
        let yTmp = ctx.device.makeBuffer(length: qkv * 2, options: .storageModeShared)!
        for t in 0..<T {
            let out = ctx.device.makeBuffer(length: qkv * 2, options: .storageModeShared)!
            try Self.run(ctx) { cb in
                kernels.encodeKDADecode(
                    commandBuffer: cb,
                    convOut: Self.halfBuffer(ctx.device, Array(conv[(t * 3 * qkv)..<((t + 1) * 3 * qkv)])),
                    a: Self.halfBuffer(ctx.device, Array(a[(t * qkv)..<((t + 1) * qkv)])),
                    b: Self.halfBuffer(ctx.device, Array(b[(t * H)..<((t + 1) * H)])),
                    gate: Self.halfBuffer(ctx.device, Array(gate[(t * qkv)..<((t + 1) * qkv)])),
                    aLog: aLog, dtBias: dtBias, oNorm: oNorm, state: stateSeq,
                    out: out, yOut: yTmp, heads: H, headDim: D, lowerBound: -5, eps: 1e-5)
            }
            stepOuts.append(out)
        }
        let outChunk = ctx.device.makeBuffer(length: T * qkv * 2, options: .storageModeShared)!
        try Self.run(ctx) { cb in
            prefill.encodeKDAChunk(commandBuffer: cb, convOut: Self.halfBuffer(ctx.device, conv),
                                   a: Self.halfBuffer(ctx.device, a), b: Self.halfBuffer(ctx.device, b),
                                   gate: Self.halfBuffer(ctx.device, gate), aLog: aLog, dtBias: dtBias, oNorm: oNorm,
                                   state: stateChunk, out: outChunk, heads: H, headDim: D, tokens: T,
                                   lowerBound: -5, eps: 1e-5)
        }
        let chunk = Glm53ForwardRunner.readFP16(outChunk, count: T * qkv)
        var mismatches = 0
        for t in 0..<T {
            let seqRow = Glm53ForwardRunner.readFP16(stepOuts[t], count: qkv)
            let chunkRow = Array(chunk[(t * qkv)..<((t + 1) * qkv)])
            mismatches += zip(seqRow, chunkRow).filter { $0 != $1 }.count
        }
        #expect(mismatches == 0, "KDA chunk differs from sequential decode in \(mismatches) outputs")
        let s1 = stateSeq.contents().bindMemory(to: Float.self, capacity: H * D * D)
        let s2 = stateChunk.contents().bindMemory(to: Float.self, capacity: H * D * D)
        var stateMismatch = 0
        for i in 0..<(H * D * D) where s1[i] != s2[i] { stateMismatch += 1 }
        #expect(stateMismatch == 0, "KDA chunk final state differs in \(stateMismatch) entries")
    }

    @Test func routerSelectOver288ExpertsMatchesTheReferenceRule() throws {
        let ctx = try MetalContext()
        let kernels = try Glm53Kernels(context: ctx)
        var rng = SystemRandomNumberGenerator()
        let E = 288, K = 8
        let indices = ctx.device.makeBuffer(length: K * 4, options: .storageModeShared)!
        let weights = ctx.device.makeBuffer(length: K * 2, options: .storageModeShared)!
        for trial in 0..<20 {
            var logits = Self.rand(E, 3, &rng)
            let bias = Self.rand(E, 0.5, &rng)
            if trial % 4 == 0 {
                // Planted ties: equal biased keys must pick the lower index.
                logits[17] = logits[200]
                var b = bias; b[17] = b[200]
                try check(logits, b)
            } else {
                try check(logits, bias)
            }
        }
        func check(_ logits: [Float], _ bias: [Float]) throws {
            try Self.run(ctx) { cb in
                kernels.encodeRouterSelect(commandBuffer: cb, logits: Self.floatBuffer(ctx.device, logits),
                                           bias: Self.f32View(ctx.device, bias), outIndices: indices,
                                           outWeights: weights, numExperts: E, routeScale: 2.5)
            }
            let scores = logits.map { Self.sigmoid($0) }
            let biased = (0..<E).map { scores[$0] + bias[$0] }
            let order = (0..<E).sorted { biased[$0] != biased[$1] ? biased[$0] > biased[$1] : $0 < $1 }
            let chosen = Array(order.prefix(K))
            var sum: Float = 0
            for e in chosen { sum += scores[e] }
            let ip = indices.contents().bindMemory(to: UInt32.self, capacity: K)
            #expect((0..<K).map { Int(ip[$0]) } == chosen)
            let got = Glm53ForwardRunner.readFP16(weights, count: K)
            for (i, e) in chosen.enumerated() {
                #expect(abs(got[i] - scores[e] / sum * 2.5) < 2e-3, "weight \(i)")
            }
        }
    }
}
