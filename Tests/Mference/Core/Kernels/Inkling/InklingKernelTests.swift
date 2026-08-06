import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// CPU-reference parity tests for the Inkling family kernels. The references
/// implement the semantics transcribed from the checkpoint's `inkling_mlx`
/// package (docs/INKLING_SMALL.md "Forward-pass contract"); every kernel is
/// validated on random data before it touches real weights.
@Suite struct InklingKernelTests {

    private static func makeKernels(_ context: MetalContext) throws -> InklingKernels {
        try InklingKernels(context: context, numRouted: 256, numShared: 2)
    }

    private static func randomHalf(_ n: Int, _ rng: inout SplitMix64,
                                   scale: Float = 1.0) -> [Float16] {
        (0..<n).map { _ in Float16(rng.uniform(0, 1) * 2 * scale - scale) }
    }

    private static func bf16(_ v: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: v.bitPattern >> 16)
    }
    private static func bf16ToFloat(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
        values.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                              options: .storageModeShared)!
        }
    }

    private static func run(_ context: MetalContext,
                            _ body: (MTLCommandBuffer) -> Void) {
        let cb = context.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    // MARK: - Short convolution

    /// Streams a random sequence one step at a time through the kernel and
    /// compares each step against a full causal depthwise conv (+ residual)
    /// computed on the CPU — this covers both the tap math and the state
    /// shifting across steps.
    @Test func sconvStepMatchesCausalConvAcrossSteps() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0x11C_0001)
        let C = 96, K = 4, steps = 7

        let wF: [Float] = (0..<C * K).map { _ in rng.uniform(0, 1) - 0.5 }
        let wBuf = Self.buffer(context.device, wF.map(Self.bf16))
        let stateBuf = context.device.makeBuffer(
            length: C * (K - 1) * MemoryLayout<Float>.size,
            options: .storageModeShared)!
        memset(stateBuf.contents(), 0, stateBuf.length)

        let xs: [[Float16]] = (0..<steps).map { _ in Self.randomHalf(C, &rng) }
        let outBuf = context.device.makeBuffer(
            length: C * MemoryLayout<Float16>.size, options: .storageModeShared)!

        for t in 0..<steps {
            let xBuf = Self.buffer(context.device, xs[t])
            Self.run(context) { cb in
                kernels.encodeSconvStep(commandBuffer: cb,
                                        x: xBuf, state: stateBuf,
                                        weight: wBuf, weightOffset: 0,
                                        out: outBuf,
                                        channels: UInt32(C), kernelSize: UInt32(K))
            }
            let got = outBuf.contents().bindMemory(to: Float16.self, capacity: C)
            for c in 0..<C {
                // Reference: causal conv over the sequence with zero left pad,
                // taps applied oldest-input-first, plus the residual.
                var acc = Float(0)
                for j in 0..<K {
                    let src = t - (K - 1) + j
                    let xv = src >= 0 ? Float(xs[src][c]) : 0
                    acc += Self.bf16ToFloat(Self.bf16(wF[c * K + j])) * xv
                }
                let expected = acc + Float(xs[t][c])
                #expect(abs(Float(got[c]) - expected) < 2e-2,
                        "step \(t) channel \(c): got \(Float(got[c])), want \(expected)")
            }
        }
    }

    @Test func sconvPrefillMatchesDecodeStepsAndFusedResidual() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0x11C_BA7C)
        let C = 96, T = 7, K = 4
        let x = Self.randomHalf(T * C, &rng, scale: 2)
        let w: [UInt16] = (0..<C * K).map { _ in
            Self.bf16(rng.uniform(0, 1) - 0.5)
        }
        let xBuf = Self.buffer(context.device, x)
        let wBuf = Self.buffer(context.device, w)
        func state() -> MTLBuffer {
            let result = context.device.makeBuffer(
                length: C * (K - 1) * MemoryLayout<Float>.size,
                options: .storageModeShared)!
            memset(result.contents(), 0, result.length)
            return result
        }

        let scalarState = state()
        let scalarOut = context.device.makeBuffer(
            length: T * C * MemoryLayout<Float16>.size,
            options: .storageModeShared)!
        Self.run(context) { cb in
            for t in 0..<T {
                kernels.encodeSconvStep(
                    commandBuffer: cb,
                    x: xBuf, xOffset: t * C * 2,
                    state: scalarState,
                    weight: wBuf, weightOffset: 0,
                    out: scalarOut, outOffset: t * C * 2,
                    channels: UInt32(C), kernelSize: UInt32(K))
            }
        }

        let batchState = state()
        let batchOut = context.device.makeBuffer(
            length: scalarOut.length, options: .storageModeShared)!
        Self.run(context) { cb in
            kernels.encodeSconvPrefill(
                commandBuffer: cb,
                x: xBuf, state: batchState,
                weight: wBuf, weightOffset: 0,
                out: batchOut,
                channels: UInt32(C), rows: UInt32(T))
        }
        let scalar = scalarOut.contents().bindMemory(
            to: Float16.self, capacity: T * C)
        let batch = batchOut.contents().bindMemory(
            to: Float16.self, capacity: T * C)
        for i in 0..<T * C { #expect(batch[i] == scalar[i], "f16 output \(i)") }
        #expect(memcmp(batchState.contents(), scalarState.contents(), batchState.length) == 0)

        let initialHidden: [Float] = (0..<T * C).map { _ in
            rng.uniform(0, 1) * 10 - 5
        }
        let scalarHidden = Self.buffer(context.device, initialHidden)
        let batchHidden = Self.buffer(context.device, initialHidden)
        let scalarResidualState = state()
        let batchResidualState = state()
        let delta = context.device.makeBuffer(
            length: C * MemoryLayout<Float>.size,
            options: .storageModeShared)!
        let rowBuffers = (0..<T).map { t in
            Self.buffer(context.device, Array(x[(t * C)..<((t + 1) * C)]))
        }
        Self.run(context) { cb in
            for t in 0..<T {
                kernels.encodeSconvStepF32Out(
                    commandBuffer: cb,
                    x: rowBuffers[t], state: scalarResidualState,
                    weight: wBuf, weightOffset: 0,
                    out: delta,
                    channels: UInt32(C), kernelSize: UInt32(K))
                kernels.encodeResidualAddF32Delta(
                    commandBuffer: cb,
                    hidden: scalarHidden,
                    hiddenOffset: t * C * MemoryLayout<Float>.size,
                    delta: delta,
                    count: UInt32(C), scale: 32)
            }
        }
        Self.run(context) { cb in
            kernels.encodeSconvPrefillResidual(
                commandBuffer: cb,
                x: xBuf, state: batchResidualState,
                weight: wBuf, weightOffset: 0,
                hidden: batchHidden,
                channels: UInt32(C), rows: UInt32(T), scale: 32)
        }
        let scalarF32 = scalarHidden.contents().bindMemory(to: Float.self, capacity: T * C)
        let batchF32 = batchHidden.contents().bindMemory(to: Float.self, capacity: T * C)
        for i in 0..<T * C { #expect(batchF32[i] == scalarF32[i], "f32 residual \(i)") }
        #expect(memcmp(batchResidualState.contents(), scalarResidualState.contents(),
                       batchResidualState.length) == 0)
    }

    // MARK: - Q/K per-head RMS norm

    @Test func qkNormMatchesReference() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0xBEEF_0002)
        let HD = 128, NQ = 4, NKV = 2
        let eps: Float = 1e-6

        let q = Self.randomHalf(NQ * HD, &rng, scale: 2)
        let k = Self.randomHalf(NKV * HD, &rng, scale: 2)
        let qw: [Float] = (0..<HD).map { _ in rng.uniform(0, 1) + 0.5 }
        let kw: [Float] = (0..<HD).map { _ in rng.uniform(0, 1) + 0.5 }

        let qBuf = Self.buffer(context.device, q)
        let kBuf = Self.buffer(context.device, k)
        Self.run(context) { cb in
            kernels.encodeQKNorm(commandBuffer: cb,
                                 q: qBuf, k: kBuf, kOffset: 0,
                                 qWeight: Self.buffer(context.device, qw.map(Self.bf16)),
                                 qWeightOffset: 0,
                                 kWeight: Self.buffer(context.device, kw.map(Self.bf16)),
                                 kWeightOffset: 0,
                                 headDim: UInt32(HD), numQHeads: UInt32(NQ),
                                 numKVHeads: UInt32(NKV), eps: eps)
        }

        func check(_ buf: MTLBuffer, _ src: [Float16], _ w: [Float], heads: Int, label: String) {
            let got = buf.contents().bindMemory(to: Float16.self, capacity: heads * HD)
            for h in 0..<heads {
                var ss = Float(0)
                for i in 0..<HD { ss += Float(src[h * HD + i]) * Float(src[h * HD + i]) }
                let inv = 1.0 / (ss / Float(HD) + eps).squareRoot()
                for i in 0..<HD {
                    let expected = Float(src[h * HD + i]) * inv
                        * Self.bf16ToFloat(Self.bf16(w[i]))
                    #expect(abs(Float(got[h * HD + i]) - expected) < 1.5e-2,
                            "\(label) head \(h) elem \(i)")
                }
            }
        }
        check(qBuf, q, qw, heads: NQ, label: "q")
        check(kBuf, k, kw, heads: NKV, label: "k")
    }

    @Test func qkNormPrefillMatchesDecodeRows() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0xBEEF_BA7C)
        let T = 5, HD = 128, NQ = 4, NKV = 2
        let q = Self.randomHalf(T * NQ * HD, &rng, scale: 2)
        let k = Self.randomHalf(T * NKV * HD, &rng, scale: 2)
        let qw = Self.buffer(context.device, (0..<HD).map { _ in
            Self.bf16(rng.uniform(0, 1) + 0.5)
        })
        let kw = Self.buffer(context.device, (0..<HD).map { _ in
            Self.bf16(rng.uniform(0, 1) + 0.5)
        })
        let batchQ = Self.buffer(context.device, q)
        let batchK = Self.buffer(context.device, k)
        let scalarQ = (0..<T).map { t in
            Self.buffer(context.device,
                        Array(q[(t * NQ * HD)..<((t + 1) * NQ * HD)]))
        }
        let scalarK = (0..<T).map { t in
            Self.buffer(context.device,
                        Array(k[(t * NKV * HD)..<((t + 1) * NKV * HD)]))
        }
        Self.run(context) { cb in
            for t in 0..<T {
                kernels.encodeQKNorm(
                    commandBuffer: cb,
                    q: scalarQ[t], k: scalarK[t], kOffset: 0,
                    qWeight: qw, qWeightOffset: 0,
                    kWeight: kw, kWeightOffset: 0,
                    headDim: UInt32(HD), numQHeads: UInt32(NQ),
                    numKVHeads: UInt32(NKV), eps: 1e-6)
            }
        }
        Self.run(context) { cb in
            kernels.encodeQKNormPrefill(
                commandBuffer: cb,
                q: batchQ, k: batchK,
                qWeight: qw, qWeightOffset: 0,
                kWeight: kw, kWeightOffset: 0,
                headDim: UInt32(HD), numQHeads: UInt32(NQ),
                numKVHeads: UInt32(NKV), rows: UInt32(T), eps: 1e-6)
        }
        let gotQ = batchQ.contents().bindMemory(to: Float16.self, capacity: q.count)
        let gotK = batchK.contents().bindMemory(to: Float16.self, capacity: k.count)
        for t in 0..<T {
            let wantQ = scalarQ[t].contents().bindMemory(to: Float16.self,
                                                        capacity: NQ * HD)
            let wantK = scalarK[t].contents().bindMemory(to: Float16.self,
                                                        capacity: NKV * HD)
            for i in 0..<NQ * HD { #expect(gotQ[t * NQ * HD + i] == wantQ[i]) }
            for i in 0..<NKV * HD { #expect(gotK[t * NKV * HD + i] == wantK[i]) }
        }
    }

    // MARK: - Attention with relative-position bias

    private static func attentionReference(
        q: [Float16], k: [Float16], v: [Float16], rel: [Float16], proj: [Float],
        HD: Int, NQ: Int, NKV: Int, seqLen: Int, kvStart: Int,
        relExtent: Int, dRel: Int, ringCap: Int,
        scale: Float, tau: Float) -> [Float] {
        var out = [Float](repeating: 0, count: NQ * HD)
        let qPos = seqLen - 1
        for h in 0..<NQ {
            let kvHead = h / (NQ / NKV)
            var logits: [Float] = []
            for p in kvStart..<seqLen {
                let phys = ringCap > 0 ? p % ringCap : p
                var qk = Float(0)
                for i in 0..<HD {
                    qk += Float(q[h * HD + i]) * Float(k[(phys * NKV + kvHead) * HD + i])
                }
                var bias = Float(0)
                let dist = qPos - p
                if dist < relExtent {
                    for i in 0..<dRel {
                        bias += Float(rel[h * dRel + i])
                            * Self.bf16ToFloat(Self.bf16(proj[i * relExtent + dist]))
                    }
                }
                logits.append(tau * (qk * scale + bias))
            }
            let mx = logits.max() ?? 0
            let exps = logits.map { expf($0 - mx) }
            let denom = exps.reduce(0, +)
            for (idx, p) in (kvStart..<seqLen).enumerated() {
                let phys = ringCap > 0 ? p % ringCap : p
                let w = exps[idx] / denom
                for i in 0..<HD {
                    out[h * HD + i] += w * Float(v[(phys * NKV + kvHead) * HD + i])
                }
            }
        }
        return out
    }

    @Test(arguments: [
        // (seqLen, kvStart, relExtent, ringCap, tau) — full layer, short ctx
        (9, 0, 16, 0, Float(1.0)),
        // full layer with log-scaling active
        (12, 0, 16, 0, Float(1.37)),
        // sliding layer with ring wrap: window 8 over 21 tokens, ring cap 8
        (21, 13, 8, 8, Float(1.0)),
    ] as [(Int, Int, Int, Int, Float)])
    func attentionDecodeMatchesReference(
        _ arg: (seqLen: Int, kvStart: Int, relExtent: Int, ringCap: Int, tau: Float)
    ) throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0xA77E_0003 &+ UInt64(arg.seqLen))
        let HD = 64, NQ = 4, NKV = 2, dRel = 16
        let scale = 1.0 / Float(HD)
        let slots = arg.ringCap > 0 ? arg.ringCap : arg.seqLen

        let q = Self.randomHalf(NQ * HD, &rng)
        let k = Self.randomHalf(slots * NKV * HD, &rng)
        let v = Self.randomHalf(slots * NKV * HD, &rng)
        let rel = Self.randomHalf(NQ * dRel, &rng)
        let proj: [Float] = (0..<dRel * arg.relExtent).map { _ in
            rng.uniform(0, 1) - 0.5
        }

        let outBuf = context.device.makeBuffer(
            length: NQ * HD * MemoryLayout<Float16>.size, options: .storageModeShared)!
        Self.run(context) { cb in
            kernels.encodeAttentionDecode(
                commandBuffer: cb,
                q: Self.buffer(context.device, q),
                k: Self.buffer(context.device, k),
                v: Self.buffer(context.device, v),
                rel: Self.buffer(context.device, rel),
                proj: Self.buffer(context.device, proj.map(Self.bf16)), projOffset: 0,
                out: outBuf,
                headDim: UInt32(HD), numQHeads: UInt32(NQ), numKVHeads: UInt32(NKV),
                seqLen: UInt32(arg.seqLen), kvStart: UInt32(arg.kvStart),
                relExtent: UInt32(arg.relExtent), dRel: UInt32(dRel),
                ringCapacity: UInt32(arg.ringCap),
                scale: scale, tau: arg.tau)
        }

        let expected = Self.attentionReference(
            q: q, k: k, v: v, rel: rel, proj: proj,
            HD: HD, NQ: NQ, NKV: NKV, seqLen: arg.seqLen, kvStart: arg.kvStart,
            relExtent: arg.relExtent, dRel: dRel, ringCap: arg.ringCap,
            scale: scale, tau: arg.tau)
        let got = outBuf.contents().bindMemory(to: Float16.self, capacity: NQ * HD)
        for i in 0..<NQ * HD {
            #expect(abs(Float(got[i]) - expected[i]) < 2e-2,
                    "elem \(i): got \(Float(got[i])), want \(expected[i])")
        }
    }

    @Test(arguments: [
        (3, 5, 0, 0, 5, Float(0.1)),
        (7, 5, 6, 11, 0, Float(0)),
    ] as [(Int, Int, Int, Int, Int, Float)])
    func attentionPrefillMatchesDecodeRows(
        _ arg: (start: Int, rows: Int, window: Int, ring: Int,
                logFloor: Int, logAlpha: Float)
    ) throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0xA77E_BA7C &+ UInt64(arg.start))
        let HD = 64, NQ = 4, NKV = 2, dRel = 16
        let end = arg.start + arg.rows
        let slots = arg.ring > 0 ? arg.ring : end
        let extent = arg.window > 0 ? arg.window : 16
        let q = Self.randomHalf(arg.rows * NQ * HD, &rng)
        let rel = Self.randomHalf(arg.rows * NQ * dRel, &rng)
        let logicalK = Self.randomHalf(end * NKV * HD, &rng)
        let logicalV = Self.randomHalf(end * NKV * HD, &rng)
        var physicalK = [Float16](repeating: 0, count: slots * NKV * HD)
        var physicalV = [Float16](repeating: 0, count: slots * NKV * HD)
        let rowWidth = NKV * HD
        for p in 0..<end {
            let slot = arg.ring > 0 ? p % arg.ring : p
            physicalK.replaceSubrange((slot * rowWidth)..<((slot + 1) * rowWidth),
                                      with: logicalK[(p * rowWidth)..<((p + 1) * rowWidth)])
            physicalV.replaceSubrange((slot * rowWidth)..<((slot + 1) * rowWidth),
                                      with: logicalV[(p * rowWidth)..<((p + 1) * rowWidth)])
        }
        let qBuf = Self.buffer(context.device, q)
        let kBuf = Self.buffer(context.device, physicalK)
        let vBuf = Self.buffer(context.device, physicalV)
        let relBuf = Self.buffer(context.device, rel)
        let proj = Self.buffer(context.device, (0..<dRel * extent).map { _ in
            Self.bf16(rng.uniform(0, 1) - 0.5)
        })
        let batchOut = context.device.makeBuffer(
            length: arg.rows * NQ * HD * MemoryLayout<Float16>.size,
            options: .storageModeShared)!
        Self.run(context) { cb in
            kernels.encodeAttentionPrefill(
                commandBuffer: cb,
                q: qBuf, k: kBuf, v: vBuf, rel: relBuf,
                proj: proj, projOffset: 0, out: batchOut,
                headDim: UInt32(HD), numQHeads: UInt32(NQ),
                numKVHeads: UInt32(NKV),
                startPosition: UInt32(arg.start), rows: UInt32(arg.rows),
                slidingWindow: UInt32(arg.window), relExtent: UInt32(extent),
                dRel: UInt32(dRel), ringCapacity: UInt32(arg.ring),
                logScalingFloor: UInt32(arg.logFloor),
                scale: 1 / Float(HD), logScalingAlpha: arg.logAlpha,
                preferTensorOps: false)
        }

        let got = batchOut.contents().bindMemory(
            to: Float16.self, capacity: arg.rows * NQ * HD)
        for t in 0..<arg.rows {
            let qRow = Self.buffer(context.device,
                Array(q[(t * NQ * HD)..<((t + 1) * NQ * HD)]))
            let relRow = Self.buffer(context.device,
                Array(rel[(t * NQ * dRel)..<((t + 1) * NQ * dRel)]))
            let scalarOut = context.device.makeBuffer(
                length: NQ * HD * MemoryLayout<Float16>.size,
                options: .storageModeShared)!
            let seqLen = arg.start + t + 1
            let kvStart = arg.window > 0 ? max(0, seqLen - arg.window) : 0
            let activeRing = arg.ring > 0 && seqLen > arg.ring ? arg.ring : 0
            let tau: Float = arg.logFloor > 0 && seqLen > arg.logFloor
                ? 1 + arg.logAlpha * log(Float(seqLen) / Float(arg.logFloor)) : 1
            Self.run(context) { cb in
                kernels.encodeAttentionDecode(
                    commandBuffer: cb,
                    q: qRow, k: kBuf, v: vBuf, rel: relRow,
                    proj: proj, projOffset: 0, out: scalarOut,
                    headDim: UInt32(HD), numQHeads: UInt32(NQ),
                    numKVHeads: UInt32(NKV), seqLen: UInt32(seqLen),
                    kvStart: UInt32(kvStart), relExtent: UInt32(extent),
                    dRel: UInt32(dRel), ringCapacity: UInt32(activeRing),
                    scale: 1 / Float(HD), tau: tau)
            }
            let want = scalarOut.contents().bindMemory(to: Float16.self,
                                                       capacity: NQ * HD)
            for i in 0..<NQ * HD {
                #expect(got[t * NQ * HD + i] == want[i], "token \(t) elem \(i)")
            }
        }
    }

    @Test func attentionTensorOpsMatchesPortableInklingShape() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        guard kernels.tensorOpsPrefillAttentionAvailable else { return }
        var rng = SplitMix64(seed: 0xA77E_7E05)
        let HD = 128, NQ = 32, NKV = 8, dRel = 16
        let start = 513, rows = 13, end = start + rows, extent = 64
        let q = Self.buffer(context.device,
                            Self.randomHalf(rows * NQ * HD, &rng))
        let k = Self.buffer(context.device,
                            Self.randomHalf(end * NKV * HD, &rng))
        let v = Self.buffer(context.device,
                            Self.randomHalf(end * NKV * HD, &rng))
        let rel = Self.buffer(context.device,
                              Self.randomHalf(rows * NQ * dRel, &rng))
        let proj = Self.buffer(context.device, (0..<dRel * extent).map { _ in
            Self.bf16(rng.uniform(0, 1) - 0.5)
        })
        let byteCount = rows * NQ * HD * MemoryLayout<Float16>.size
        let portable = context.device.makeBuffer(
            length: byteCount, options: .storageModeShared)!
        let tensorOps = context.device.makeBuffer(
            length: byteCount, options: .storageModeShared)!

        func encode(_ output: MTLBuffer, preferTensorOps: Bool) {
            Self.run(context) { cb in
                kernels.encodeAttentionPrefill(
                    commandBuffer: cb,
                    q: q, k: k, v: v, rel: rel,
                    proj: proj, projOffset: 0, out: output,
                    headDim: UInt32(HD), numQHeads: UInt32(NQ),
                    numKVHeads: UInt32(NKV),
                    startPosition: UInt32(start), rows: UInt32(rows),
                    slidingWindow: 0, relExtent: UInt32(extent),
                    dRel: UInt32(dRel), ringCapacity: 0,
                    logScalingFloor: 8,
                    scale: 1 / Float(HD), logScalingAlpha: 0.1,
                    preferTensorOps: preferTensorOps)
            }
        }
        encode(portable, preferTensorOps: false)
        encode(tensorOps, preferTensorOps: true)

        let expected = portable.contents().bindMemory(
            to: Float16.self, capacity: rows * NQ * HD)
        let actual = tensorOps.contents().bindMemory(
            to: Float16.self, capacity: rows * NQ * HD)
        var maxDifference: Float = 0
        for i in 0..<rows * NQ * HD {
            maxDifference = max(
                maxDifference, abs(Float(actual[i]) - Float(expected[i])))
        }
        #expect(maxDifference <= 0.04,
                "TensorOps relative attention max difference \(maxDifference)")
    }

    @Test func attentionTensorOpsMatchesPortableAcrossRingWrap() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        guard kernels.tensorOpsPrefillAttentionAvailable else { return }
        var rng = SplitMix64(seed: 0xA77E_7106)
        let HD = 128, NQ = 32, NKV = 8, dRel = 16
        let start = 700, rows = 13, end = start + rows
        let window = 512, ring = 640, extent = window
        let q = Self.buffer(context.device,
                            Self.randomHalf(rows * NQ * HD, &rng))
        let relative = Self.buffer(context.device,
                                   Self.randomHalf(rows * NQ * dRel, &rng))
        let rowWidth = NKV * HD
        let logicalK = Self.randomHalf(end * rowWidth, &rng)
        let logicalV = Self.randomHalf(end * rowWidth, &rng)
        var physicalK = [Float16](repeating: 0, count: ring * rowWidth)
        var physicalV = [Float16](repeating: 0, count: ring * rowWidth)
        for position in 0..<end {
            let slot = position % ring
            physicalK.replaceSubrange(
                (slot * rowWidth)..<((slot + 1) * rowWidth),
                with: logicalK[(position * rowWidth)..<((position + 1) * rowWidth)])
            physicalV.replaceSubrange(
                (slot * rowWidth)..<((slot + 1) * rowWidth),
                with: logicalV[(position * rowWidth)..<((position + 1) * rowWidth)])
        }
        let k = Self.buffer(context.device, physicalK)
        let v = Self.buffer(context.device, physicalV)
        let proj = Self.buffer(context.device, (0..<dRel * extent).map { _ in
            Self.bf16(rng.uniform(0, 1) - 0.5)
        })
        let byteCount = rows * NQ * HD * MemoryLayout<Float16>.size
        let portable = context.device.makeBuffer(
            length: byteCount, options: .storageModeShared)!
        let tensorOps = context.device.makeBuffer(
            length: byteCount, options: .storageModeShared)!

        func encode(_ output: MTLBuffer, preferTensorOps: Bool) {
            Self.run(context) { cb in
                kernels.encodeAttentionPrefill(
                    commandBuffer: cb,
                    q: q, k: k, v: v, rel: relative,
                    proj: proj, projOffset: 0, out: output,
                    headDim: UInt32(HD), numQHeads: UInt32(NQ),
                    numKVHeads: UInt32(NKV),
                    startPosition: UInt32(start), rows: UInt32(rows),
                    slidingWindow: UInt32(window), relExtent: UInt32(extent),
                    dRel: UInt32(dRel), ringCapacity: UInt32(ring),
                    logScalingFloor: 128_000,
                    scale: 1 / Float(HD), logScalingAlpha: 0.1,
                    preferTensorOps: preferTensorOps)
            }
        }
        encode(portable, preferTensorOps: false)
        encode(tensorOps, preferTensorOps: true)

        let expected = portable.contents().bindMemory(
            to: Float16.self, capacity: rows * NQ * HD)
        let actual = tensorOps.contents().bindMemory(
            to: Float16.self, capacity: rows * NQ * HD)
        var maxDifference: Float = 0
        for i in 0..<rows * NQ * HD {
            maxDifference = max(
                maxDifference, abs(Float(actual[i]) - Float(expected[i])))
        }
        #expect(maxDifference <= 0.04,
                "ring TensorOps relative attention max difference \(maxDifference)")
    }

    // MARK: - Sigmoid router

    @Test func routerSelectMatchesReference() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0x60D_0004)
        let nRouted = 256, nShared = 2, topK = 6
        let routeScale: Float = 8.0
        let globalScale: Float = 1.25

        for trial in 0..<8 {
            let logits: [Float] = (0..<nRouted + nShared).map { _ in
                rng.uniform(0, 1) * 8 - 4
            }
            let bias: [Float] = (0..<nRouted).map { _ in
                rng.uniform(0, 1) * 0.5 - 0.25
            }
            // GEMV is exercised separately; feed logits directly by writing
            // into the kernel's logits buffer and encoding only the select.
            kernels.routerLogits.contents().copyMemory(
                from: logits, byteCount: logits.count * 4)

            let outIdx = context.device.makeBuffer(length: topK * 4,
                                                   options: .storageModeShared)!
            let outW = context.device.makeBuffer(length: topK * 2,
                                                 options: .storageModeShared)!
            Self.run(context) { cb in
                kernels.encodeRouterSelectOnly(
                    commandBuffer: cb,
                    gateBias: Self.buffer(context.device, bias), gateBiasOffset: 0,
                    globalScale: Self.buffer(context.device, [globalScale]),
                    globalScaleOffset: 0,
                    outIndices: outIdx, outWeights: outW,
                    numRouted: UInt32(nRouted), numShared: UInt32(nShared),
                    topK: UInt32(topK), routeScale: routeScale)
            }

            // CPU reference.
            let scores = (0..<nRouted).map { 1 / (1 + expf(-logits[$0])) + bias[$0] }
            let refIdx = scores.enumerated()
                .sorted { $0.element > $1.element }.prefix(topK).map(\.offset)
            func logsigmoid(_ x: Float) -> Float { min(x, 0) - log1pf(expf(-abs(x))) }
            let sel = refIdx.map { logsigmoid(logits[$0]) }
                + (0..<nShared).map { logsigmoid(logits[nRouted + $0]) }
            let mx = sel.max()!
            let exps = sel.map { expf($0 - mx) }
            let denom = exps.reduce(0, +)
            let refWeights = exps.map { $0 / denom * routeScale * globalScale }

            let gotIdx = outIdx.contents().bindMemory(to: UInt32.self, capacity: topK)
            let gotW = outW.contents().bindMemory(to: Float16.self, capacity: topK)
            let gotGammas = kernels.sharedGammas.contents()
                .bindMemory(to: Float.self, capacity: nShared)

            #expect(Set((0..<topK).map { Int(gotIdx[$0]) }) == Set(refIdx),
                    "trial \(trial) expert set")
            // Weights follow the kernel's index order; compare via lookup.
            var refByIdx: [Int: Float] = [:]
            for (i, e) in refIdx.enumerated() { refByIdx[e] = refWeights[i] }
            for i in 0..<topK {
                let e = Int(gotIdx[i])
                #expect(abs(Float(gotW[i]) - (refByIdx[e] ?? -1)) < 1e-2,
                        "trial \(trial) weight for expert \(e)")
            }
            for s in 0..<nShared {
                #expect(abs(gotGammas[s] - refWeights[topK + s]) < 1e-3,
                        "trial \(trial) gamma \(s)")
            }
        }
    }

    @Test func routerPrefillMatchesDecodeRows() throws {
        let context = try MetalContext()
        let kernels = try Self.makeKernels(context)
        var rng = SplitMix64(seed: 0x60D_BA7C)
        let T = 5, D = 64, routed = 32, shared = 2, topK = 6
        let weights = Self.buffer(context.device, (0..<(routed + shared) * D).map { _ in
            Self.bf16(rng.uniform(0, 1) - 0.5)
        })
        let hidden = Self.buffer(context.device, Self.randomHalf(T * D, &rng, scale: 2))
        let ones = Self.buffer(context.device, [UInt16](repeating: 0x3F80, count: D))
        let bias = Self.buffer(context.device, (0..<routed).map { _ in
            rng.uniform(0, 1) * 0.5 - 0.25
        })
        let global = Self.buffer(context.device, [Float(1.25)])
        let scalarIdx = context.device.makeBuffer(length: T * 8 * 4,
                                                   options: .storageModeShared)!
        let scalarW = context.device.makeBuffer(length: T * 8 * 2,
                                                 options: .storageModeShared)!
        let scalarGamma = context.device.makeBuffer(length: T * shared * 4,
                                                     options: .storageModeShared)!
        Self.run(context) { cb in
            for t in 0..<T {
                kernels.encodeRouter(
                    commandBuffer: cb,
                    weights: weights, weightsOffset: 0,
                    hidden: hidden, hiddenOffset: t * D * 2,
                    onesScale: ones,
                    gateBias: bias, gateBiasOffset: 0,
                    globalScale: global, globalScaleOffset: 0,
                    outIndices: scalarIdx, outIndicesOffset: t * 8 * 4,
                    outWeights: scalarW, outWeightsOffset: t * 8 * 2,
                    gammasOut: scalarGamma, gammasOffset: t * shared * 4,
                    numRouted: UInt32(routed), numShared: UInt32(shared),
                    topK: UInt32(topK), routeScale: 8, d: UInt32(D))
            }
        }
        let batchIdx = context.device.makeBuffer(length: T * 8 * 4,
                                                  options: .storageModeShared)!
        let batchW = context.device.makeBuffer(length: T * 8 * 2,
                                                options: .storageModeShared)!
        let batchGamma = context.device.makeBuffer(length: T * shared * 4,
                                                    options: .storageModeShared)!
        let logits = context.device.makeBuffer(length: T * (routed + shared) * 4,
                                                options: .storageModeShared)!
        Self.run(context) { cb in
            kernels.encodeRouterPrefill(
                commandBuffer: cb,
                weights: weights, weightsOffset: 0,
                hidden: hidden, effectiveScale: ones, logits: logits,
                gateBias: bias, gateBiasOffset: 0,
                globalScale: global, globalScaleOffset: 0,
                outIndices: batchIdx, outWeights: batchW, gammasOut: batchGamma,
                numRouted: UInt32(routed), numShared: UInt32(shared),
                topK: UInt32(topK), routeScale: 8,
                d: UInt32(D), rows: UInt32(T))
        }
        let sIdx = scalarIdx.contents().bindMemory(to: UInt32.self, capacity: T * 8)
        let bIdx = batchIdx.contents().bindMemory(to: UInt32.self, capacity: T * 8)
        let sW = scalarW.contents().bindMemory(to: Float16.self, capacity: T * 8)
        let bW = batchW.contents().bindMemory(to: Float16.self, capacity: T * 8)
        let sG = scalarGamma.contents().bindMemory(to: Float.self, capacity: T * shared)
        let bG = batchGamma.contents().bindMemory(to: Float.self, capacity: T * shared)
        for t in 0..<T {
            for i in 0..<topK {
                #expect(bIdx[t * 8 + i] == sIdx[t * 8 + i])
                #expect(bW[t * 8 + i] == sW[t * 8 + i])
            }
            for i in 0..<shared { #expect(bG[t * shared + i] == sG[t * shared + i]) }
        }
    }
}
