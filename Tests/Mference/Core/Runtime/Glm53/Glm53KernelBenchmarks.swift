import Foundation
import Metal
import Testing
@testable import Mference

/// **Measurement**, not a gate: wall time of each GLM-5.3 decode kernel class
/// at the production shape on random data, as achieved GB/s of weight bytes,
/// so the per-token GPU budget (measured ~86 ms at first light against ~15 ms
/// of weight traffic at memory bandwidth) can be attributed. Env-gated:
/// `MFERENCE_GLM53_KBENCH=1`. Prints one line per kernel.
@Suite(.serialized) struct Glm53KernelBenchmarks {

    private static var enabled: Bool { ProcessInfo.processInfo.environment["MFERENCE_GLM53_KBENCH"] == "1" }

    private static func randomBytes(_ device: MTLDevice, _ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: max(1, count), options: .storageModeShared)!
        let p = b.contents().bindMemory(to: UInt32.self, capacity: count / 4)
        var x: UInt32 = 0x9E37_79B9
        for i in 0..<(count / 4) { x = x &* 1_664_525 &+ 1_013_904_223; p[i] = x }
        return b
    }
    private static func halfBuffer(_ device: MTLDevice, _ count: Int, scale: Float = 0.5) -> MTLBuffer {
        let b = device.makeBuffer(length: max(1, count) * 2, options: .storageModeShared)!
        let p = b.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { p[i] = Float16(Float(i % 97) / 97 * scale - scale / 2) }
        return b
    }
    /// INT8 g64 weights [m, n] with BF16 scales/biases laid out as the install does.
    private static func int8Matrix(_ device: MTLDevice, m: Int, n: Int) -> TensorView {
        let groups = n / 64
        let wBytes = m * n, sBytes = m * groups * 2
        let buf = randomBytes(device, wBytes + 2 * sBytes)
        let sp = buf.contents().advanced(by: wBytes).bindMemory(to: UInt16.self, capacity: 2 * m * groups)
        for i in 0..<(2 * m * groups) { sp[i] = Quantization.bf16Bits(i % 2 == 0 ? 0.01 : -1.2) }
        return TensorView(buffer: buf, offset: 0, length: UInt64(wBytes),
                          scaleOffset: UInt64(wBytes), scaleLength: UInt64(sBytes),
                          biasOffset: UInt64(wBytes + sBytes), biasLength: UInt64(sBytes),
                          shape: (UInt32(m), UInt32(n), 0, 0), dtype: 0)
    }

    /// Times `iterations` command buffers of `encode` and reports GB/s over `bytes` each.
    private static func time(_ ctx: MetalContext, label: String, bytes: Int, iterations: Int = 20,
                             encode: (MTLCommandBuffer) -> Void) throws {
        // Warm-up.
        let warm = try #require(ctx.queue.makeCommandBuffer())
        encode(warm); warm.commit(); warm.waitUntilCompleted()
        var gpu: Double = 0
        let start = Date()
        for _ in 0..<iterations {
            let cb = try #require(ctx.queue.makeCommandBuffer())
            encode(cb); cb.commit(); cb.waitUntilCompleted()
            gpu += cb.gpuEndTime - cb.gpuStartTime
        }
        let wall = Date().timeIntervalSince(start) / Double(iterations)
        let perGPU = gpu / Double(iterations)
        print(String(format: "  [kbench] %-44@ gpu %7.3f ms  wall %7.3f ms  %7.1f GB/s  (%.1f MB)",
                     label as NSString, perGPU * 1e3, wall * 1e3,
                     Double(bytes) / perGPU / 1e9, Double(bytes) / 1e6))
    }

    @Test func batchedPrefillKernelsAtProductionShape() throws {
        guard Self.enabled else { return }
        let ctx = try MetalContext()
        let device = ctx.device
        let k = try Glm53PrefillKernels(context: ctx, swigluLimit: 10)
        let hidden = 4096, T = 64
        let x = Self.halfBuffer(device, T * 32768)
        let y = Self.halfBuffer(device, T * 16384)
        print("  [kbench] GLM-5.3-Flash batched prefill kernels, T = \(T), production shapes")
        func gemm(_ w: TensorView, m: Int, n: Int, _ cb: MTLCommandBuffer) {
            k.encodeInt8GEMM(commandBuffer: cb, weights: w, x: x, xStride: n, y: y, yStride: m, m: m, n: n, tokens: T)
        }
        let sq = Self.int8Matrix(device, m: 8192, n: hidden)
        try Self.time(ctx, label: "gemm 8192x4096 x T (kda q)", bytes: 8192 * hidden) { gemm(sq, m: 8192, n: hidden, $0) }
        let oS = Self.int8Matrix(device, m: hidden, n: 32768)
        try Self.time(ctx, label: "gemm 4096x32768 x T (sparse o_proj)", bytes: hidden * 32768) { gemm(oS, m: hidden, n: 32768, $0) }
        let sh = Self.int8Matrix(device, m: 2048, n: hidden)
        try Self.time(ctx, label: "gemm 2048x4096 x T (shared)", bytes: 2048 * hidden) { gemm(sh, m: 2048, n: hidden, $0) }
        let small = Self.int8Matrix(device, m: 128, n: hidden)
        try Self.time(ctx, label: "gemm 128x4096 x T (f_a)", bytes: 128 * hidden) { gemm(small, m: 128, n: hidden, $0) }
        let up = Self.int8Matrix(device, m: 8192, n: 128)
        try Self.time(ctx, label: "gemm 8192x128 x T (f_b)", bytes: 8192 * 128) { gemm(up, m: 8192, n: 128, $0) }
        let dense = Self.int8Matrix(device, m: 12288, n: hidden)
        try Self.time(ctx, label: "gemm 12288x4096 x T (dense)", bytes: 12288 * hidden) { gemm(dense, m: 12288, n: hidden, $0) }
        let router = device.makeBuffer(length: 288 * hidden * 2, options: .storageModeShared)!
        let routerView = TensorView(buffer: router, offset: 0, length: UInt64(288 * hidden * 2), scaleOffset: 0, scaleLength: 0,
                                    biasOffset: 0, biasLength: 0, shape: (288, UInt32(hidden), 0, 0), dtype: 1)
        let logits = device.makeBuffer(length: T * 288 * 4, options: .storageModeShared)!
        try Self.time(ctx, label: "bf16 gemm 288x4096 x T (router)", bytes: 288 * hidden * 2) { cb in
            k.encodeBF16GEMM(commandBuffer: cb, weights: routerView, x: x, xStride: hidden, y: logits, yStride: 288,
                             outputFloat32: true, m: 288, n: hidden, tokens: T)
        }
        let H = 64, D = 128, qkv = H * D
        let f32 = { (n: Int) -> TensorView in
            let buf = device.makeBuffer(length: n * 4, options: .storageModeShared)!
            return TensorView(buffer: buf, offset: 0, length: UInt64(n * 4), scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0, shape: (UInt32(n), 0, 0, 0), dtype: 3)
        }
        let bf = { (n: Int) -> TensorView in
            let buf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
            return TensorView(buffer: buf, offset: 0, length: UInt64(n * 2), scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0, shape: (UInt32(n), 0, 0, 0), dtype: 1)
        }
        let state = device.makeBuffer(length: H * D * D * 4, options: .storageModeShared)!
        let conv = Self.halfBuffer(device, T * 3 * qkv), a = Self.halfBuffer(device, T * qkv)
        let b = Self.halfBuffer(device, T * H), gate = Self.halfBuffer(device, T * qkv)
        try Self.time(ctx, label: "kda chunk 64 heads x T", bytes: T * 3 * qkv * 2) { cb in
            k.encodeKDAChunk(commandBuffer: cb, convOut: conv, a: a, b: b, gate: gate, aLog: f32(H), dtBias: f32(qkv),
                             oNorm: bf(D), state: state, out: y, heads: H, headDim: D, tokens: T, lowerBound: -5, eps: 1e-5)
        }
        let streams = Self.halfBuffer(device, T * 4 * hidden)
        let partials = device.makeBuffer(length: T * 25 * 4, options: .storageModeShared)!
        let pre = device.makeBuffer(length: T * 16, options: .storageModeShared)!
        let post = device.makeBuffer(length: T * 16, options: .storageModeShared)!
        let comb = device.makeBuffer(length: T * 64, options: .storageModeShared)!
        try Self.time(ctx, label: "hc weights + collapse + place-mix x T", bytes: T * 4 * hidden * 2 * 3) { cb in
            k.encodeHCWeights(commandBuffer: cb, streams: streams, fn: f32(24 * 4 * hidden), base: f32(24), scale: f32(3),
                              partials: partials, outPre: pre, outPost: post, outComb: comb, hcMult: 4, hidden: hidden,
                              sinkhornIters: 20, hcEps: 1e-6, rmsEps: 1e-5, tokens: T)
            k.encodeHCCollapse(commandBuffer: cb, streams: streams, pre: pre, x: x, hcMult: 4, hidden: hidden, tokens: T)
            k.encodeHCPlaceMix(commandBuffer: cb, streams: streams, sub: x, post: post, comb: comb, outStreams: streams,
                               hcMult: 4, hidden: hidden, tokens: T)
        }
        let lat = Self.halfBuffer(device, 2048 * 512)
        let qLat = Self.halfBuffer(device, T * 64 * 512), oLat = Self.halfBuffer(device, T * 64 * 512)
        try Self.time(ctx, label: "latent attention causal 64 heads x T (base 1024)", bytes: 64 * T * 1056 * 512 * 2) { cb in
            k.encodeLatentAttentionCausal(commandBuffer: cb, qLatent: qLat, latents: lat, out: oLat, heads: 64,
                                          latentDim: 512, base: 1024, tokens: T, scale: 0.0625)
        }
        let embedQ = Self.int8Matrix(device, m: 64 * 512, n: 256)
        try Self.time(ctx, label: "headed gemv embed_q x T", bytes: T * 64 * 512 * 256) { cb in
            k.encodeHeadedGEMV(commandBuffer: cb, weights: embedQ, x: x, y: qLat, heads: 64, m: 512, n: 256, tokens: T)
        }
        // Grouped experts: 16 resident experts, T x 8 routes spread over them.
        let stride = 14_155_776
        let slabBuf = Self.randomBytes(device, 16 * stride)
        let slab = ResidentExpertSlab(buffer: slabBuf, baseOffset: 0, expertStride: stride)
        let offsets = MoEExpertOffsets(gateWOff: 0, gateSOff: 4_194_304, gateBOff: 4_456_448,
                                       upWOff: 4_718_592, upSOff: 8_912_896, upBOff: 9_175_040,
                                       downWOff: 9_437_184, downSOff: 13_631_488, downBOff: 13_893_632)
        let K = 8, pairs = T * K
        let pairToken = device.makeBuffer(length: pairs * 4, options: .storageModeShared)!
        let routePair = device.makeBuffer(length: pairs * 4, options: .storageModeShared)!
        let segStart = device.makeBuffer(length: 17 * 4, options: .storageModeShared)!
        let active = device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        let pt = pairToken.contents().bindMemory(to: UInt32.self, capacity: pairs)
        let rp = routePair.contents().bindMemory(to: UInt32.self, capacity: pairs)
        let sp = segStart.contents().bindMemory(to: UInt32.self, capacity: 17)
        let ap = active.contents().bindMemory(to: UInt32.self, capacity: 16)
        var pair = 0
        for e in 0..<16 {
            ap[e] = UInt32(e); sp[e] = UInt32(pair)
            for t in 0..<T where (t + e) % 2 == 0 { pt[pair] = UInt32(t); rp[t * K + (e / 2)] = UInt32(pair); pair += 1 }
        }
        sp[16] = UInt32(pair)
        let acts = Self.halfBuffer(device, pairs * 2048)
        let partial = device.makeBuffer(length: pairs * hidden * 4, options: .storageModeShared)!
        let weights = Self.halfBuffer(device, pairs, scale: 0.3)
        try Self.time(ctx, label: "grouped moe 16 experts, \(pair) routes", bytes: 16 * stride) { cb in
            k.encodeGroupedMoE(commandBuffer: cb, slab: slab, offsets: offsets, x: x, acts: acts, partial: partial,
                               pairToken: pairToken, segStart: segStart, activeExperts: active, activeCount: 16,
                               routePair: routePair, weights: weights, residual: x, y: y, d: hidden, f: 2048, topK: K, tokens: T)
        }
        try Self.time(ctx, label: "router select batched x T", bytes: T * 288 * 4) { cb in
            k.encodeRouterSelect(commandBuffer: cb, logits: logits, bias: f32(288), outIndices: routePair,
                                 outWeights: weights, numExperts: 288, routeScale: 2.5, tokens: T)
        }
    }

    @Test func decodeKernelsAtProductionShape() throws {
        guard Self.enabled else { return }
        let ctx = try MetalContext()
        let cfg = ArchConfig.glm53Flash_320B_A18B
        let device = ctx.device
        let int8 = try DequantInt8GEMV(context: ctx, additionalShapes: cfg.decodeInt8GEMVShapes)
        let kernels = try Glm53Kernels(context: ctx)
        let rms = try RMSNorm(context: ctx)
        let hidden = 4096
        let x = Self.halfBuffer(device, 32768)
        let y = Self.halfBuffer(device, 160_000)
        func gemv(_ w: TensorView, m: Int, n: Int, _ cb: MTLCommandBuffer) {
            int8.encode(commandBuffer: cb, weights: w.buffer, weightsOffset: 0,
                        scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                        biases: w.buffer, biasesOffset: Int(w.biasOffset),
                        x: x, y: y, m: UInt32(m), n: UInt32(n))
        }
        print("  [kbench] GLM-5.3-Flash decode kernels, production shapes, random data")

        // INT8 GEMVs: the KDA q/k/v/o square, the sparse o_proj, the head.
        let sq = Self.int8Matrix(device, m: 8192, n: hidden)
        try Self.time(ctx, label: "int8 gemv 8192x4096 (kda q/k/v)", bytes: 8192 * hidden + 2 * 8192 * 64 * 2) { gemv(sq, m: 8192, n: hidden, $0) }
        let oK = Self.int8Matrix(device, m: hidden, n: 8192)
        try Self.time(ctx, label: "int8 gemv 4096x8192 (kda o_proj)", bytes: hidden * 8192) { gemv(oK, m: hidden, n: 8192, $0) }
        let oS = Self.int8Matrix(device, m: hidden, n: 32768)
        try Self.time(ctx, label: "int8 gemv 4096x32768 (sparse o_proj)", bytes: hidden * 32768) { gemv(oS, m: hidden, n: 32768, $0) }
        let qb = Self.int8Matrix(device, m: 16384, n: 1536)
        try Self.time(ctx, label: "int8 gemv 16384x1536 (q_b)", bytes: 16384 * 1536) { gemv(qb, m: 16384, n: 1536, $0) }
        let sh = Self.int8Matrix(device, m: 2048, n: hidden)
        try Self.time(ctx, label: "int8 gemv 2048x4096 (shared gate/up)", bytes: 2048 * hidden) { gemv(sh, m: 2048, n: hidden, $0) }
        let shd = Self.int8Matrix(device, m: hidden, n: 2048)
        try Self.time(ctx, label: "int8 gemv 4096x2048 (shared down)", bytes: 2048 * hidden) { gemv(shd, m: hidden, n: 2048, $0) }
        let small = Self.int8Matrix(device, m: 128, n: hidden)
        try Self.time(ctx, label: "int8 gemv 128x4096 (f_a / g_a)", bytes: 128 * hidden) { gemv(small, m: 128, n: hidden, $0) }
        let up = Self.int8Matrix(device, m: 8192, n: 128)
        try Self.time(ctx, label: "int8 gemv 8192x128 (f_b / g_b)", bytes: 8192 * 128) { gemv(up, m: 8192, n: 128, $0) }
        let head = Self.int8Matrix(device, m: 154_880, n: hidden)
        try Self.time(ctx, label: "int8 gemv 154880x4096 (lm_head)", bytes: 154_880 * hidden) { gemv(head, m: 154_880, n: hidden, $0) }
        let dense = Self.int8Matrix(device, m: 12288, n: hidden)
        try Self.time(ctx, label: "int8 gemv 12288x4096 (dense gate/up)", bytes: 12288 * hidden) { gemv(dense, m: 12288, n: hidden, $0) }

        // A whole KDA layer's projections back to back (12 dispatches).
        try Self.time(ctx, label: "kda layer: 12 gemvs", bytes: 3 * 8192 * hidden + hidden * 8192 + 2 * (128 * hidden + 8192 * 128) + 64 * hidden) { cb in
            gemv(sq, m: 8192, n: hidden, cb); gemv(sq, m: 8192, n: hidden, cb); gemv(sq, m: 8192, n: hidden, cb)
            gemv(small, m: 128, n: hidden, cb); gemv(up, m: 8192, n: 128, cb)
            gemv(small, m: 128, n: hidden, cb); gemv(up, m: 8192, n: 128, cb)
            gemv(small, m: 64, n: hidden, cb)
            gemv(oK, m: hidden, n: 8192, cb)
        }

        // KDA recurrence, 64 heads of 128.
        let H = 64, D = 128
        let conv = Self.halfBuffer(device, 3 * H * D), a = Self.halfBuffer(device, H * D)
        let b = Self.halfBuffer(device, H), gate = Self.halfBuffer(device, H * D)
        let f32 = { (n: Int) -> TensorView in
            let buf = device.makeBuffer(length: n * 4, options: .storageModeShared)!
            return TensorView(buffer: buf, offset: 0, length: UInt64(n * 4), scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0, shape: (UInt32(n), 0, 0, 0), dtype: 3)
        }
        let bf = { (n: Int) -> TensorView in
            let buf = device.makeBuffer(length: n * 2, options: .storageModeShared)!
            return TensorView(buffer: buf, offset: 0, length: UInt64(n * 2), scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0, shape: (UInt32(n), 0, 0, 0), dtype: 1)
        }
        let state = device.makeBuffer(length: H * D * D * 4, options: .storageModeShared)!
        let out = Self.halfBuffer(device, H * D), yOut = Self.halfBuffer(device, H * D)
        try Self.time(ctx, label: "kda decode 64x128 (state 4 MB rw)", bytes: 2 * H * D * D * 4) { cb in
            kernels.encodeKDADecode(commandBuffer: cb, convOut: conv, a: a, b: b, gate: gate,
                                    aLog: f32(H), dtBias: f32(H * D), oNorm: bf(D), state: state,
                                    out: out, yOut: yOut, heads: H, headDim: D, lowerBound: -5, eps: 1e-5)
        }
        let tail = Self.halfBuffer(device, 3 * 3 * H * D)
        try Self.time(ctx, label: "conv decode 24576 channels", bytes: 4 * 3 * H * D * 2) { cb in
            kernels.encodeConvDecode(commandBuffer: cb, tail: tail, mixed: conv, convWeight: bf(3 * H * D * 4).buffer,
                                     convWeightOffset: 0, out: yOut, channels: 3 * H * D, taps: 4)
        }

        // mHC: weights (fn 24 x 16384 fp32), collapse, place-mix.
        let streams = Self.halfBuffer(device, 4 * hidden), streams2 = Self.halfBuffer(device, 4 * hidden)
        let fn = f32(24 * 4 * hidden), base = f32(24), scale = f32(3)
        let pre = device.makeBuffer(length: 16, options: .storageModeShared)!
        let post = device.makeBuffer(length: 16, options: .storageModeShared)!
        let comb = device.makeBuffer(length: 64, options: .storageModeShared)!
        let sub = Self.halfBuffer(device, hidden)
        try Self.time(ctx, label: "hc weights + collapse + rmsnorm + place-mix", bytes: 24 * 4 * hidden * 4 + 3 * 4 * hidden * 2) { cb in
            kernels.encodeHCWeights(commandBuffer: cb, streams: streams, fn: fn, base: base, scale: scale,
                                    outPre: pre, outPost: post, outComb: comb, hcMult: 4, hidden: hidden,
                                    sinkhornIters: 20, hcEps: 1e-6, rmsEps: 1e-5)
            kernels.encodeHCCollapse(commandBuffer: cb, streams: streams, pre: pre, x: sub, hcMult: 4, hidden: hidden)
            rms.encodeBF16W(commandBuffer: cb, x: sub, weight: bf(hidden).buffer, weightOffset: 0, out: sub,
                            d: UInt32(hidden), eps: 1e-5)
            kernels.encodeHCPlaceMix(commandBuffer: cb, streams: streams, sub: sub, post: post, comb: comb,
                                     outStreams: streams2, hcMult: 4, hidden: hidden)
        }

        // Latent attention at 512 over 2048 cached rows, and the folds.
        let lat = Self.halfBuffer(device, 2048 * 512)
        let qLat = Self.halfBuffer(device, 64 * 512), oLat = Self.halfBuffer(device, 64 * 512)
        let sel = device.makeBuffer(length: 4, options: .storageModeShared)!
        try Self.time(ctx, label: "latent attention 64 heads x 2048 rows x 512", bytes: 64 * 2048 * 512 * 2) { cb in
            kernels.encodeLatentAttention(commandBuffer: cb, qLatent: qLat, latents: lat, selected: sel, out: oLat,
                                          heads: 64, latentDim: 512, cachedRows: 2048,
                                          selectedCount: Glm53Kernels.attendAll, scale: 0.0625)
        }
        let embedQ = Self.int8Matrix(device, m: 64 * 512, n: 256)
        let unembed = Self.int8Matrix(device, m: 64 * 512, n: 512)
        try Self.time(ctx, label: "headed gemv embed_q + unembed_out", bytes: 64 * 512 * 256 + 64 * 512 * 512) { cb in
            kernels.encodeHeadedInt8GEMV(commandBuffer: cb, weights: embedQ, x: x, y: qLat, heads: 64, m: 512, n: 256)
            kernels.encodeHeadedInt8GEMV(commandBuffer: cb, weights: unembed, x: qLat, y: y, heads: 64, m: 512, n: 512)
        }

        // Routed experts, resident: 8 of 16 slab experts through the slot-map bodies.
        let moe = try MoE(context: ctx, siluActivation: true, specializedD: 4096, specializedF: 2048,
                          specializedNumExperts: 288, specializedTopK: 8, swigluLimit: 10)
        let stride = 14_155_776
        let slab = Self.randomBytes(device, 16 * stride)
        // Companion layout: gate/up/down INT4 [2048x4096]/2 = 4 MB each, scales+biases 2048*64*2*2.
        let offsets = MoEExpertOffsets(gateWOff: 0, gateSOff: 4_194_304, gateBOff: 4_456_448,
                                       upWOff: 4_718_592, upSOff: 8_912_896, upBOff: 9_175_040,
                                       downWOff: 9_437_184, downSOff: 13_631_488, downBOff: 13_893_632)
        let indices = device.makeBuffer(length: 32, options: .storageModeShared)!
        let ip = indices.contents().bindMemory(to: UInt32.self, capacity: 8)
        for i in 0..<8 { ip[i] = UInt32(i * 2) }
        let table = device.makeBuffer(length: 16 * 2, options: .storageModeShared)!
        let tp = table.contents().bindMemory(to: Int16.self, capacity: 16)
        for i in 0..<16 { tp[i] = Int16(i) }
        let slotOffsets = device.makeBuffer(length: 32, options: .storageModeShared)!
        let allHit = device.makeBuffer(length: 4, options: .storageModeShared)!
        let acts = Self.halfBuffer(device, 8 * 2048), weights = Self.halfBuffer(device, 8, scale: 0.3)
        let residual = Self.halfBuffer(device, hidden), mlpOut = Self.halfBuffer(device, hidden)
        try Self.time(ctx, label: "moe resident 8 experts (phase1+phase2)", bytes: 8 * stride) { cb in
            moe.encodeRoutedResidentFFN(commandBuffer: cb, slab: slab, slabOffset: 0, expertStride: stride,
                                        indices: indices, identityTable: table, slotOffsets: slotOffsets,
                                        allHit: allHit, routedOffsets: offsets, x: x, acts: acts,
                                        routingWeights: weights, residual: residual, y: mlpOut,
                                        numExperts: 16, d: 4096, f: 2048, topK: 8)
        }
        let routerLogits = device.makeBuffer(length: 288 * 4, options: .storageModeShared)!
        try Self.time(ctx, label: "router select 288 (serial)", bytes: 288 * 8) { cb in
            kernels.encodeRouterSelect(commandBuffer: cb, logits: routerLogits, bias: f32(288),
                                       outIndices: indices, outWeights: weights, numExperts: 288, routeScale: 2.5)
        }
    }
}
