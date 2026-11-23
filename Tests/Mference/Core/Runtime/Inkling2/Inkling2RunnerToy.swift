import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// A synthetic Inkling 2 model at Mini's width, written directly in the
/// shipped file's formats into shared Metal buffers. K3's attention kernels
/// are fixed to Mini's attention shape (2,304 wide, 24/8 heads of 128) and
/// K2's router to 128 experts, so only the expert width (64), the vocabulary
/// (512 rows, 500 tokens) and the content are small. Every tensor kind of the
/// shipped file is present: Q8_0 dense projections, embedding and head; BF16
/// sink and dynamic-tap projections; F32 norms, clips, convolutions, residual
/// and message queries, loop gates and router; NVFP4 experts with per-expert
/// scales on layers 0 to 19 and Q8_0 experts on 20 to 23.
///
/// `runnerWeights` feeds `Inkling2ForwardRunner`; `referenceWeights` reads
/// the same bytes through `Inkling2BufferMatrix` for `Inkling2ReferenceModel`.
/// Every tensor has its own seeded stream, so its values do not depend on the
/// layout. About 1.6 GB of buffers.
struct Inkling2RunnerToy {
    static let config = Inkling2ReferenceConfig(
        hiddenSize: 2304, numLayers: 24, coreStart: 6, coreEnd: 18, numPasses: 3, gateRank: 64,
        attnResBlockSize: 6, slidingLayers: (0..<24).map { ($0 + 1) % 6 != 0 },
        numHeads: 24, numKVHeads: 8, headDim: 128, slidingWindow: 1024, relDim: 16, relExtent: 1024,
        qkClip: 22.627416997969522, relClip: 8.0, logScaleFloor: 8192, logScaleAlpha: 0.1,
        convKernel: 4, numExperts: 128, topK: 8, expertFF: 64, routeScale: 8.0,
        gateSoftcap: 4.0, upSoftcap: 25.0, finalLogitSoftcap: 30.0,
        vocabPadded: 512, vocabUnpadded: 500, rmsEps: 1e-6)

    let dense: MTLBuffer
    let floats: MTLBuffer
    let experts: [MTLBuffer]
    let expertScales: MTLBuffer
    let runnerWeights: Inkling2RunnerWeights
    let referenceWeights: Inkling2ReferenceWeights

    /// Byte layout planned first, filled after the buffer exists.
    private final class Plan {
        var size = 0
        var jobs: [(UnsafeMutableRawPointer) -> Void] = []
        func add(_ bytes: Int, align: Int = 16, _ fill: @escaping (UnsafeMutableRawPointer) -> Void) -> Int {
            size = (size + align - 1) / align * align
            let at = size
            size += bytes
            jobs.append { fill($0 + at) }
            return at
        }
        func make(_ device: MTLDevice, _ name: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(16, size), options: .storageModeShared) else {
                throw Inkling2ForwardRunnerError.allocationFailed(name)
            }
            b.label = name
            for job in jobs { job(b.contents()) }
            return b
        }
    }

    // MARK: Fills (one SplitMix stream per tensor name)

    /// Every block gets its own scale (0.5 to 1.5 times `amplitude / 127`),
    /// so a block-scale indexing error anywhere in a Q8_0 product shows.
    private static func q8(_ p: UnsafeMutableRawPointer, rows: Int, cols: Int, amplitude: Float, rng: SplitMix64) {
        var rng = rng
        for b in 0..<(rows * cols / 32) {
            let block = p + b * 34
            let scale = Float16(amplitude / 127 * rng.uniform(0.5, 1.5)).bitPattern
            block.storeBytes(of: scale.littleEndian, as: UInt16.self)
            // Random int8 codes, eight per store (a -128 decodes the same on
            // the GPU and in the CPU reference).
            for w in 0..<4 { (block + 2 + w * 8).storeBytes(of: rng.next(), as: UInt64.self) }
        }
    }

    private static func bf16(_ p: UnsafeMutableRawPointer, count: Int, amplitude: Float, rng: SplitMix64) {
        var rng = rng
        for i in 0..<count {
            let bits = UInt16(truncatingIfNeeded: rng.uniform(-amplitude, amplitude).bitPattern >> 16)
            (p + 2 * i).storeBytes(of: bits.littleEndian, as: UInt16.self)
        }
    }

    private static func f32(_ p: UnsafeMutableRawPointer, count: Int, lo: Float, hi: Float, rng: SplitMix64) {
        var rng = rng
        let out = p.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { out[i] = rng.uniform(lo, hi) }
    }

    /// NVFP4 rows: block scale bytes 100 ..< 110 (24 ..< 40 after the halving),
    /// random codes.
    private static func nvfp4(_ p: UnsafeMutableRawPointer, rows: Int, k: Int, rng: SplitMix64) {
        var rng = rng
        for b in 0..<(rows * k / 64) {
            let block = p + b * 36
            for s in 0..<4 { (block + s).storeBytes(of: UInt8(100 + rng.next() % 10), as: UInt8.self) }
            for w in 0..<4 { (block + 4 + 8 * w).storeBytes(of: rng.next(), as: UInt64.self) }
        }
    }

    // swiftlint:disable:next function_body_length
    init(device: MTLDevice, seed: UInt64 = 0x1A_2B) throws {
        let c = Self.config
        let d = c.hiddenSize
        let k = c.convKernel
        let tree = SeedTree(seed)
        let densePlan = Plan()
        let floatPlan = Plan()
        func q8(_ name: String, _ rows: Int, _ cols: Int, _ amplitude: Float) -> Int {
            let rng = tree.key(name)
            return densePlan.add(rows * cols / 32 * 34) { Self.q8($0, rows: rows, cols: cols, amplitude: amplitude, rng: rng) }
        }
        func bf16(_ name: String, _ rows: Int, _ cols: Int, _ amplitude: Float) -> Int {
            let rng = tree.key(name)
            return densePlan.add(rows * cols * 2) { Self.bf16($0, count: rows * cols, amplitude: amplitude, rng: rng) }
        }
        func f32(_ name: String, _ count: Int, _ lo: Float, _ hi: Float) -> Int {
            let rng = tree.key(name)
            return floatPlan.add(count * 4) { Self.f32($0, count: count, lo: lo, hi: hi, rng: rng) }
        }

        struct Offsets {
            var q = 0, k = 0, v = 0, rel = 0, gate = 0, output = 0, sink = 0
            var kTaps: Int?, vTaps: Int?
            var attnNorm: Int?, inputNorms: [Int] = []
            var qClip = 0, relClip = 0, relBias = 0, kConv = 0, vConv = 0, attnPostConv = 0, ffnPostConv = 0
            var attnRes: Int?, ffnRes = 0, ffnNorm = 0, router = 0, routerBias = 0
            var msgQueries: Int?, down: [Int] = [], gain: [Int] = [], forget: [Int] = []
        }

        let embedding = q8("token_embd", c.vocabPadded, d, 1.0)
        let output = q8("output", c.vocabPadded, d, 0.08)
        let embeddingNorm = f32("token_embd_norm", d, 0.8, 1.2)
        let outputRes = f32("output_res_query", d, -1, 1)
        let outputNorm = f32("output_norm", d, 0.8, 1.2)
        var offsets: [Offsets] = []
        for layer in 0..<c.numLayers {
            let p = "blk.\(layer)."
            var o = Offsets()
            let global = !c.slidingLayers[layer]
            o.q = q8(p + "attn_q", c.qDim, d, 0.05)
            o.k = q8(p + "attn_k", c.kvDim, d, 0.05)
            o.v = q8(p + "attn_v", c.kvDim, d, 0.05)
            o.rel = q8(p + "attn_rel", c.numHeads * c.relDim, d, 0.05)
            o.gate = q8(p + "attn_gate", c.qDim, d, 0.05)
            o.output = q8(p + "attn_output", d, c.qDim, 0.03)
            o.sink = bf16(p + "attn_sink_proj", c.numHeads, d, 0.05)
            if global {
                o.kTaps = bf16(p + "attn_k_conv_proj", c.numKVHeads * k, d, 0.01)
                o.vTaps = bf16(p + "attn_v_conv_proj", c.numKVHeads * k, d, 0.01)
                o.inputNorms = ["q", "k", "v"].map { f32(p + "attn_\($0)_inp_norm", d, 0.8, 1.2) }
            } else {
                o.attnNorm = f32(p + "attn_norm", d, 0.8, 1.2)
            }
            o.qClip = f32(p + "attn_q_clip", c.qDim, 0.6, 1.4)
            o.relClip = f32(p + "attn_rel_clip", c.numHeads * c.relDim, 0.6, 1.4)
            o.relBias = f32(p + "attn_rel_bias", c.relativeExtent(layer: layer) * c.relDim, -1, 1)
            o.kConv = f32(p + "attn_k_conv", c.kvDim * k, -0.5, 0.5)
            o.vConv = f32(p + "attn_v_conv", c.kvDim * k, -0.5, 0.5)
            o.attnPostConv = f32(p + "attn_post_conv", d * k, -0.5, 0.5)
            o.ffnPostConv = f32(p + "ffn_post_conv", d * k, -0.5, 0.5)
            if layer > 0 { o.attnRes = f32(p + "attn_res_query", (global ? 3 : 1) * d, -1, 1) }
            o.ffnRes = f32(p + "ffn_res_query", d, -1, 1)
            o.ffnNorm = f32(p + "ffn_norm", d, 0.8, 1.2)
            o.router = f32(p + "ffn_gate_inp", c.numExperts * d, -0.05, 0.05)
            o.routerBias = f32(p + "exp_probs_b", c.numExperts, -0.1, 0.1)
            if c.isCore(layer) {
                o.msgQueries = f32(p + "loop_msg_query", 4 * d, -1, 1)
                let gates = ["q", "k", "v", "ffn"]
                o.down = gates.map { f32(p + "loop_\($0)_ctrl_down", c.gateRank * 2 * d, -0.02, 0.02) }
                o.gain = gates.map { f32(p + "loop_\($0)_ctrl_gain", d * c.gateRank, -0.3, 0.3) }
                o.forget = gates.map { f32(p + "loop_\($0)_ctrl_forget", d * c.gateRank, -0.3, 0.3) }
            }
            offsets.append(o)
        }
        dense = try densePlan.make(device, "toy.dense")
        floats = try floatPlan.make(device, "toy.f32")

        // Experts: one buffer per layer, blob = gate_up then down (256-byte
        // aligned), stride rounded to 16 KiB, as the install lays them out.
        let shape = Inkling2MoEShape(d: d, f: c.expertFF, experts: c.numExperts, topK: c.topK)
        func layout(_ encoding: Inkling2MoEEncoding) -> (down: Int, stride: Int) {
            let down = (shape.gateUpBytes(encoding) + 255) / 256 * 256
            return (down, (down + shape.downBytes(encoding) + 16_383) / 16_384 * 16_384)
        }
        var expertBuffers: [MTLBuffer] = []
        for layer in 0..<c.numLayers {
            let encoding: Inkling2MoEEncoding = layer < 20 ? .nvfp4 : .q8_0
            let (downAt, stride) = layout(encoding)
            let plan = Plan()
            for e in 0..<c.numExperts {
                let gu = tree.key("blk.\(layer).ffn_gate_up_exps.\(e)")
                let dn = tree.key("blk.\(layer).ffn_down_exps.\(e)")
                _ = plan.add(stride, align: 16_384) { base in
                    if encoding == .nvfp4 {
                        Self.nvfp4(base, rows: 2 * c.expertFF, k: d, rng: gu)
                        Self.nvfp4(base + downAt, rows: d, k: c.expertFF, rng: dn)
                    } else {
                        Self.q8(base, rows: 2 * c.expertFF, cols: d, amplitude: 0.1, rng: gu)
                        Self.q8(base + downAt, rows: d, cols: c.expertFF, amplitude: 0.2, rng: dn)
                    }
                }
            }
            expertBuffers.append(try plan.make(device, "toy.experts.\(layer)"))
        }
        experts = expertBuffers
        let scalePlan = Plan()
        let scalesAt = (0..<c.numLayers).map { layer -> Int in
            let rng = tree.key("blk.\(layer).exps.scale")
            let nvfp4 = layer < 20
            return scalePlan.add(2 * c.numExperts * 4) { p in
                if nvfp4 { Self.f32(p, count: 2 * c.numExperts, lo: 5e-4, hi: 1.5e-3, rng: rng) }
                else { Self.f32(p, count: 2 * c.numExperts, lo: 1, hi: 1, rng: rng) }
            }
        }
        let scaleBuffer = try scalePlan.make(device, "toy.expert-scales")
        expertScales = scaleBuffer
        let layerBuffers = expertBuffers

        // MARK: Runner and reference views of the same bytes
        let f = floats, dn = dense
        func ref(_ at: Int) -> Inkling2BufferRef { .init(f, offset: at) }
        func m(_ buffer: MTLBuffer, _ at: Int, _ rows: Int, _ cols: Int,
               _ format: Inkling2BufferMatrix.Format) -> Inkling2BufferMatrix {
            Inkling2BufferMatrix(buffer: buffer, offset: at, rows: rows, cols: cols, format: format)
        }
        func v(_ at: Int, _ count: Int) -> [Float] { Inkling2BufferMatrix.floats(f, offset: at, count: count) }

        var layers: [Inkling2RunnerWeights.Layer] = []
        var refLayers: [Inkling2ReferenceLayerWeights] = []
        var loopLayers: [Inkling2ReferenceLoopWeights] = []
        for (layer, o) in offsets.enumerated() {
            let encoding: Inkling2MoEEncoding = layer < 20 ? .nvfp4 : .q8_0
            let (downAt, stride) = layout(encoding)
            let scaleBase = scalesAt[layer]
            let pool = try Inkling2ExpertPool(buffer: layerBuffers[layer], encoding: encoding, shape: shape,
                                              expertStride: stride, downOffset: downAt,
                                              gateUpScales: scaleBuffer, gateUpScalesOffset: scaleBase,
                                              downScales: scaleBuffer,
                                              downScalesOffset: scaleBase + c.numExperts * 4)
            let loop = o.msgQueries.map { mq in
                Inkling2RunnerWeights.Loop(messageQueries: ref(mq), down: o.down.map(ref),
                                           gain: o.gain.map(ref), forget: o.forget.map(ref))
            }
            layers.append(.init(
                isGlobal: !c.slidingLayers[layer], attnNorm: o.attnNorm.map(ref), inputNorms: o.inputNorms.map(ref),
                q: o.q, k: o.k, v: o.v, rel: o.rel, gate: o.gate, output: o.output,
                sinkProj: o.sink, kConvProj: o.kTaps, vConvProj: o.vTaps,
                qClip: ref(o.qClip), relClip: ref(o.relClip), relBias: ref(o.relBias),
                kConv: ref(o.kConv), vConv: ref(o.vConv), attnPostConv: ref(o.attnPostConv),
                ffnPostConv: ref(o.ffnPostConv), attnResQuery: o.attnRes.map(ref), ffnResQuery: ref(o.ffnRes),
                ffnNorm: ref(o.ffnNorm), router: ref(o.router), routerBias: ref(o.routerBias),
                experts: pool, loop: loop))

            let scales = Inkling2BufferMatrix.floats(scaleBuffer, offset: scaleBase, count: 2 * c.numExperts)
            func expert(_ e: Int, down: Bool) -> Inkling2BufferMatrix {
                let at = e * stride + (down ? downAt : 0)
                let rows = down ? d : 2 * c.expertFF, cols = down ? c.expertFF : d
                let format: Inkling2BufferMatrix.Format = encoding == .nvfp4
                    ? .nvfp4(scale: scales[(down ? c.numExperts : 0) + e]) : .q8_0
                return m(layerBuffers[layer], at, rows, cols, format)
            }
            let queries = c.slidingLayers[layer] ? 1 : 3
            refLayers.append(Inkling2ReferenceLayerWeights(
                wq: m(dn, o.q, c.qDim, d, .q8_0), wk: m(dn, o.k, c.kvDim, d, .q8_0),
                wv: m(dn, o.v, c.kvDim, d, .q8_0), wo: m(dn, o.output, d, c.qDim, .q8_0),
                outputGate: m(dn, o.gate, c.qDim, d, .q8_0),
                rel: m(dn, o.rel, c.numHeads * c.relDim, d, .q8_0),
                sinkProj: m(dn, o.sink, c.numHeads, d, .bf16),
                qClipScale: v(o.qClip, c.qDim), relClipScale: v(o.relClip, c.numHeads * c.relDim),
                relBias: v(o.relBias, c.relativeExtent(layer: layer) * c.relDim),
                kConv: v(o.kConv, c.kvDim * k), vConv: v(o.vConv, c.kvDim * k),
                attnPostConv: v(o.attnPostConv, d * k), ffnPostConv: v(o.ffnPostConv, d * k),
                attnNorm: o.attnNorm.map { v($0, d) },
                qInputNorm: o.inputNorms.isEmpty ? nil : v(o.inputNorms[0], d),
                kInputNorm: o.inputNorms.isEmpty ? nil : v(o.inputNorms[1], d),
                vInputNorm: o.inputNorms.isEmpty ? nil : v(o.inputNorms[2], d),
                kConvProj: o.kTaps.map { m(dn, $0, c.numKVHeads * k, d, .bf16) },
                vConvProj: o.vTaps.map { m(dn, $0, c.numKVHeads * k, d, .bf16) },
                attnResQueries: o.attnRes.map { at in (0..<queries).map { v(at + $0 * d * 4, d) } } ?? [],
                ffnResQuery: v(o.ffnRes, d), ffnNorm: v(o.ffnNorm, d),
                router: m(f, o.router, c.numExperts, d, .f32), routerBias: v(o.routerBias, c.numExperts),
                expertGateUp: (0..<c.numExperts).map { expert($0, down: false) },
                expertDown: (0..<c.numExperts).map { expert($0, down: true) }))
            if let mq = o.msgQueries {
                loopLayers.append(Inkling2ReferenceLoopWeights(
                    messageQueries: (0..<4).map { v(mq + $0 * d * 4, d) },
                    controlDown: o.down.map { m(f, $0, c.gateRank, 2 * d, .f32) },
                    controlGain: o.gain.map { m(f, $0, d, c.gateRank, .f32) },
                    controlForget: o.forget.map { m(f, $0, d, c.gateRank, .f32) }))
            }
        }
        runnerWeights = Inkling2RunnerWeights(
            dense: dense, embedding: embedding, embeddingNorm: ref(embeddingNorm),
            outputResQuery: ref(outputRes), outputNorm: ref(outputNorm), output: output,
            layers: layers, vocabRows: c.vocabPadded, vocab: c.vocabUnpadded, gateRank: c.gateRank,
            routeScale: c.routeScale, finalLogitSoftcap: c.finalLogitSoftcap, rmsEps: c.rmsEps)
        referenceWeights = Inkling2ReferenceWeights(
            embedding: m(dense, embedding, c.vocabPadded, d, .q8_0),
            embeddingNorm: v(embeddingNorm, d), outputResQuery: v(outputRes, d), outputNorm: v(outputNorm, d),
            output: m(dense, output, c.vocabPadded, d, .q8_0),
            layers: refLayers, loopLayers: loopLayers)
    }
}
