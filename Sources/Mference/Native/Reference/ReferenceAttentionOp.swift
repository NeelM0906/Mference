import Foundation
import Metal

extension Native {
    /// The reference `AttentionOp`. Two kernels per call:
    /// `ref_attn_append_kv` (k RMSNorm and RoPE, then K and V into the pages,
    /// F16 or INT8 per the geometry) and `ref_attn_core` (one thread per
    /// (row, head): q RMSNorm and RoPE, a max pass and an exp-sum pass over
    /// positions `0...p`, the weighted V sum, then `× sigmoid(gate)`).
    public final class ReferenceAttentionOp: AttentionOp {
        let context: MetalContext
        private var config: Qwen35Config?
        private var geometry: KVPageGeometry?
        private var append: MTLComputePipelineState?
        private var core: MTLComputePipelineState?

        public init(context: MetalContext) {
            self.context = context
        }

        /// Throws unless every rotated pair maps to llama.cpp's IMROPE t, h
        /// or w section (`AttentionOp` contract, step 2).
        public static func checkIMROPE(sections: [Int], ropeDims: Int) throws {
            let s = sections + [Int](repeating: 0, count: max(0, 4 - sections.count))
            let sectDims = s[0] + s[1] + s[2] + s[3]
            guard sectDims > 0 else { return }
            for i in 0..<ropeDims / 2 {
                let sector = i % sectDims
                let ok = (sector % 3 == 1 && sector < 3 * s[1]) || (sector % 3 == 2 && sector < 3 * s[2])
                    || (sector % 3 == 0 && sector < 3 * s[0])
                guard ok else {
                    throw NativeError.unsupported("RoPE pair \(i) falls in IMROPE section e (sections \(sections))")
                }
            }
        }

        public func prepare(config: Qwen35Config, geometry: KVPageGeometry) throws {
            guard config.keyLength == config.valueLength, config.keyLength == geometry.headDim,
                  geometry.headDim <= 256, config.kvHeadCount == geometry.kvHeads,
                  config.headCount % config.kvHeadCount == 0,
                  config.ropeDimensionCount % 2 == 0, config.ropeDimensionCount <= config.keyLength,
                  geometry.attentionLayers == config.fullAttentionLayers.count else {
                throw NativeError.invalidShape("reference attention: config and geometry disagree")
            }
            try Self.checkIMROPE(sections: config.ropeSections, ropeDims: config.ropeDimensionCount)
            append = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_attn_append_kv")
            core = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_attn_core")
            self.config = config
            self.geometry = geometry
        }

        public func encodeDecode(_ encoder: MTLComputeCommandEncoder, attnLayer: Int,
                                 weights: AttentionLayerWeights, qGate: TensorRef, k: TensorRef, v: TensorRef,
                                 startPosition: Int, kv: KVPageTable, output: TensorRef) throws {
            guard qGate.rows == 1 else { throw NativeError.invalidShape("attention decode needs 1 row") }
            try encode(encoder, attnLayer: attnLayer, weights: weights, qGate: qGate, k: k, v: v,
                       startPosition: startPosition, kv: kv, output: output)
        }

        public func encodeVerify(_ encoder: MTLComputeCommandEncoder, attnLayer: Int,
                                 weights: AttentionLayerWeights, qGate: TensorRef, k: TensorRef, v: TensorRef,
                                 startPosition: Int, kv: KVPageTable, output: TensorRef) throws {
            guard (2...16).contains(qGate.rows) else {
                throw NativeError.invalidShape("attention verify needs 2...16 rows")
            }
            try encode(encoder, attnLayer: attnLayer, weights: weights, qGate: qGate, k: k, v: v,
                       startPosition: startPosition, kv: kv, output: output)
        }

        public func encodePrefill(_ encoder: MTLComputeCommandEncoder, attnLayer: Int,
                                  weights: AttentionLayerWeights, qGate: TensorRef, k: TensorRef, v: TensorRef,
                                  startPosition: Int, kv: KVPageTable, output: TensorRef) throws {
            guard qGate.rows >= 1 else { throw NativeError.invalidShape("attention prefill needs rows") }
            try encode(encoder, attnLayer: attnLayer, weights: weights, qGate: qGate, k: k, v: v,
                       startPosition: startPosition, kv: kv, output: output)
        }

        struct Params {
            var rows: UInt32
            var startPosition: UInt32
            var headCount: UInt32
            var kvHeadCount: UInt32
            var headDim: UInt32
            var ropeDims: UInt32
            var eps: Float
            var freqBase: Float
            var scale: Float
            var qGateStride: UInt32
            var kStride: UInt32
            var vStride: UInt32
            var outStride: UInt32
            var tokensPerPage: UInt32
            var attnLayer: UInt32
            var attentionLayers: UInt32
            var int8: UInt32
            var pageBytes: UInt32
        }

        private func encode(_ encoder: MTLComputeCommandEncoder, attnLayer: Int, weights: AttentionLayerWeights,
                            qGate: TensorRef, k: TensorRef, v: TensorRef, startPosition: Int, kv: KVPageTable,
                            output: TensorRef) throws {
            guard let config, let geometry, let append, let core else {
                throw NativeError.resource("ReferenceAttentionOp.prepare not called")
            }
            guard kv.geometry == geometry else { throw NativeError.invalidShape("KV geometry differs from prepare") }
            let rows = qGate.rows
            let H = config.headCount, KVH = config.kvHeadCount, D = config.keyLength
            try ReferenceMetal.checkF32(qGate, rows: rows, cols: H * D * 2, "attention qGate")
            try ReferenceMetal.checkF32(k, rows: rows, cols: KVH * D, "attention k")
            try ReferenceMetal.checkF32(v, rows: rows, cols: KVH * D, "attention v")
            try ReferenceMetal.checkF32(output, rows: rows, cols: H * D, "attention output")
            guard (0..<geometry.attentionLayers).contains(attnLayer) else {
                throw NativeError.invalidShape("attnLayer \(attnLayer)")
            }
            guard kv.tokenCount == startPosition + rows,
                  kv.pageCount * geometry.tokensPerPage >= kv.tokenCount else {
                throw NativeError.stateOutOfRange("KV holds \(kv.tokenCount) tokens, step needs \(startPosition + rows)")
            }
            for norm in [weights.qNorm, weights.kNorm] {
                guard norm.type == .f32, norm.cols == D, norm.rows == 1, norm.layout == .gguf else {
                    throw NativeError.invalidShape("\(norm.name): expected F32 [\(D)]")
                }
            }
            let params = Params(rows: UInt32(rows), startPosition: UInt32(startPosition), headCount: UInt32(H),
                                kvHeadCount: UInt32(KVH), headDim: UInt32(D),
                                ropeDims: UInt32(config.ropeDimensionCount), eps: config.rmsNormEpsilon,
                                freqBase: config.ropeFreqBase, scale: 1 / Float(D).squareRoot(),
                                qGateStride: UInt32(qGate.rowStride), kStride: UInt32(k.rowStride),
                                vStride: UInt32(v.rowStride), outStride: UInt32(output.rowStride),
                                tokensPerPage: UInt32(geometry.tokensPerPage), attnLayer: UInt32(attnLayer),
                                attentionLayers: UInt32(geometry.attentionLayers),
                                int8: geometry.precision == .int8 ? 1 : 0, pageBytes: UInt32(geometry.pageBytes))

            encoder.setBuffer(k.buffer, offset: k.offset, index: 0)
            encoder.setBuffer(v.buffer, offset: v.offset, index: 1)
            encoder.setBuffer(weights.kNorm.buffer, offset: weights.kNorm.offset, index: 2)
            encoder.setBuffer(kv.arena, offset: 0, index: 3)
            encoder.setBuffer(kv.pageTable, offset: kv.pageTableOffset, index: 4)
            ReferenceMetal.setBytes(encoder, params, index: 5)
            ReferenceMetal.dispatch(encoder, append, width: 2 * KVH, height: rows)

            encoder.setBuffer(qGate.buffer, offset: qGate.offset, index: 0)
            encoder.setBuffer(weights.qNorm.buffer, offset: weights.qNorm.offset, index: 1)
            encoder.setBuffer(kv.arena, offset: 0, index: 2)
            encoder.setBuffer(kv.pageTable, offset: kv.pageTableOffset, index: 3)
            encoder.setBuffer(output.buffer, offset: output.offset, index: 4)
            ReferenceMetal.setBytes(encoder, params, index: 5)
            ReferenceMetal.dispatch(encoder, core, width: H, height: rows)
        }
    }
}

extension Native.ReferenceCPU {
    /// CPU twin of the reference attention, over a CPU K/V history.
    public final class Attention {
        public let headCount: Int
        public let kvHeadCount: Int
        public let headDim: Int
        public let ropeDims: Int
        public let eps: Float
        public let freqBase: Float
        public let precision: Native.KVPrecision
        /// Stored K and V as the kernels read them back (F16- or INT8-rounded),
        /// `[position][kvHead][dim]` flattened per position.
        public private(set) var keys: [[Float]] = []
        public private(set) var values: [[Float]] = []

        public init(headCount: Int, kvHeadCount: Int, headDim: Int, ropeDims: Int, eps: Float, freqBase: Float,
                    precision: Native.KVPrecision) {
            self.headCount = headCount
            self.kvHeadCount = kvHeadCount
            self.headDim = headDim
            self.ropeDims = ropeDims
            self.eps = eps
            self.freqBase = freqBase
            self.precision = precision
        }

        public static func headRMSNorm(_ v: inout [Float], weight: [Float], eps: Float) {
            var sum: Float = 0
            for x in v { sum += x * x }
            let scale = 1 / (sum / Float(v.count) + eps).squareRoot()
            for i in v.indices { v[i] = (v[i] * scale) * weight[i] }
        }

        public static func ropeNeox(_ v: inout [Float], dims n: Int, position: Int, base: Float) {
            let invNDims = -1 / Float(n)
            for i in 0..<n / 2 {
                let theta = Float(position) * pow(base, invNDims * Float(2 * i))
                let c = cos(theta), s = sin(theta)
                let x0 = v[i], x1 = v[i + n / 2]
                v[i] = x0 * c - x1 * s
                v[i + n / 2] = x0 * s + x1 * c
            }
        }

        /// The stored (rounded) form of one K or V head row.
        public func stored(_ row: [Float]) -> [Float] {
            switch precision {
            case .f16:
                return row.map { Float(Float16($0)) }
            case .int8:
                var amax: Float = 0
                for x in row { amax = max(amax, abs(x)) }
                let s = Float(Float16(amax / 127))
                return row.map { x in
                    let q: Float = s == 0 ? 0 : min(max((x / s).rounded(.toNearestOrEven), -127), 127)
                    return s * q
                }
            }
        }

        /// One step: append `rows` rows at the next positions, return the gated
        /// output `[rows x headCount*headDim]`.
        public func step(qGate: [Float], k: [Float], v: [Float], rows: Int, qNorm: [Float],
                         kNorm: [Float]) -> [Float] {
            let D = headDim, H = headCount, KVH = kvHeadCount
            let start = keys.count
            for r in 0..<rows {
                var kRow: [Float] = []
                var vRow: [Float] = []
                for h in 0..<KVH {
                    var kh = Array(k[(r * KVH + h) * D ..< (r * KVH + h + 1) * D])
                    Self.headRMSNorm(&kh, weight: kNorm, eps: eps)
                    Self.ropeNeox(&kh, dims: ropeDims, position: start + r, base: freqBase)
                    kRow += stored(kh)
                    vRow += stored(Array(v[(r * KVH + h) * D ..< (r * KVH + h + 1) * D]))
                }
                keys.append(kRow)
                values.append(vRow)
            }
            let scale = 1 / Float(D).squareRoot()
            var out = [Float](repeating: 0, count: rows * H * D)
            for r in 0..<rows {
                let pos = start + r
                for h in 0..<H {
                    let base = r * H * 2 * D + h * 2 * D
                    var q = Array(qGate[base ..< base + D])
                    Self.headRMSNorm(&q, weight: qNorm, eps: eps)
                    Self.ropeNeox(&q, dims: ropeDims, position: pos, base: freqBase)
                    let kvh = h / (H / KVH)
                    var m = -Float.infinity
                    for t in 0...pos {
                        var s: Float = 0
                        for d in 0..<D { s += q[d] * keys[t][kvh * D + d] }
                        m = max(m, s * scale)
                    }
                    var acc = [Float](repeating: 0, count: D)
                    var sum: Float = 0
                    for t in 0...pos {
                        var s: Float = 0
                        for d in 0..<D { s += q[d] * keys[t][kvh * D + d] }
                        let e = exp(s * scale - m)
                        sum += e
                        for d in 0..<D { acc[d] += e * values[t][kvh * D + d] }
                    }
                    for d in 0..<D {
                        out[r * H * D + h * D + d] = (acc[d] / sum) * Native.ReferenceCPU.sigmoid(qGate[base + D + d])
                    }
                }
            }
            return out
        }
    }
}
