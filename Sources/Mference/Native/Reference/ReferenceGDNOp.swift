import Foundation
import Metal

extension Native {
    /// The reference `GDNOp`. `ref_gdn_scan` gives one thread to each state
    /// row `S[h][j][:]`; it walks the step's rows in order, recomputing the
    /// conv1d, SiLU and L2 norm of its q/k head (`h % groupCount`) for each
    /// row, then applies the delta rule exactly as ggml's
    /// `gated_delta_net` CPU op. `ref_gdn_gated_norm` then applies the
    /// per-head RMSNorm and `× silu(z)` in place on the output.
    public final class ReferenceGDNOp: GDNOp {
        let context: MetalContext
        private var config: Qwen35Config?
        private var layout: RecurrentLayout?
        private var scan: MTLComputePipelineState?
        private var gatedNorm: MTLComputePipelineState?

        public init(context: MetalContext) {
            self.context = context
        }

        public func prepare(config: Qwen35Config, layout: RecurrentLayout) throws {
            guard layout.convKernel == config.ssmConvKernel, layout.convKernel >= 2,
                  layout.convChannels == config.ssmConvChannels,
                  layout.valueHeads == config.ssmTimeStepRank,
                  layout.keyDim == config.ssmStateSize, layout.valueDim == config.ssmStateSize,
                  layout.keyDim <= 256, config.ssmInnerSize == layout.valueHeads * layout.valueDim,
                  config.ssmGroupCount >= 1 else {
                throw NativeError.invalidShape("reference GDN: config and recurrent layout disagree")
            }
            scan = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_gdn_scan")
            gatedNorm = try context.modulePipeline(module: ReferenceMetal.module, name: "ref_gdn_gated_norm")
            self.config = config
            self.layout = layout
        }

        public func encodeDecode(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights, qkv: TensorRef,
                                 z: TensorRef, alpha: TensorRef, beta: TensorRef, stateIn: RecurrentStateRef,
                                 stateOut: RecurrentStateRef, output: TensorRef) throws {
            guard qkv.rows == 1 else { throw NativeError.invalidShape("GDN decode needs 1 row") }
            try encode(encoder, weights: weights, qkv: qkv, z: z, alpha: alpha, beta: beta, stateIn: stateIn,
                       stateOut: stateOut, arena: nil, output: output)
        }

        public func encodeVerify(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights, qkv: TensorRef,
                                 z: TensorRef, alpha: TensorRef, beta: TensorRef, stateIn: RecurrentStateRef,
                                 stateOut: RecurrentStateRef, arena: RecurrentVerifyArena,
                                 output: TensorRef) throws {
            guard (2...16).contains(qkv.rows) else { throw NativeError.invalidShape("GDN verify needs 2...16 rows") }
            guard arena.slots >= qkv.rows, arena.slotStride >= arena.layout.layerBytes,
                  arena.layout == stateIn.layout else {
                throw NativeError.invalidShape("verify arena has \(arena.slots) slots for \(qkv.rows) rows")
            }
            try encode(encoder, weights: weights, qkv: qkv, z: z, alpha: alpha, beta: beta, stateIn: stateIn,
                       stateOut: stateOut, arena: arena, output: output)
        }

        public func encodePrefill(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights, qkv: TensorRef,
                                  z: TensorRef, alpha: TensorRef, beta: TensorRef, stateIn: RecurrentStateRef,
                                  stateOut: RecurrentStateRef, output: TensorRef) throws {
            guard qkv.rows >= 1 else { throw NativeError.invalidShape("GDN prefill needs rows") }
            try encode(encoder, weights: weights, qkv: qkv, z: z, alpha: alpha, beta: beta, stateIn: stateIn,
                       stateOut: stateOut, arena: nil, output: output)
        }

        struct Params {
            var rows: UInt32
            var convKernel: UInt32
            var convChannels: UInt32
            var valueHeads: UInt32
            var groups: UInt32
            var dim: UInt32
            var eps: Float
            var qkvStride: UInt32
            var alphaStride: UInt32
            var betaStride: UInt32
            var outStride: UInt32
            var zStride: UInt32
            var ssmOffset: UInt32
            var writeArena: UInt32
            var arenaStride: UInt32
        }

        private func encode(_ encoder: MTLComputeCommandEncoder, weights: GDNLayerWeights, qkv: TensorRef,
                            z: TensorRef, alpha: TensorRef, beta: TensorRef, stateIn: RecurrentStateRef,
                            stateOut: RecurrentStateRef, arena: RecurrentVerifyArena?, output: TensorRef) throws {
            guard let config, let layout, let scan, let gatedNorm else {
                throw NativeError.resource("ReferenceGDNOp.prepare not called")
            }
            let rows = qkv.rows
            let H = layout.valueHeads, d = layout.valueDim
            try ReferenceMetal.checkF32(qkv, rows: rows, cols: layout.convChannels, "GDN qkv")
            try ReferenceMetal.checkF32(z, rows: rows, cols: H * d, "GDN z")
            try ReferenceMetal.checkF32(alpha, rows: rows, cols: H, "GDN alpha")
            try ReferenceMetal.checkF32(beta, rows: rows, cols: H, "GDN beta")
            try ReferenceMetal.checkF32(output, rows: rows, cols: H * d, "GDN output")
            guard stateIn.layout == layout, stateOut.layout == layout else {
                throw NativeError.invalidShape("GDN state layout differs from prepare")
            }
            guard !(stateIn.buffer === stateOut.buffer && abs(stateIn.offset - stateOut.offset) < layout.layerBytes) else {
                throw NativeError.invalidShape("GDN stateIn and stateOut overlap")
            }
            for w in [weights.conv1d, weights.dtBias, weights.a, weights.norm] where w.layout != .gguf {
                throw NativeError.unsupported("reference GDN reads \(w.name) only in the .gguf layout")
            }
            guard weights.conv1d.type == .f32, weights.conv1d.cols == layout.convKernel,
                  weights.conv1d.rows == layout.convChannels, weights.dtBias.type == .f32, weights.dtBias.cols == H,
                  weights.a.type == .f32, weights.a.cols == H, weights.norm.type == .f32, weights.norm.cols == d else {
                throw NativeError.invalidShape("GDN weights do not match the layout")
            }
            let params = Params(rows: UInt32(rows), convKernel: UInt32(layout.convKernel),
                                convChannels: UInt32(layout.convChannels), valueHeads: UInt32(H),
                                groups: UInt32(config.ssmGroupCount), dim: UInt32(d), eps: config.rmsNormEpsilon,
                                qkvStride: UInt32(qkv.rowStride), alphaStride: UInt32(alpha.rowStride),
                                betaStride: UInt32(beta.rowStride), outStride: UInt32(output.rowStride),
                                zStride: UInt32(z.rowStride), ssmOffset: UInt32(layout.ssmOffset),
                                writeArena: arena == nil ? 0 : 1, arenaStride: UInt32(arena?.slotStride ?? 0))
            encoder.setBuffer(qkv.buffer, offset: qkv.offset, index: 0)
            encoder.setBuffer(alpha.buffer, offset: alpha.offset, index: 1)
            encoder.setBuffer(beta.buffer, offset: beta.offset, index: 2)
            encoder.setBuffer(weights.conv1d.buffer, offset: weights.conv1d.offset, index: 3)
            encoder.setBuffer(weights.dtBias.buffer, offset: weights.dtBias.offset, index: 4)
            encoder.setBuffer(weights.a.buffer, offset: weights.a.offset, index: 5)
            encoder.setBuffer(stateIn.buffer, offset: stateIn.offset, index: 6)
            encoder.setBuffer(stateOut.buffer, offset: stateOut.offset, index: 7)
            if let arena {
                encoder.setBuffer(arena.buffer, offset: arena.offset, index: 8)
            } else {
                encoder.setBuffer(stateOut.buffer, offset: stateOut.offset, index: 8)
            }
            encoder.setBuffer(output.buffer, offset: output.offset, index: 9)
            ReferenceMetal.setBytes(encoder, params, index: 10)
            ReferenceMetal.dispatch(encoder, scan, width: d, height: H)

            encoder.setBuffer(output.buffer, offset: output.offset, index: 0)
            encoder.setBuffer(z.buffer, offset: z.offset, index: 1)
            encoder.setBuffer(weights.norm.buffer, offset: weights.norm.offset, index: 2)
            ReferenceMetal.setBytes(encoder, params, index: 3)
            ReferenceMetal.dispatch(encoder, gatedNorm, width: H, height: rows)
        }
    }
}

extension Native.ReferenceCPU {
    /// CPU twin of the reference GDN over one layer's state.
    public struct GDNState: Equatable {
        /// `[convKernel - 1][convChannels]`, row 0 oldest.
        public var conv: [Float]
        /// `[valueHeads][valueDim][keyDim]`.
        public var ssm: [Float]
        public init(conv: [Float], ssm: [Float]) {
            self.conv = conv
            self.ssm = ssm
        }
    }

    public struct GDNShape {
        public var convKernel: Int
        public var convChannels: Int
        public var valueHeads: Int
        public var groups: Int
        public var dim: Int
        public var eps: Float
        public init(convKernel: Int, convChannels: Int, valueHeads: Int, groups: Int, dim: Int, eps: Float) {
            self.convKernel = convKernel
            self.convChannels = convChannels
            self.valueHeads = valueHeads
            self.groups = groups
            self.dim = dim
            self.eps = eps
        }
    }

    /// `rows` rows of the GDN core; returns the output and the state after
    /// each row (the last is the new state).
    public static func gdn(shape s: GDNShape, qkv: [Float], z: [Float], alpha: [Float], beta: [Float], rows: Int,
                           conv1d: [Float], dtBias: [Float], a: [Float], norm: [Float],
                           state: GDNState) -> (output: [Float], states: [GDNState]) {
        let C = s.convChannels, K = s.convKernel, H = s.valueHeads, d = s.dim, G = s.groups
        func input(_ c: Int, _ tt: Int) -> Float {
            tt < 0 ? state.conv[(K - 1 + tt) * C + c] : qkv[tt * C + c]
        }
        func conv(_ c: Int, _ r: Int) -> Float {
            var acc: Float = 0
            for k in 0..<K { acc += input(c, r - (K - 1) + k) * conv1d[c * K + k] }
            return silu(acc)
        }
        func l2(_ v: inout [Float]) {
            var sum: Float = 0
            for x in v { sum += x * x }
            let n = Float(v.count)
            let scale = 1 / (sum / n + s.eps / n).squareRoot()
            let post = 1 / n.squareRoot()
            for i in v.indices { v[i] = (v[i] * scale) * post }
        }
        let scale = 1 / Float(d).squareRoot()
        var S = state.ssm
        var out = [Float](repeating: 0, count: rows * H * d)
        var states: [GDNState] = []
        for r in 0..<rows {
            var qs: [[Float]] = []
            var ks: [[Float]] = []
            for g in 0..<G {
                var q = (0..<d).map { conv(g * d + $0, r) }
                var k = (0..<d).map { conv(G * d + g * d + $0, r) }
                l2(&q)
                l2(&k)
                qs.append(q)
                ks.append(k)
            }
            for h in 0..<H {
                let g = h % G
                let biased = alpha[r * H + h] + dtBias[h]
                let softplus = biased > 20 ? biased : log(1 + exp(biased))
                let decay = exp(softplus * a[h])
                let b = sigmoid(beta[r * H + h])
                for j in 0..<d {
                    let v = conv(2 * G * d + h * d + j, r)
                    let base = (h * d + j) * d
                    for i in 0..<d { S[base + i] = S[base + i] * decay }
                    var kv: Float = 0
                    for i in 0..<d { kv += S[base + i] * ks[g][i] }
                    let delta = (v - kv) * b
                    for i in 0..<d { S[base + i] = S[base + i] + ks[g][i] * delta }
                    var o: Float = 0
                    for i in 0..<d { o += S[base + i] * qs[g][i] }
                    out[r * H * d + h * d + j] = o * scale
                }
            }
            var newConv = [Float](repeating: 0, count: (K - 1) * C)
            for k in 0..<(K - 1) {
                for c in 0..<C { newConv[k * C + c] = input(c, r + 1 - (K - 1) + k) }
            }
            states.append(GDNState(conv: newConv, ssm: S))
        }
        for r in 0..<rows {
            for h in 0..<H {
                let base = r * H * d + h * d
                var sum: Float = 0
                for i in 0..<d { sum += out[base + i] * out[base + i] }
                let sc = 1 / (sum / Float(d) + s.eps).squareRoot()
                for i in 0..<d { out[base + i] = ((out[base + i] * sc) * norm[i]) * silu(z[base + i]) }
            }
        }
        return (out, states)
    }
}
