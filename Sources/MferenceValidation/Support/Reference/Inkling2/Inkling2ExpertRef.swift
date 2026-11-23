import Foundation
import Accelerate

/// What happens to an expert product's input before the multiply. The
/// default is `.none`; which mode the shipped model uses is decided by K0's
/// measurement against the base reference.
public enum Inkling2ExpertInputQuantization: Sendable, Equatable {
    /// The input is used as given (the transformers reference, mlx-lm).
    case none
    /// ggml's CPU product (`vec_dot_type = Q8_0` for NVFP4 and Q8_0 weights):
    /// each 32-value block of the input is rounded through Q8_0. Stock
    /// llama.cpp b12635 runs the NVFP4 experts on the CPU (Metal has no NVFP4
    /// kernel there), so this is what the base reference does for layers 0-19.
    case q8_0Blocks
    /// `hf_quant_config.json` `required_activation_quantization`: NVFP4 with
    /// a per-token outer scale (`amax / 1536`) and the four-over-six block
    /// scale rule for NVFP4 experts; MXFP8 (E4M3 values, UE8M0 scale per 32)
    /// for the vendor's 8-bit experts. Routing weights apply before the down
    /// projection's quantizer, as in the vendor's serving code.
    case vendorRequired
}

/// Byte views of one expert, laid out as in the GGUF: the merged gate-up
/// matrix (`2F` rows of `D` weights: rows `0..<F` gate, `F..<2F` up), then
/// the down matrix (`D` rows of `F` weights). `gateUpScale` and `downScale`
/// are the per-expert `.scale` tensors of an NVFP4 layer (1 for Q8_0 layers).
public struct Inkling2ExpertWeights {
    public let encoding: Inkling2ExpertEncoding
    public let d: Int
    public let f: Int
    public let gateUp: [UInt8]
    public let down: [UInt8]
    public let gateUpScale: Float
    public let downScale: Float

    public init(encoding: Inkling2ExpertEncoding, d: Int, f: Int,
                gateUp: [UInt8], down: [UInt8], gateUpScale: Float, downScale: Float) {
        precondition(gateUp.count == 2 * f * encoding.rowBytes(k: d))
        precondition(down.count == d * encoding.rowBytes(k: f))
        self.encoding = encoding
        self.d = d
        self.f = f
        self.gateUp = gateUp
        self.down = down
        self.gateUpScale = gateUpScale
        self.downScale = downScale
    }

    /// Bytes of one expert blob: gate-up then down, unpadded.
    public var blobBytes: Int { gateUp.count + down.count }

    /// Decoded gate-up rows, row-major `[2F][D]`, without the per-expert scale.
    public func decodedGateUp() -> [Float] {
        let rowBytes = encoding.rowBytes(k: d)
        return gateUp.withUnsafeBytes { raw in
            var out = [Float](); out.reserveCapacity(2 * f * d)
            for row in 0..<(2 * f) {
                out += encoding.dequantize(UnsafeRawBufferPointer(rebasing: raw[(row * rowBytes)...]), k: d)
            }
            return out
        }
    }

    /// Decoded down rows, row-major `[D][F]`, without the per-expert scale.
    public func decodedDown() -> [Float] {
        let rowBytes = encoding.rowBytes(k: f)
        return down.withUnsafeBytes { raw in
            var out = [Float](); out.reserveCapacity(d * f)
            for row in 0..<d {
                out += encoding.dequantize(UnsafeRawBufferPointer(rebasing: raw[(row * rowBytes)...]), k: f)
            }
            return out
        }
    }
}

/// FP32 reference for one routed expert of Inkling 2 Mini and for the
/// routed sum, with llama.cpp b12635's semantics (`build_moe_ffn` with
/// `LLM_FFN_SITU`, merged gate-up, per-expert `.scale` applied after the
/// product, `build_lora_mm_id`):
///
///     gu   = gateUpScale * (W13 . x)        FP32, scale before any narrowing
///     act  = situ(gu[0..<F], gu[F..<2F])
///     y    = downScale * (W2 . act)
///     situ(g, u) = 4 tanh(g / 4) * sigmoid(g) * 25 tanh(u / 25)
///
/// Products accumulate in FP32 over weights decoded exactly. The kernel under
/// test decodes into half tiles and accumulates in simdgroup matrices, so the
/// summation order differs.
public enum Inkling2ExpertRef {
    /// `activation.situ_beta` and `activation.situ_linear_beta` in the GGUF.
    public static let situBeta: Float = 4
    public static let situLinearBeta: Float = 25

    /// ggml's op order: `beta * tanh(g * (1 / beta)) * sigmoid(g) * lb * tanh(u * (1 / lb))`.
    public static func situ(gate g: Float, up u: Float,
                            beta: Float = situBeta, linearBeta: Float = situLinearBeta) -> Float {
        let capped = tanhf(g * (1 / beta)) * beta
        let sigmoid = 1 / (1 + expf(-g))
        let linear = linearBeta > 0 ? tanhf(u * (1 / linearBeta)) * linearBeta : u
        return capped * sigmoid * linear
    }

    /// `y = W x` for row-major `w` of `rows x k`, FP32 (`vDSP_dotpr`).
    public static func matvec(_ w: [Float], rows: Int, k: Int, _ x: [Float]) -> [Float] {
        precondition(w.count == rows * k && x.count == k)
        var y = [Float](repeating: 0, count: rows)
        w.withUnsafeBufferPointer { pw in
            x.withUnsafeBufferPointer { px in
                for r in 0..<rows {
                    var dot: Float = 0
                    vDSP_dotpr(pw.baseAddress! + r * k, 1, px.baseAddress!, 1, &dot, vDSP_Length(k))
                    y[r] = dot
                }
            }
        }
        return y
    }

    /// The input as the product sees it under `mode`, already rescaled to the
    /// input's own units.
    public static func quantizedInput(_ x: [Float], encoding: Inkling2ExpertEncoding,
                                      mode: Inkling2ExpertInputQuantization) -> [Float] {
        switch mode {
        case .none: return x
        case .q8_0Blocks: return Inkling2Q8_0.roundTrip(x)
        case .vendorRequired:
            switch encoding {
            case .nvfp4:
                let q = Inkling2ActivationQuant.nvfp4FourOverSix(x)
                let outer = q.rowMax / Float(Inkling2ActivationQuant.nvfp4MaxProduct)
                return q.inner.map { $0 * outer }
            case .q8_0:
                return Inkling2ActivationQuant.mxfp8(x)
            }
        }
    }

    /// The gate-up product and activation: `F` values.
    public static func activation(_ expert: Inkling2ExpertWeights, decodedGateUp: [Float]? = nil,
                                  x: [Float], mode: Inkling2ExpertInputQuantization = .none) -> [Float] {
        precondition(x.count == expert.d)
        let w = decodedGateUp ?? expert.decodedGateUp()
        let input = quantizedInput(x, encoding: expert.encoding, mode: mode)
        let gu = matvec(w, rows: 2 * expert.f, k: expert.d, input)
        return (0..<expert.f).map { i in
            situ(gate: gu[i] * expert.gateUpScale, up: gu[expert.f + i] * expert.gateUpScale)
        }
    }

    /// The down product of an activation, `D` values, with `preScale`
    /// multiplied into the activation before the down input's quantizer.
    public static func down(_ expert: Inkling2ExpertWeights, decodedDown: [Float]? = nil,
                            activation act: [Float], preScale: Float = 1,
                            mode: Inkling2ExpertInputQuantization = .none) -> [Float] {
        precondition(act.count == expert.f)
        let w = decodedDown ?? expert.decodedDown()
        let scaled = preScale == 1 ? act : act.map { $0 * preScale }
        let input = quantizedInput(scaled, encoding: expert.encoding, mode: mode)
        return matvec(w, rows: expert.d, k: expert.f, input).map { $0 * expert.downScale }
    }

    /// One expert: `down(situ(gate-up(x)))`, unweighted.
    public static func forward(_ expert: Inkling2ExpertWeights, x: [Float],
                               mode: Inkling2ExpertInputQuantization = .none) -> [Float] {
        down(expert, activation: activation(expert, x: x, mode: mode), mode: mode)
    }

    /// The routed sum for one token: `sum_r weights[r] * expert_{ids[r]}(x)`.
    /// Under `.vendorRequired` the weight multiplies the activation before
    /// the down quantizer; otherwise it multiplies the expert's output, as in
    /// ggml (`experts * weights`). Ranks add in order.
    public static func routedSum(experts: (Int) -> Inkling2ExpertWeights, ids: [Int], weights: [Float],
                                 x: [Float], mode: Inkling2ExpertInputQuantization = .none) -> [Float] {
        precondition(ids.count == weights.count)
        var out: [Float]? = nil
        for (rank, id) in ids.enumerated() {
            let expert = experts(id)
            let act = activation(expert, x: x, mode: mode)
            let y: [Float]
            if mode == .vendorRequired {
                y = down(expert, activation: act, preScale: weights[rank], mode: mode)
            } else {
                y = down(expert, activation: act, mode: mode).map { $0 * weights[rank] }
            }
            if out == nil { out = y } else { for i in 0..<y.count { out![i] += y[i] } }
        }
        return out ?? [Float](repeating: 0, count: x.count)
    }
}

/// The vendor's expert-input quantizers, from Ollama's port of the vendor's
/// serving reference (`mlx/nvfp4_activation.go`,
/// `mlxrunner/model/inkling2/experts.go`): FP32 arithmetic with no fused
/// multiply-add, E4M3 and E2M1 rounded to nearest, ties to even.
public enum Inkling2ActivationQuant {
    /// The vendor's outer-scale denominator for activations (6 x 256).
    public static let nvfp4MaxProduct = 1536

    public struct NVFP4Row {
        /// E2M1 value times its E4M3 block scale, per element.
        public let inner: [Float]
        /// The row's absolute maximum; `inner * rowMax / 1536` is the input.
        public let rowMax: Float
        /// Per 16-value block: whether the 4-range scale won.
        public let usedFourRange: [Bool]
    }

    /// Nonnegative E4M3 (FN) round to nearest even, no saturation (callers clamp to 448).
    public static func roundE4M3(_ x: Float) -> Float {
        let exponent = (x.bitPattern >> 23) & 0xFF
        let spacing = Float(bitPattern: (max(exponent, 121) - 3) << 23)
        return (x / spacing).rounded(.toNearestOrEven) * spacing
    }

    /// E2M1 round to nearest even, signed.
    public static func roundE2M1(_ x: Float) -> Float {
        let m = abs(x)
        var q: Float = m > 0.25 ? 0.5 : 0
        if m >= 0.75 { q = 1 }
        if m > 1.25 { q = 1.5 }
        if m >= 1.75 { q = 2 }
        if m > 2.5 { q = 3 }
        if m >= 3.5 { q = 4 }
        if m > 5 { q = 6 }
        return x < 0 ? -q : q
    }

    /// NVFP4 four-over-six: per 16-value block, try the scales that map the
    /// block maximum to 6 and to 4, keep the one with the smaller total
    /// absolute error (a tie keeps 6).
    public static func nvfp4FourOverSix(_ x: [Float], maxProduct: Int = nvfp4MaxProduct) -> NVFP4Row {
        precondition(x.count % 16 == 0)
        var rowMax: Float = 0
        for v in x { rowMax = max(rowMax, abs(v)) }
        let normalizer = min(Float(maxProduct) / rowMax, .greatestFiniteMagnitude)
        let decoder = 1 / normalizer
        let outer = rowMax * (1 / Float(maxProduct))
        var inner = [Float](repeating: 0, count: x.count)
        var four = [Bool](repeating: false, count: x.count / 16)
        for b in 0..<(x.count / 16) {
            var blockMax: Float = 0
            for i in 0..<16 { blockMax = max(blockMax, abs(x[b * 16 + i])) }
            let scaleA = roundE4M3(min(blockMax * (normalizer * (1 / Float(6))), 448))
            let scaleB = roundE4M3(min(blockMax * (normalizer * 0.25), 448))
            let invA = min(1 / (scaleA * decoder), .greatestFiniteMagnitude)
            let invB = min(1 / (scaleB * decoder), .greatestFiniteMagnitude)
            var candA = [Float](repeating: 0, count: 16), candB = candA
            var errorA: Float = 0, errorB: Float = 0
            for i in 0..<16 {
                let v = x[b * 16 + i]
                candA[i] = roundE2M1(v * invA) * scaleA
                candB[i] = roundE2M1(v * invB) * scaleB
                errorA += abs(v - candA[i] * outer)
                errorB += abs(v - candB[i] * outer)
            }
            four[b] = errorB < errorA
            for i in 0..<16 { inner[b * 16 + i] = four[b] ? candB[i] : candA[i] }
        }
        return NVFP4Row(inner: inner, rowMax: rowMax, usedFourRange: four)
    }

    /// Signed E4M3 (FN) value nearest to `x` (ties to even), `|x| <= 448`.
    public static func roundE4M3Signed(_ x: Float) -> Float {
        let m = roundE4M3(abs(x))
        return x < 0 ? -m : m
    }

    /// MXFP8 (E4M3 values, a power-of-two UE8M0 scale per 32 values): the
    /// scale is `blockMax / 448` rounded up to a power of two.
    public static func mxfp8(_ x: [Float]) -> [Float] {
        precondition(x.count % 32 == 0)
        var out = [Float](repeating: 0, count: x.count)
        let minimum = Float(bitPattern: 0x0040_0000)  // 2^-127
        for b in 0..<(x.count / 32) {
            var blockMax: Float = 0
            for i in 0..<32 { blockMax = max(blockMax, abs(x[b * 32 + i])) }
            let raw = blockMax * (1 / Float(448))
            var exponent = Int32(bitPattern: raw.bitPattern) >> 23
            if raw.bitPattern & 0x7F_FFFF != 0 { exponent += 1 }
            exponent = min(exponent, 254)
            if !(raw > minimum) { exponent = 0 }
            let encoded = Float(bitPattern: UInt32(exponent) << 23)
            let scale = max(encoded, minimum)
            let inverse = exponent == 0 ? Float(bitPattern: 0x7F00_0000) : 1 / scale
            for i in 0..<32 {
                let scaled = min(max(x[b * 32 + i] * inverse, -448), 448)
                out[b * 32 + i] = roundE4M3Signed(scaled) * scale
            }
        }
        return out
    }
}

/// FP32 reference for the router of one layer visit (`build_moe_ffn` with
/// sigmoid gating, `exp_probs_b` selection bias, normalized weights and
/// `expert_weights_scale` 8).
public enum Inkling2RouterRef {
    public struct Route: Equatable {
        public let ids: [Int]
        public let weights: [Float]
    }

    public static let routeScale: Float = 8
    /// ggml clamps the weight sum to the smallest normal FP16.
    public static let minimumWeightSum: Float = 6.103515625e-5

    /// Logits `W x` for the F32 router matrix `[experts][d]`.
    public static func logits(weight: [Float], experts: Int, x: [Float]) -> [Float] {
        Inkling2ExpertRef.matvec(weight, rows: experts, k: x.count, x)
    }

    /// Top-`k` by `sigmoid(logit) + bias` (a tie takes the lower expert),
    /// weights from the unbiased sigmoid, normalized to sum 1, times 8.
    public static func route(logits: [Float], selectionBias: [Float], topK: Int = 8,
                             routeScale: Float = routeScale) -> Route {
        precondition(logits.count == selectionBias.count && topK <= logits.count)
        let probs = logits.map { 1 / (1 + expf(-$0)) }
        let selection = zip(probs, selectionBias).map { $0 + $1 }
        let ids = selection.indices.sorted {
            selection[$0] != selection[$1] ? selection[$0] > selection[$1] : $0 < $1
        }.prefix(topK).map { $0 }
        let chosen = ids.map { probs[$0] }
        let sum = max(chosen.reduce(0, +), minimumWeightSum)
        return Route(ids: ids, weights: chosen.map { $0 / sum * routeScale })
    }
}
