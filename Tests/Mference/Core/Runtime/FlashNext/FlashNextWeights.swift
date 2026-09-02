import Foundation
import Metal
@testable import Mference
@testable import MferenceRepackCore

/// Reads every weight the Flash-Next reference forward needs out of a loaded
/// `Model`, as float32.
///
/// All access goes through the real loader: `Model.resident` /
/// `Model.normWeight` (which applies the family's `(1 + w)` bake),
/// `Model.routedExpert` plus `packedExpertsLayout`, and `PleRowPool`. Nothing
/// here reads the source checkpoint.
///
/// Dequantization dispatches on the stored dtype, so the same runner works
/// against a BF16-passthrough install and an INT4 affine group-64 one. INT4
/// decode is `Quantization.dequantizeInt4Affine`, the runtime's own reference,
/// which `Int4AffineEncoderParityTests` locks bit-for-bit to the repacker's
/// `Int4AffineEncoder`.
///
/// `int4RoundTrip` is the honest answer to the goldens' dtype caveat: with it
/// on, every tensor the production planner *would* quantize is pushed through
/// `Int4AffineEncoder.encodeTensor` and decoded back, so a parity run reports
/// what an INT4 install would actually cost.
final class FlashNextWeights {

    let model: Model
    let config: ArchConfig
    let int4RoundTrip: Bool
    private var cache: [String: [Float]] = [:]
    private let trunk: String

    init(model: Model, int4RoundTrip: Bool = false) {
        self.model = model
        self.config = model.config
        self.int4RoundTrip = int4RoundTrip
        self.trunk = "model.language_model."
    }

    // MARK: - Raw reads

    /// Widen a BF16 `TensorView` to float32.
    static func readBF16(_ view: TensorView) -> [Float] {
        let count = Int(view.length) / MemoryLayout<UInt16>.stride
        let base = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Quantization.bf16ToFloat(base[$0]) }
    }

    /// Decode any stored dtype to float32, row-major.
    static func read(_ view: TensorView) -> [Float] {
        switch view.dtype {
        case 1:
            return readBF16(view)
        case 3:
            let count = Int(view.length) / MemoryLayout<Float>.stride
            let base = view.buffer.contents().advanced(by: Int(view.offset))
                .assumingMemoryBound(to: Float.self)
            return (0..<count).map { base[$0] }
        case 0:
            let packedCount = Int(view.length)
            let companion = Int(view.scaleLength) / MemoryLayout<UInt16>.stride
            let raw = view.buffer.contents()
            let packed = (0..<packedCount).map {
                raw.advanced(by: Int(view.offset) + $0)
                    .assumingMemoryBound(to: UInt8.self).pointee
            }
            let scales = (0..<companion).map {
                raw.advanced(by: Int(view.scaleOffset) + $0 * 2)
                    .assumingMemoryBound(to: UInt16.self).pointee
            }
            let biases = (0..<companion).map {
                raw.advanced(by: Int(view.biasOffset) + $0 * 2)
                    .assumingMemoryBound(to: UInt16.self).pointee
            }
            return Quantization.dequantizeInt4Affine(
                Quantization.Int4AffineRow(packed: packed, scales: scales,
                                           biases: biases),
                n: packedCount * 2)
        default:
            preconditionFailure("unsupported tensor dtype \(view.dtype)")
        }
    }

    /// Force `values` (laid out as rows of `rowLength`) through the exact INT4
    /// affine group-64 encode/decode a production install would apply. A row
    /// length the group size does not divide is not quantizable, and the
    /// planner leaves such tensors BF16 — so this returns them unchanged.
    static func int4RoundTripped(_ values: [Float], rowLength: Int) -> [Float] {
        guard rowLength > 0, rowLength % Int4AffineEncoder.groupSize == 0 else {
            return values
        }
        let encoded = values.withUnsafeBufferPointer {
            Int4AffineEncoder.encodeTensor($0, rowLength: rowLength)
        }
        return Quantization.dequantizeInt4Affine(
            Quantization.Int4AffineRow(packed: encoded.packed,
                                       scales: encoded.scales,
                                       biases: encoded.biases),
            n: values.count)
    }

    // MARK: - Cached accessors

    /// A projection matrix `[rows, cols]`, row-major, float32.
    func matrix(_ view: TensorView, key: String, rows: Int, cols: Int) -> [Float] {
        if let hit = cache[key] { return hit }
        var values = Self.read(view)
        precondition(values.count == rows * cols,
                     "\(key): \(values.count) values, expected \(rows)x\(cols)")
        if int4RoundTrip, view.dtype == 1 {
            values = Self.int4RoundTripped(values, rowLength: cols)
        }
        cache[key] = values
        return values
    }

    private func resident(_ name: String, rows: Int, cols: Int) throws -> [Float] {
        matrix(try model.resident(name: name), key: name, rows: rows, cols: cols)
    }

    /// A norm weight, with the family's `(1 + w)` bake already applied where it
    /// belongs. Norms are never quantized by the planner, so the round-trip
    /// mode leaves them alone.
    private func norm(_ name: String) throws -> [Float] {
        if let hit = cache[name] { return hit }
        let values = Self.read(try model.normWeight(name: name))
        cache[name] = values
        return values
    }

    private func layer(_ L: Int) -> String { "\(trunk)layers.\(L)." }

    // MARK: - Global

    var hidden: Int { config.hiddenSize }
    var bundle: Int { config.flashNext.hcCount * config.hiddenSize }

    func embedding() throws -> [Float] {
        try resident("\(trunk)embed_tokens.weight",
                     rows: config.vocabSize, cols: hidden)
    }

    func lmHead() throws -> [Float] {
        try resident("lm_head.weight", rows: config.vocabSize, cols: hidden)
    }

    // MARK: - Hyper connections

    struct GatedResidualWeights {
        let norm: [Float]        // [bundle]
        let mixDown: [Float]     // [lowRank, bundle]
        let mixUp: [Float]       // [bundle, lowRank]
        let inject: [Float]?     // [hcCount, bundle]; nil for the global mixer
    }

    func hyperConnection(site: Model.HyperConnectionSite, layer L: Int) throws
        -> GatedResidualWeights {
        let prefix = layer(L) + site.rawValue + "."
        let rank = config.flashNext.hcLowRank
        return GatedResidualWeights(
            norm: try norm(prefix + "hc_norm.weight"),
            mixDown: try resident(prefix + "input_mix_weight_down.weight",
                                  rows: rank, cols: bundle),
            mixUp: try resident(prefix + "input_mix_weight_up.weight",
                                rows: bundle, cols: rank),
            inject: try resident(prefix + "block_inject_weight.weight",
                                 rows: config.flashNext.hcCount, cols: bundle))
    }

    func globalMixer() throws -> GatedResidualWeights {
        let prefix = "\(trunk)hyper_connection_mixer."
        let rank = config.flashNext.hcLowRank
        return GatedResidualWeights(
            norm: try norm(prefix + "hc_norm.weight"),
            mixDown: try resident(prefix + "input_mix_weight_down.weight",
                                  rows: rank, cols: bundle),
            mixUp: try resident(prefix + "input_mix_weight_up.weight",
                                rows: bundle, cols: rank),
            inject: nil)
    }

    // MARK: - Full attention

    struct AttentionWeights {
        let q: [Float]           // [2 * numHeads * headDim, hidden]
        let k: [Float]           // [numKVHeads * headDim, hidden]
        let v: [Float]           // [numKVHeads * headDim, hidden]
        let o: [Float]           // [hidden, numHeads * headDim]
        let qNorm: [Float]       // [headDim]
        let kNorm: [Float]       // [headDim]
        let indexerQK: [Float]   // [(nHeads + nKV) * indexerHeadDim, hidden]
        let indexerQNorm: [Float]
        let indexerKNorm: [Float]
    }

    func attention(layer L: Int) throws -> AttentionWeights {
        let p = layer(L) + "self_attn."
        let qDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let fn = config.flashNext
        let indexRows = (fn.indexerNumHeads + fn.indexerNumKVHeads) * fn.indexerHeadDim
        return AttentionWeights(
            q: try resident(p + "q_proj.weight", rows: 2 * qDim, cols: hidden),
            k: try resident(p + "k_proj.weight", rows: kvDim, cols: hidden),
            v: try resident(p + "v_proj.weight", rows: kvDim, cols: hidden),
            o: try resident(p + "o_proj.weight", rows: hidden, cols: qDim),
            qNorm: try norm(p + "q_norm.weight"),
            kNorm: try norm(p + "k_norm.weight"),
            indexerQK: try resident(p + "indexer.index_qk_proj.weight",
                                    rows: indexRows, cols: hidden),
            indexerQNorm: try norm(p + "indexer.q_layernorm.weight"),
            indexerKNorm: try norm(p + "indexer.k_layernorm.weight"))
    }

    // MARK: - Gated DeltaNet

    struct GDNWeights {
        let qkv: [Float]         // [2 * Hk * Dk + Hv * Dv, hidden]
        let z: [Float]           // [Hv * Dv, hidden]
        let a: [Float]           // [Hv, hidden]
        let b: [Float]           // [Hv, hidden]
        let conv: [Float]        // [qkvDim, kernel]
        let aLog: [Float]        // [Hv]
        let dtBias: [Float]      // [Hv]
        let norm: [Float]        // [Dv] — ones-centered, NOT baked
        let out: [Float]         // [hidden, Hv * Dv]
    }

    func gdn(layer L: Int) throws -> GDNWeights {
        let p = layer(L) + "linear_attn."
        let la = config.linearAttention
        let qkvDim = 2 * la.numKHeads * la.keyHeadDim + la.numVHeads * la.valueHeadDim
        let valueDim = la.numVHeads * la.valueHeadDim
        return GDNWeights(
            qkv: try resident(p + "in_proj_qkv.weight", rows: qkvDim, cols: hidden),
            z: try resident(p + "in_proj_z.weight", rows: valueDim, cols: hidden),
            a: try resident(p + "in_proj_a.weight", rows: la.numVHeads, cols: hidden),
            b: try resident(p + "in_proj_b.weight", rows: la.numVHeads, cols: hidden),
            conv: try resident(p + "conv1d.weight",
                               rows: qkvDim, cols: la.convKernelSize),
            aLog: try norm(p + "A_log"),
            dtBias: try norm(p + "dt_bias"),
            norm: try norm(p + "norm.weight"),
            out: try resident(p + "out_proj.weight", rows: hidden, cols: valueDim))
    }

    // MARK: - MoE

    struct MoEWeights {
        let router: [Float]              // [numExperts, hidden]
        let sharedGate: [Float]          // [1, hidden]
        let sharedGateProj: [Float]      // [inter, hidden]
        let sharedUpProj: [Float]        // [inter, hidden]
        let sharedDownProj: [Float]      // [hidden, inter]
    }

    func moe(layer L: Int) throws -> MoEWeights {
        let p = layer(L) + "mlp."
        let inter = config.intermediateSize
        return MoEWeights(
            router: try resident(p + "gate.weight",
                                 rows: config.numExperts, cols: hidden),
            sharedGate: try resident(p + "shared_expert_gate.weight",
                                     rows: 1, cols: hidden),
            sharedGateProj: try resident(p + "shared_expert.gate_proj.weight",
                                         rows: inter, cols: hidden),
            sharedUpProj: try resident(p + "shared_expert.up_proj.weight",
                                       rows: inter, cols: hidden),
            sharedDownProj: try resident(p + "shared_expert.down_proj.weight",
                                         rows: hidden, cols: inter))
    }

    struct ExpertWeights {
        let gate: [Float]   // [moeIntermediate, hidden]
        let up: [Float]     // [moeIntermediate, hidden]
        let down: [Float]   // [hidden, moeIntermediate]
    }

    /// One routed expert, read through the real streaming backend and the real
    /// `packed_experts/layout.json` sub-tensor offsets.
    func expert(layer L: Int, expert E: Int) throws -> ExpertWeights {
        let key = "expert:\(L):\(E)"
        let inter = config.moeIntermediateSize
        if let hit = cache[key] {
            let half = inter * hidden
            return ExpertWeights(gate: Array(hit[0..<half]),
                                 up: Array(hit[half..<(2 * half)]),
                                 down: Array(hit[(2 * half)...]))
        }
        let blob = try model.routedExpert(layer: L, expert: E)
        let entry = model.packedExpertsLayout.expert(layer: L, expert: E)
        func slice(_ role: String, rows: Int, cols: Int) -> [Float] {
            guard let sub = entry.subTensors[role] else {
                preconditionFailure("layout has no \(role) for L\(L) E\(E)")
            }
            let view = TensorView(
                buffer: blob.buffer,
                offset: blob.offset + sub.offset,
                length: sub.size,
                scaleOffset: entry.subTensors[role + "_scales"]
                    .map { blob.offset + $0.offset } ?? 0,
                scaleLength: entry.subTensors[role + "_scales"]?.size ?? 0,
                biasOffset: entry.subTensors[role + "_biases"]
                    .map { blob.offset + $0.offset } ?? 0,
                biasLength: entry.subTensors[role + "_biases"]?.size ?? 0,
                shape: (UInt32(rows), UInt32(cols), 0, 0),
                // A blob with no companion slices is stored dense BF16.
                dtype: entry.subTensors[role + "_scales"] == nil ? 1 : 0)
            var values = Self.read(view)
            precondition(values.count == rows * cols)
            if int4RoundTrip, entry.subTensors[role + "_scales"] == nil {
                values = Self.int4RoundTripped(values, rowLength: cols)
            }
            return values
        }
        let gate = slice("gate", rows: inter, cols: hidden)
        let up = slice("up", rows: inter, cols: hidden)
        let down = slice("down", rows: hidden, cols: inter)
        cache[key] = gate + up + down
        return ExpertWeights(gate: gate, up: up, down: down)
    }

    // MARK: - PLE

    struct PLEWeights {
        let keyProj: [Float]        // [bundle, hidden]
        let valueProj: [Float]      // [hidden, hidden]
        let conv: [Float]           // [bundle, kernel]
        let normConv: [Float]       // [bundle]
        let normKey: [Float]        // [bundle]
        let normQuery: [Float]      // [bundle]
        let multipliers: [Int64]    // [ngramSize]
        let headOffsets: [Int64]    // [ngramHeads]
        let headVocabSizes: [Int64] // [ngramHeads]
        let pool: PleRowPool
    }

    func ple(layer L: Int) throws -> PLEWeights {
        let p = layer(L) + "ple."
        return PLEWeights(
            keyProj: try resident(p + "key_proj.weight", rows: bundle, cols: hidden),
            valueProj: try resident(p + "value_proj.weight",
                                    rows: hidden, cols: hidden),
            conv: try resident(p + "conv1d.weight",
                               rows: bundle, cols: config.flashNext.pleConvKernelSize),
            normConv: try norm(p + "norm_conv.weight"),
            normKey: try norm(p + "norm_key.weight"),
            normQuery: try norm(p + "norm_query.weight"),
            multipliers: try model.pleLayerMultipliers(layer: L),
            headOffsets: try model.pleNgramHeadOffsets(layer: L),
            headVocabSizes: try model.pleNgramHeadVocabSizes(layer: L),
            pool: try model.openPleRowPool(layer: L))
    }
}
