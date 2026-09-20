import Metal
@testable import Mference

/// FP32 semantic oracle for the one-layer draft, using the independently
/// transcribed HC/QSA/attention/MoE references. No production GPU encoders.
/// This composition is not an upstream full-MTP golden or acceptance gate.
final class FlashNextMTPReference {
    private let model: Model
    private let pool: PreadExpertStreamer
    private var resident: [String: [Float]] = [:]
    private var keys: [Float] = []
    private var values: [Float] = []
    private var indexKeys: [Float] = []
    private var position = 0

    init(model: Model, device: MTLDevice) throws {
        self.model = model
        let weights = try FlashNextMTPWeights(model: model)
        pool = try PreadExpertStreamer(layout: weights.expertLayout, device: device, slotCount: 1)
    }

    private func read(_ name: String, norm: Bool = false) throws -> [Float] {
        let key = (norm ? "norm:" : "raw:") + name
        if let cached = resident[key] { return cached }
        let view: TensorView
        if norm { view = try model.normWeight(name: name) }
        else { view = try model.resident(name: name) }
        let result = FlashNextWeights.read(view)
        resident[key] = result
        return result
    }

    private func hc(_ site: String, injection: Bool = true) throws -> FlashNextHyperConnectionReference.Weights {
        .init(norm: try read(site + ".hc_norm.weight", norm: true),
            mixDown: try read(site + ".input_mix_weight_down.weight"),
            mixUp: try read(site + ".input_mix_weight_up.weight"),
            inject: injection ? try read(site + ".block_inject_weight.weight") : nil)
    }

    private func expert(_ id: Int) throws -> FlashNextExpertReference.Expert {
        let cfg = model.config, blob = try pool.loadExpert(layer: 0, expert: id)
        let d = cfg.hiddenSize, f = cfg.moeIntermediateSize
        let bytes = UInt64(d * f / 2), companion = UInt64(d * f / 64 * 2)
        let projection = bytes + 2 * companion
        func matrix(_ number: UInt64, rows: Int, columns: Int) -> [Float] {
            let base = blob.offset + number * projection
            return FlashNextWeights.read(TensorView(buffer: blob.buffer, offset: base, length: bytes,
                scaleOffset: base + bytes, scaleLength: companion,
                biasOffset: base + bytes + companion, biasLength: companion,
                shape: (UInt32(rows), UInt32(columns), 0, 0), dtype: 0))
        }
        return .init(gate: matrix(0, rows: f, columns: d), up: matrix(1, rows: f, columns: d),
                     down: matrix(2, rows: d, columns: f))
    }

    func append(embedding: [Float], hidden: [Float]) throws -> (hidden: [Float], logits: [Float]) {
        let cfg = model.config, fn = cfg.flashNext
        let d = cfg.hiddenSize, bundle = cfg.residualStreamWidth
        let geometry = FlashNextHyperConnectionReference.Geometry(hidden: d, hcCount: fn.hcCount,
            lowRank: fn.hcLowRank, eps: 1e-6)
        let e = FlashNextIndexerReference.rmsNorm(embedding, offset: 0, count: d,
            weight: try read("mtp.pre_fc_norm_embedding.weight", norm: true), eps: 1e-6)
        let h = FlashNextIndexerReference.rmsNorm(hidden, offset: 0, count: bundle,
            weight: try read("mtp.pre_fc_norm_hidden.weight", norm: true), eps: 1e-6)
        let projectedE = FlashNextRouterReference.matVec(try read("mtp.fc_embedding.weight"), rows: d, cols: d, x: e)
        let hiddenFC = try read("mtp.fc_hidden.weight")
        var hyper: [Float] = []
        for stream in 0..<fn.hcCount {
            let input = Array(h[(stream * d)..<((stream + 1) * d)])
            let row = FlashNextRouterReference.matVec(hiddenFC, rows: d, cols: d, x: input)
            hyper += zip(projectedE, row).map { $0 + $1 }
        }
        let attentionMix = FlashNextHyperConnectionReference.gatedResidual(hyper,
            try hc("mtp.layers.0.attn_hyper_connection"), rows: 1, g: geometry)
        let prefix = "mtp.layers.0.self_attn."
        let rotary = Int(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor)
        let selected = FlashNextIndexerReference.run(x: attentionMix.mixed, hidden: d, rows: 1, startPosition: position,
            indexerQK: try read(prefix + "indexer.index_qk_proj.weight"),
            qNorm: try read(prefix + "indexer.q_layernorm.weight", norm: true),
            kNorm: try read(prefix + "indexer.k_layernorm.weight", norm: true), rawKeys: indexKeys,
            g: .init(numHeads: fn.indexerNumHeads, numKVHeads: fn.indexerNumKVHeads, headDim: fn.indexerHeadDim,
                compressRatio: fn.indexerCompressRatio, blockBudget: fn.indexerBlockBudget,
                rotaryDim: rotary, theta: Float(cfg.fullRopeTheta), eps: 1e-6))
        indexKeys = selected.rawKeys
        let attention = FlashNextAttentionReference.run(x: attentionMix.mixed, hidden: d, rows: 1, startPosition: position,
            w: .init(q: try read(prefix + "q_proj.weight"), k: try read(prefix + "k_proj.weight"),
                v: try read(prefix + "v_proj.weight"), o: try read(prefix + "o_proj.weight"),
                qNorm: try read(prefix + "q_norm.weight", norm: true), kNorm: try read(prefix + "k_norm.weight", norm: true)),
            selected: selected.selected, keys: keys, values: values, scale: 1 / Float(cfg.fullHeadDim).squareRoot(),
            g: .init(numHeads: cfg.numHeads, numKVHeads: cfg.numFullKVHeads, headDim: cfg.fullHeadDim,
                rotaryDim: rotary, theta: Float(cfg.fullRopeTheta), eps: 1e-6))
        keys = attention.keys
        values = attention.values
        hyper = FlashNextHyperConnectionReference.injectBlock(hyper, block: attention.out,
            inject: attentionMix.inject!, rows: 1, g: geometry)
        let mlpMix = FlashNextHyperConnectionReference.gatedResidual(hyper,
            try hc("mtp.layers.0.mlp_hyper_connection"), rows: 1, g: geometry)
        let mlp = "mtp.layers.0.mlp."
        let routeLogits = FlashNextRouterReference.matVec(try read(mlp + "gate.weight"),
            rows: cfg.numExperts, cols: d, x: mlpMix.mixed)
        let routes = FlashNextRouterReference.select(logits: routeLogits, k: cfg.topKExperts)
        let output = try FlashNextExpertReference.block(experts: routes.indices.map { try expert($0) },
            weights: routes.weights,
            shared: .init(gateRow: read(mlp + "shared_expert_gate.weight"),
                gateProj: read(mlp + "shared_expert.gate_proj.weight"),
                upProj: read(mlp + "shared_expert.up_proj.weight"), downProj: read(mlp + "shared_expert.down_proj.weight")),
            x: mlpMix.mixed, hidden: d, moeIntermediate: cfg.moeIntermediateSize, sharedIntermediate: cfg.intermediateSize)
        hyper = FlashNextHyperConnectionReference.injectBlock(hyper, block: output, inject: mlpMix.inject!, rows: 1, g: geometry)
        let headInput = FlashNextHyperConnectionReference.gatedResidual(hyper,
            try hc("mtp.hyper_connection_mixer", injection: false), rows: 1, g: geometry).mixed
        let logits = FlashNextRouterReference.matVec(try read("lm_head.weight"), rows: cfg.vocabSize, cols: d, x: headInput)
        position += 1
        return (hyper, logits)
    }
}
