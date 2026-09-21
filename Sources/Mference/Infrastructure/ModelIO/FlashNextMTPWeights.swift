import Foundation
import Metal

/// Validated native draft weights only. Construction neither enables MTP nor
/// allocates a KV cache/expert cache. The caller must retain this object and
/// open `expertLayout` with its own bounded or resident streamer.
///
/// This one-layer sidecar is not a trunk layer: it has no GDN or PLE and owns
/// a separate expert pool, whose canonical INT4 layout the repacker writes
/// without a layout.json. Do not infer its offsets from the trunk's layout.
struct FlashNextMTPWeights {
    let embeddingNorm: TensorView
    let hiddenNorm: TensorView
    let embeddingProjection: FlashNextWeightMatrix
    let hiddenProjection: FlashNextWeightMatrix
    let attentionHC: FlashNextHyperConnections.Weights
    let mlpHC: FlashNextHyperConnections.Weights
    let mixer: FlashNextHyperConnections.Weights
    let attention: FlashNextAttention.Weights
    let indexerProjection: FlashNextWeightMatrix
    let indexerQNorm: TensorView
    let indexerKNorm: TensorView
    let router: FlashNextWeightMatrix
    let sharedGate: FlashNextWeightMatrix
    let sharedGateProjection: FlashNextWeightMatrix
    let sharedUp: FlashNextWeightMatrix
    let sharedDown: FlashNextWeightMatrix
    let expertLayout: StreamLayout
    let expertOffsets: MoEExpertOffsets

    init(model: Model) throws {
        guard model.config.family == .qwen38flashnext,
              model.manifest.sidecars?["mtp"]?.carried == true,
              model.manifest.sidecars?["mtp"]?.tensorCount == 31 else {
            throw Self.invalid("requires the carried 31-source-tensor Flash-Next MTP sidecar")
        }
        let cfg = model.config
        let d = cfg.hiddenSize, bundle = cfg.residualStreamWidth
        let fn = cfg.flashNext
        func matrix(_ name: String, _ rows: Int, _ cols: Int) throws -> FlashNextWeightMatrix {
            let view = try model.resident(name: name)
            try Self.validate(view, name: name, rows: rows, columns: cols)
            return .from(view)
        }
        func norm(_ name: String, _ count: Int) throws -> TensorView {
            let view = try model.resident(name: name)
            try Self.validate(view, name: name, rows: count, columns: nil)
            return try model.normWeight(name: name)
        }
        func hc(_ prefix: String, inject: Bool) throws -> FlashNextHyperConnections.Weights {
            let weight = try norm(prefix + ".hc_norm.weight", bundle)
            return .init(norm: weight.buffer, normOffset: Int(weight.offset),
                mixDown: try matrix(prefix + ".input_mix_weight_down.weight", fn.hcLowRank, bundle),
                mixUp: try matrix(prefix + ".input_mix_weight_up.weight", bundle, fn.hcLowRank),
                inject: inject ? try matrix(prefix + ".block_inject_weight.weight", fn.hcCount, bundle) : nil)
        }
        embeddingNorm = try norm("mtp.pre_fc_norm_embedding.weight", d)
        hiddenNorm = try norm("mtp.pre_fc_norm_hidden.weight", bundle)
        embeddingProjection = try matrix("mtp.fc_embedding.weight", d, d)
        hiddenProjection = try matrix("mtp.fc_hidden.weight", d, d)
        attentionHC = try hc("mtp.layers.0.attn_hyper_connection", inject: true)
        mlpHC = try hc("mtp.layers.0.mlp_hyper_connection", inject: true)
        mixer = try hc("mtp.hyper_connection_mixer", inject: false)
        let attn = "mtp.layers.0.self_attn."
        let qNorm = try norm(attn + "q_norm.weight", cfg.fullHeadDim)
        let kNorm = try norm(attn + "k_norm.weight", cfg.fullHeadDim)
        let qDim = cfg.numHeads * cfg.fullHeadDim
        let kvDim = cfg.numFullKVHeads * cfg.fullHeadDim
        attention = .init(q: try matrix(attn + "q_proj.weight", 2 * qDim, d),
            k: try matrix(attn + "k_proj.weight", kvDim, d),
            v: try matrix(attn + "v_proj.weight", kvDim, d),
            o: try matrix(attn + "o_proj.weight", d, qDim),
            qNorm: qNorm.buffer, qNormOffset: Int(qNorm.offset),
            kNorm: kNorm.buffer, kNormOffset: Int(kNorm.offset))
        indexerProjection = try matrix(attn + "indexer.index_qk_proj.weight",
            (fn.indexerNumHeads + fn.indexerNumKVHeads) * fn.indexerHeadDim, d)
        indexerQNorm = try norm(attn + "indexer.q_layernorm.weight", fn.indexerHeadDim)
        indexerKNorm = try norm(attn + "indexer.k_layernorm.weight", fn.indexerHeadDim)
        let mlp = "mtp.layers.0.mlp."
        router = try matrix(mlp + "gate.weight", cfg.numExperts, d)
        sharedGate = try matrix(mlp + "shared_expert_gate.weight", 1, d)
        sharedGateProjection = try matrix(mlp + "shared_expert.gate_proj.weight", cfg.intermediateSize, d)
        sharedUp = try matrix(mlp + "shared_expert.up_proj.weight", cfg.intermediateSize, d)
        sharedDown = try matrix(mlp + "shared_expert.down_proj.weight", d, cfg.intermediateSize)
        (expertLayout, expertOffsets) = try Self.loadExpertLayout(model: model)
    }

    /// Validate before calling `.from`, which deliberately preconditions its
    /// trusted inputs. Malformed install metadata must throw, not trap or bind
    /// an incorrectly sized scale/bias slice to a GPU kernel.
    static func validate(_ view: TensorView, name: String, rows: Int, columns: Int?) throws {
        guard rows > 0, view.shape.0 == UInt32(rows),
              view.shape.1 == UInt32(columns ?? 0),
              view.shape.2 == 0, view.shape.3 == 0 else {
            throw invalid("\(name): unexpected shape")
        }
        func fits(_ offset: UInt64, _ length: UInt64) -> Bool {
            offset <= UInt64(view.buffer.length) && length <= UInt64(view.buffer.length) - offset
        }
        guard fits(view.offset, view.length), fits(view.scaleOffset, view.scaleLength),
              fits(view.biasOffset, view.biasLength) else {
            throw invalid("\(name): tensor slice is outside its buffer")
        }
        let elements = UInt64(rows) * UInt64(columns ?? 1)
        if view.dtype == 1 {
            guard view.length == elements * 2, view.scaleLength == 0, view.biasLength == 0 else {
                throw invalid("\(name): invalid BF16 payload")
            }
        } else if view.dtype == 0, let columns, columns > 0, columns % 64 == 0 {
            guard view.length == elements / 2 || view.length == elements,
                  view.scaleLength == elements / 64 * 2,
                  view.biasLength == elements / 64 * 2 else {
                throw invalid("\(name): expected INT4/INT8 group-64 with BF16 companions")
            }
        } else {
            throw invalid("\(name): unsupported dtype or quantized shape")
        }
    }

    private static func loadExpertLayout(model: Model) throws -> (StreamLayout, MoEExpertOffsets) {
        let pools = model.manifest.auxiliaryExpertPools?.filter { $0.name == "mtp" } ?? []
        guard pools.count == 1, let pool = pools.first,
              pool.directory == "packed_experts_mtp", pool.layers.count == 1,
              pool.layers[0].layer == 0, pool.layers[0].file == "layer_00.bin",
              pool.expertsPerLayer == model.config.numExperts else {
            throw invalid("expected one canonical layer-0 MTP expert pool")
        }
        let d = UInt64(model.config.hiddenSize), f = UInt64(model.config.moeIntermediateSize)
        guard d > 0, f > 0, d % 64 == 0, f % 64 == 0 else {
            throw invalid("expert dimensions must be positive multiples of 64")
        }
        let w = d * f / 2, aux = d * f / 64 * 2
        let projection = w + 2 * aux
        let payload = 3 * projection
        let stride = (payload + 16_383) / 16_384 * 16_384
        guard payload <= UInt64(UInt32.max), pool.expertStride == stride else {
            throw invalid("expert stride does not match the canonical INT4 group-64 layout")
        }
        let relative = pool.directory + "/" + pool.layers[0].file
        let url = model.directoryURL.appendingPathComponent(relative)
        let root = model.directoryURL.resolvingSymlinksInPath().path + "/"
        guard url.resolvingSymlinksInPath().path.hasPrefix(root) else {
            throw invalid("expert pool escapes the install directory")
        }
        let bytes = stride * UInt64(pool.expertsPerLayer)
        guard let entry = model.manifest.files[relative], entry.size == bytes else {
            throw invalid("expert pool is absent from the manifest or has the wrong size")
        }
        let actual = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber
        guard actual?.uint64Value == bytes else { throw invalid("expert pool file size mismatch") }
        if model.integrityPolicy == .fullSha256 {
            try Sha256Verifier.verifyFile(at: url, named: relative, expectedHex: entry.sha256)
        }
        // Model.load bound the trusted receipt to this manifest and checked
        // that all file entries agree. Check this file's actual size above
        // before a caller can open it; receipt mode deliberately skips hashing.
        let offsets = MoEExpertOffsets(gateWOff: 0, gateSOff: UInt32(w), gateBOff: UInt32(w + aux),
            upWOff: UInt32(projection), upSOff: UInt32(projection + w), upBOff: UInt32(projection + w + aux),
            downWOff: UInt32(2 * projection), downSOff: UInt32(2 * projection + w),
            downBOff: UInt32(2 * projection + w + aux))
        return (StreamLayout(path: url.path, streamOffset: 0, streamSize: bytes,
            expertsPerLayer: pool.expertsPerLayer, expertStride: stride), offsets)
    }

    private static func invalid(_ detail: String) -> ModelError {
        .indexCorrupt(detail: "Flash-Next MTP: " + detail)
    }
}
