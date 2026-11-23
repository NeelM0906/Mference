import Foundation

/// Shape and constants of an Inkling 2 Mini (`inkling2_loop`) text decoder,
/// as the whole-model CPU reference reads them.
///
/// Field names follow the upstream llama.cpp hyperparameters (the same file
/// runs there as the base reference), with the vendor `config.json` key in
/// each doc comment. `mini` is the shipped checkpoint; `Inkling2ToyCheckpoint`
/// builds a tiny one with the same schedule.
public struct Inkling2ReferenceConfig: Sendable, Equatable {
    /// `hidden_size` (2304).
    public var hiddenSize: Int
    /// Stored decoder layers, `num_hidden_layers` (24).
    public var numLayers: Int
    /// First looped layer, `loop_config.core_start` (6).
    public var coreStart: Int
    /// One past the last looped layer, `loop_config.core_end` (18).
    public var coreEnd: Int
    /// Passes over the looped layers, `loop_config.num_passes` (3).
    public var numPasses: Int
    /// Rank of the four message gates per looped layer, `loop_config.gate_rank` (64).
    public var gateRank: Int
    /// Layers per attention-residual block, `n_layers_per_attn_res_block` (6).
    public var attnResBlockSize: Int
    /// Per stored layer: true for a sliding-window layer (`local_layer_ids`),
    /// false for a global (full-attention) layer.
    public var slidingLayers: [Bool]
    /// Query heads (`num_attention_heads`, 24; the same for sliding layers).
    public var numHeads: Int
    /// Key/value heads (`num_key_value_heads`, 8). Query head `h` reads KV
    /// head `h / (numHeads / numKVHeads)`.
    public var numKVHeads: Int
    /// `head_dim` (128).
    public var headDim: Int
    /// `sliding_window_size` (1024): a sliding layer attends to backward
    /// distances `0 ..< slidingWindow`, and its relative-bias table has this
    /// many rows.
    public var slidingWindow: Int
    /// Width of the per-head relative state, `d_rel` (16).
    public var relDim: Int
    /// Rows of a global layer's relative-bias table, `rel_extent` (1024).
    /// The bias is zero at backward distances at or past the extent.
    public var relExtent: Int
    /// L2 clip of query and key head vectors, `qk_l2_clip_threshold`.
    public var qkClip: Float
    /// L2 clip of the relative state, `rel_l2_clip_threshold` (8).
    public var relClip: Float
    /// `log_scaling_n_floor` (8192).
    public var logScaleFloor: Int
    /// `log_scaling_alpha` (0.1). Global layers scale the query and the
    /// relative state by `1 + alpha * log(max(1, (pos + 1) / floor))`.
    public var logScaleAlpha: Float
    /// Short-convolution taps, `sconv_kernel_size` (4).
    public var convKernel: Int
    /// `n_routed_experts` (128).
    public var numExperts: Int
    /// `num_experts_per_tok` (8).
    public var topK: Int
    /// Expert width, `intermediate_size` (1152).
    public var expertFF: Int
    /// `route_scale` (8): the normalized routing weights sum to this.
    public var routeScale: Float
    /// `mlp_gate_softcap` (4): bounds the linear gate factor.
    public var gateSoftcap: Float
    /// `mlp_up_softcap` (25): bounds `up`.
    public var upSoftcap: Float
    /// `final_logit_softcapping` (30). Zero disables it.
    public var finalLogitSoftcap: Float
    /// Rows of the stored embedding and output head, `vocab_size` (201,024).
    public var vocabPadded: Int
    /// Logits the model produces, `unpadded_vocab_size` (200,058).
    public var vocabUnpadded: Int
    /// `rms_norm_eps` (1e-6); also the epsilon of every weightless norm.
    public var rmsEps: Float

    public init(hiddenSize: Int, numLayers: Int, coreStart: Int, coreEnd: Int,
                numPasses: Int, gateRank: Int, attnResBlockSize: Int,
                slidingLayers: [Bool], numHeads: Int, numKVHeads: Int,
                headDim: Int, slidingWindow: Int, relDim: Int, relExtent: Int,
                qkClip: Float, relClip: Float, logScaleFloor: Int,
                logScaleAlpha: Float, convKernel: Int, numExperts: Int,
                topK: Int, expertFF: Int, routeScale: Float,
                gateSoftcap: Float, upSoftcap: Float,
                finalLogitSoftcap: Float, vocabPadded: Int,
                vocabUnpadded: Int, rmsEps: Float) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.coreStart = coreStart
        self.coreEnd = coreEnd
        self.numPasses = numPasses
        self.gateRank = gateRank
        self.attnResBlockSize = attnResBlockSize
        self.slidingLayers = slidingLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.slidingWindow = slidingWindow
        self.relDim = relDim
        self.relExtent = relExtent
        self.qkClip = qkClip
        self.relClip = relClip
        self.logScaleFloor = logScaleFloor
        self.logScaleAlpha = logScaleAlpha
        self.convKernel = convKernel
        self.numExperts = numExperts
        self.topK = topK
        self.expertFF = expertFF
        self.routeScale = routeScale
        self.gateSoftcap = gateSoftcap
        self.upSoftcap = upSoftcap
        self.finalLogitSoftcap = finalLogitSoftcap
        self.vocabPadded = vocabPadded
        self.vocabUnpadded = vocabUnpadded
        self.rmsEps = rmsEps
    }

    /// Inkling 2 Mini at `thinkingmachines/Inkling-2-Mini` `b85c3cf`
    /// (`config.json` `text_config` and `loop_config`).
    public static let mini = Inkling2ReferenceConfig(
        hiddenSize: 2304, numLayers: 24, coreStart: 6, coreEnd: 18,
        numPasses: 3, gateRank: 64, attnResBlockSize: 6,
        slidingLayers: (0..<24).map { ($0 + 1) % 6 != 0 },
        numHeads: 24, numKVHeads: 8, headDim: 128,
        slidingWindow: 1024, relDim: 16, relExtent: 1024,
        qkClip: 22.627416997969522, relClip: 8.0,
        logScaleFloor: 8192, logScaleAlpha: 0.1, convKernel: 4,
        numExperts: 128, topK: 8, expertFF: 1152, routeScale: 8.0,
        gateSoftcap: 4.0, upSoftcap: 25.0, finalLogitSoftcap: 30.0,
        vocabPadded: 201_024, vocabUnpadded: 200_058, rmsEps: 1e-6)

    // MARK: Derived shapes

    public var qDim: Int { numHeads * headDim }
    public var kvDim: Int { numKVHeads * headDim }
    public var headsPerKVHead: Int { numHeads / numKVHeads }
    public var numCoreLayers: Int { coreEnd - coreStart }
    /// Layer visits per token, which is also the number of logical cache
    /// layers (48 for Mini).
    public var numVisits: Int { numLayers + (numPasses - 1) * numCoreLayers }
    public func isCore(_ layer: Int) -> Bool { layer >= coreStart && layer < coreEnd }
    /// Rows of layer `layer`'s relative-bias table.
    public func relativeExtent(layer: Int) -> Int {
        slidingLayers[layer] ? slidingWindow : relExtent
    }

    /// The fixed per-token schedule, in execution order: prelude, pass 1,
    /// pass 2, ..., coda.
    public var schedule: [Inkling2Visit] {
        var visits: [Inkling2Visit] = []
        visits.reserveCapacity(numVisits)
        for layer in 0..<coreStart {
            visits.append(Inkling2Visit(order: visits.count, layer: layer, pass: 0, cacheSlot: layer))
        }
        for pass in 0..<numPasses {
            for layer in coreStart..<coreEnd {
                visits.append(Inkling2Visit(order: visits.count, layer: layer, pass: pass,
                                            cacheSlot: cacheSlot(layer: layer, pass: pass)))
            }
        }
        for layer in coreEnd..<numLayers {
            visits.append(Inkling2Visit(order: visits.count, layer: layer, pass: 0, cacheSlot: layer))
        }
        return visits
    }

    /// The logical cache layer of stored layer `layer` in loop pass `pass`.
    /// Pass 0 (and every layer outside the core) uses the layer's own index;
    /// pass `p >= 1` of core layer `l` uses
    /// `numLayers + (p - 1) * numCoreLayers + (l - coreStart)`, the
    /// numbering of upstream llama.cpp and transformers (pass 2 of Mini is
    /// 24 to 35, pass 3 is 36 to 47).
    public func cacheSlot(layer: Int, pass: Int) -> Int {
        guard pass > 0 else { return layer }
        precondition(isCore(layer), "only core layers have later passes")
        return numLayers + (pass - 1) * numCoreLayers + (layer - coreStart)
    }

    /// Checks the invariants the reference relies on; throws a readable
    /// message for a malformed config.
    public func validate() throws {
        func fail(_ message: String) -> Error {
            NSError(domain: "Inkling2ReferenceConfig", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: message])
        }
        guard slidingLayers.count == numLayers else { throw fail("slidingLayers has \(slidingLayers.count) entries for \(numLayers) layers") }
        guard coreStart > 0, coreStart < coreEnd, coreEnd <= numLayers else { throw fail("invalid loop core [\(coreStart), \(coreEnd))") }
        guard coreStart % attnResBlockSize == 0, coreEnd % attnResBlockSize == 0 else { throw fail("loop core must align with the attention-residual blocks") }
        guard numPasses >= 1, gateRank >= 1 else { throw fail("invalid pass count or gate rank") }
        guard numKVHeads > 0, numHeads % numKVHeads == 0 else { throw fail("query heads must be a multiple of KV heads") }
        guard topK >= 1, topK <= numExperts else { throw fail("invalid top-k") }
        guard convKernel >= 2 else { throw fail("convolution needs at least two taps") }
        guard vocabUnpadded >= 1, vocabUnpadded <= vocabPadded else { throw fail("invalid vocabulary") }
        guard slidingWindow >= 1, relExtent >= 1, logScaleFloor >= 1 else { throw fail("invalid window, extent or log floor") }
    }
}

/// One layer visit of the per-token schedule.
public struct Inkling2Visit: Sendable, Equatable {
    /// Index in execution order (0 ..< numVisits).
    public let order: Int
    /// Stored layer whose weights run.
    public let layer: Int
    /// Loop pass (0 for the prelude, the coda and the first core pass).
    public let pass: Int
    /// Logical cache layer holding this visit's KV and convolution state.
    public let cacheSlot: Int
}
