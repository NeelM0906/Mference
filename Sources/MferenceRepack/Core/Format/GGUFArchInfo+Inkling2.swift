import Foundation

extension ArchInfo {
    /// `manifest.arch` for an Inkling 2 GGUF (`general.architecture ==
    /// "inkling2_loop"`), built from the GGUF's own metadata and tensor table,
    /// because a GGUF upload has no `config.json`. Key names follow llama.cpp
    /// tag b12635 (`src/llama-arch.cpp`, `src/models/inkling2*.cpp`,
    /// `conversion/inkling2.py`). Every value is read, never defaulted; a
    /// missing key or an inconsistent pair fails closed. An install from the
    /// pinned source then requires the result to equal `ArchInfo.inkling2Mini`
    /// exactly (`requirePinnedInkling2Shape`), so it can only carry what the
    /// runner is built for.
    ///
    /// Facts the runtime fixes rather than the GGUF: the router is sigmoid
    /// with renormalized top-k (llama.cpp hard-codes both for this
    /// architecture); the attention scale is `head_dim^-0.5`; the
    /// relative-bias projection is `numHeads * d_rel` wide.
    static func fromGGUFInkling2(metadata md: [String: GGUF.Value],
                                 tensors: [GGUF.TensorInfo],
                                 path: String) throws -> ArchInfo {
        func invalid(_ d: String) -> RepackError { .ggufInvalid(path: path, detail: "arch: \(d)") }
        let arch = "inkling2_loop"
        guard md["general.architecture"]?.string == arch else {
            throw invalid("general.architecture is not \(arch)")
        }
        func value(_ key: String) -> GGUF.Value? { md[arch + "." + key] }
        func int(_ key: String) throws -> Int {
            guard let v = value(key)?.uint64, let n = Int(exactly: v) else {
                throw invalid("missing or non-integer \(arch).\(key)")
            }
            return n
        }
        /// A scalar, or a per-layer array that must hold one value.
        func uniformInt(_ key: String, layers: Int) throws -> Int {
            switch value(key) {
            case .array(let items)?:
                let values = try items.map { v -> Int in
                    guard let u = v.uint64, let n = Int(exactly: u) else { throw invalid("\(arch).\(key) entry") }
                    return n
                }
                guard values.count == layers, let first = values.first, values.allSatisfy({ $0 == first }) else {
                    throw invalid("\(arch).\(key) is not one value on all \(layers) layers")
                }
                return first
            default:
                return try int(key)
            }
        }
        /// An f32 value widened exactly (`Double(Float)`), as the manifest records it.
        func float(_ key: String) throws -> Double {
            switch value(key) {
            case .f32(let f)?: return Double(f)
            case .f64(let d)?: return d
            default: throw invalid("missing or non-float \(arch).\(key)")
            }
        }
        func bool(_ key: String) throws -> Bool {
            guard case .bool(let b)? = value(key) else { throw invalid("missing or non-bool \(arch).\(key)") }
            return b
        }
        let byName = Dictionary(tensors.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        func tensor(_ name: String) throws -> GGUF.TensorInfo {
            guard let t = byName[name] else { throw invalid("missing tensor \(name)") }
            return t
        }

        let numLayers = try int("block_count")
        let hidden = try int("embedding_length")
        guard numLayers > 0, hidden > 0 else { throw invalid("block_count or embedding_length is 0") }
        let numHeads = try uniformInt("attention.head_count", layers: numLayers)
        let numKVHeads = try uniformInt("attention.head_count_kv", layers: numLayers)
        let headDim = try int("attention.key_length")
        guard try int("attention.value_length") == headDim,
              try int("attention.key_length_swa") == headDim,
              try int("attention.value_length_swa") == headDim else {
            throw invalid("key, value and sliding head widths differ")
        }
        guard try int("rope.dimension_count") == 0 else { throw invalid("rope.dimension_count is not 0") }

        // The runtime kernels default to this epsilon; it is not an axis.
        guard try float("attention.layer_norm_rms_epsilon") == Double(Float(1e-6)) else {
            throw invalid("attention.layer_norm_rms_epsilon is not 1e-6")
        }

        // Sliding layers: `sliding_window_pattern[i] == true`.
        guard case .array(let pattern)? = value("attention.sliding_window_pattern"),
              pattern.count == numLayers else {
            throw invalid("attention.sliding_window_pattern is not one entry per layer")
        }
        let mask: [UInt8] = try pattern.map {
            guard case .bool(let sliding) = $0 else { throw invalid("sliding_window_pattern entry") }
            return sliding ? 0 : 1
        }

        // The vocabulary: the tokenizer holds the unpadded rows.
        let vocab = try int("vocab_size")
        let unpadded = try int("vocab_size_unpadded")
        guard case .array(let tokens)? = md["tokenizer.ggml.tokens"], tokens.count == unpadded,
              unpadded <= vocab else {
            throw invalid("tokenizer size is not vocab_size_unpadded \(unpadded)")
        }
        let output = try tensor("output.weight")
        let embd = try tensor("token_embd.weight")
        guard output.ne == [UInt64(hidden), UInt64(vocab)], embd.ne == output.ne else {
            throw invalid("output/token_embd shapes \(output.ne) / \(embd.ne) vs vocab \(vocab)")
        }

        let numExperts = try int("expert_count")
        let expertWidth = try uniformInt("expert_feed_forward_length", layers: numLayers)
        let dRel = try int("attention.relative_length")
        let relExtent = try int("attention.relative_extent")
        let window = try int("attention.sliding_window")
        guard !(try bool("loop.varying")) else { throw invalid("a variable-loop file is not Inkling 2 Mini") }

        // Mechanisms the tensor table proves present.
        guard let firstFull = mask.firstIndex(of: 1) else { throw invalid("no full-attention layer") }
        let rel = try tensor("blk.0.attn_rel.weight")
        let (relWidth, relOverflow) = numHeads.multipliedReportingOverflow(by: dRel)
        guard !relOverflow, rel.ne == [UInt64(hidden), UInt64(relWidth)] else {
            throw invalid("attn_rel shape \(rel.ne) is not [hidden, heads * d_rel]")
        }
        _ = try tensor("blk.\(firstFull).attn_k_conv_proj.weight")
        let sink = try tensor("blk.0.attn_sink_proj.weight")
        guard sink.ne == [UInt64(hidden), UInt64(numHeads)] else { throw invalid("attn_sink_proj shape \(sink.ne)") }
        _ = try tensor("token_embd_norm.weight")
        let conv = try tensor("blk.0.attn_k_conv.weight")
        let convKernel = try int("shortconv.l_cache")
        guard conv.ne.first == UInt64(convKernel) else { throw invalid("attn_k_conv width \(conv.ne)") }

        // Routed expert formats, one per layer, shared by gate-up and down.
        var formats: [String] = []
        var routerBias = true
        for L in 0..<numLayers {
            let gateUp = try tensor("blk.\(L).ffn_gate_up_exps.weight")
            let down = try tensor("blk.\(L).ffn_down_exps.weight")
            guard gateUp.type == down.type, gateUp.type == .nvfp4 || gateUp.type == .q8_0 else {
                throw invalid("layer \(L) experts are \(gateUp.type.name) / \(down.type.name)")
            }
            formats.append(gateUp.type.name)
            routerBias = routerBias && byName["blk.\(L).exp_probs_b.bias"] != nil
        }

        let ink = Inkling2Axes(
            relDRel: dRel,
            relExtent: relExtent,
            relProjDim: relWidth,
            relLogScalingFloor: try int("attention.log_scale_floor"),
            relLogScalingAlpha: try float("attention.log_scale_alpha"),
            sconvKernelSize: convKernel,
            embedNormEnabled: true,
            routerGateBias: routerBias,
            unpaddedVocabSize: unpadded,
            attentionSinks: true,
            loopCoreStart: try int("loop.start"),
            loopCoreEnd: try int("loop.end"),
            loopNumPasses: try int("loop.count"),
            loopGateRank: try int("loop.gate_rank"),
            attnResBlockSize: try int("attn_res.block_size"),
            qkClipThreshold: try float("attention.qk_clip"),
            relClipThreshold: try float("attention.relative_clip"),
            attentionSinkProjected: true,
            expertGateSoftcap: try float("activation.situ_beta"),
            expertUpSoftcap: try float("activation.situ_linear_beta"),
            routedExpertFormats: formats)

        let info = ArchInfo(
            hiddenSize: hidden,
            intermediateSize: try int("feed_forward_length"),
            moeIntermediateSize: expertWidth,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            numFullKVHeads: numKVHeads,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: vocab,
            slidingWindow: window,
            finalLogitSoftcap: try float("final_logit_softcapping"),
            ropeTheta: 0,
            fullRopeTheta: 0,
            partialRotaryFactor: 0,
            numLayers: numLayers,
            numExperts: numExperts,
            topKExperts: try uniformInt("expert_used_count", layers: numLayers),
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: "situ",
            family: .inkling2Loop,
            attnOutputGate: false,
            attentionScale: pow(Double(headDim), -0.5),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            routerScoringFunc: "sigmoid",
            routedScalingFactor: try float("expert_weights_scale"),
            swigluLimit: 0,
            numSharedExperts: 0,
            numDenseLayers: 0,
            denseIntermediateSize: 0,
            routerNormAfterTopK: true,
            qkNorm: false,
            inkling2: ink)
        return info
    }

    /// Refuses an arch that is not exactly the pinned Inkling 2 Mini shape,
    /// naming the first field that differs.
    static func requirePinnedInkling2Shape(_ info: ArchInfo, path: String) throws {
        guard info == .inkling2Mini else {
            throw RepackError.ggufInvalid(
                path: path,
                detail: "arch: not the pinned Inkling 2 Mini shape: \(firstDifference(info, .inkling2Mini))")
        }
    }

    /// The first differing field, for a refusal that says what is wrong.
    static func firstDifference(_ a: ArchInfo, _ b: ArchInfo) -> String {
        let left = Mirror(reflecting: a).children
        let right = Array(Mirror(reflecting: b).children)
        for (index, child) in left.enumerated() {
            let other = right[index].value
            if "\(child.value)" != "\(other)" {
                return "\(child.label ?? "?") \(child.value) != \(other)"
            }
        }
        return "none"
    }
}
