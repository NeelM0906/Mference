import Foundation

extension ArchInfo {
    /// `manifest.arch` for a Qwen3.8-Flash-Next GGUF (`general.architecture ==
    /// "qwen4exp"`), built from the GGUF's own metadata because a GGUF upload
    /// has no `config.json`. Key names follow llama.cpp b1b770d9
    /// (`src/llama-arch.cpp`, `conversion/qwen4exp.py`). Every value is read,
    /// never defaulted; a missing key or an inconsistent pair fails closed.
    ///
    /// Two PLE fields differ from the vendor config because the GGUF stores the
    /// n-gram table differently: `pleNgramShardCount` is 1 (the converter
    /// concatenated the vendor's 128 shards into one tensor) and
    /// `pleNgramVocabSizeBase` is 0 (the GGUF carries the explicit per-head
    /// sizes in `ple.head_vocab_sizes` instead of the base).
    static func fromGGUFQwen4Exp(metadata md: [String: GGUF.Value],
                                 tensors: [GGUF.TensorInfo],
                                 path: String) throws -> ArchInfo {
        func invalid(_ d: String) -> RepackError { .ggufInvalid(path: path, detail: "arch: \(d)") }
        guard md["general.architecture"]?.string == "qwen4exp" else {
            throw invalid("general.architecture is not qwen4exp")
        }
        func int(_ key: String) throws -> Int {
            guard let v = md["qwen4exp." + key]?.uint64, let n = Int(exactly: v) else {
                throw invalid("missing or non-integer qwen4exp.\(key)")
            }
            return n
        }
        func ints(_ key: String) throws -> [Int] {
            guard case .array(let a)? = md["qwen4exp." + key] else { throw invalid("missing qwen4exp.\(key)") }
            return try a.map { v in
                guard let u = v.uint64, let n = Int(exactly: u) else { throw invalid("qwen4exp.\(key) entry") }
                return n
            }
        }
        func float(_ key: String) throws -> Double {
            switch md["qwen4exp." + key] {
            case .f32(let f)?: return Double(f)
            case .f64(let d)?: return d
            default: throw invalid("missing or non-float qwen4exp.\(key)")
            }
        }
        func tensor(_ name: String) throws -> GGUF.TensorInfo {
            guard let t = tensors.first(where: { $0.name == name }) else { throw invalid("missing tensor \(name)") }
            return t
        }

        let numLayers = try int("block_count")
        let headDim = try int("attention.key_length")
        guard try int("attention.value_length") == headDim else { throw invalid("key_length != value_length") }
        let rotaryDims = try int("rope.dimension_count")
        guard rotaryDims > 0, rotaryDims <= headDim, headDim % rotaryDims == 0 else {
            throw invalid("rope.dimension_count \(rotaryDims) vs head dim \(headDim)")
        }

        // Hybrid schedule: every `full_attention_interval`-th layer is full
        // attention (llama.cpp's qwen35 rule), and only those layers carry a
        // nonzero indexer compress ratio.
        let interval = try int("full_attention_interval")
        guard interval > 0 else { throw invalid("full_attention_interval 0") }
        let mask: [UInt8] = (0..<numLayers).map { ($0 + 1) % interval == 0 ? 1 : 2 }
        let ratios = try ints("attention.compress_ratios")
        guard ratios.count == numLayers else { throw invalid("compress_ratios has \(ratios.count) entries") }
        let fullRatios = Set(zip(mask, ratios).filter { $0.0 == 1 }.map(\.1))
        guard zip(mask, ratios).allSatisfy({ ($0.0 == 1) == ($0.1 != 0) }), fullRatios.count == 1,
              let compressRatio = fullRatios.first else {
            throw invalid("compress_ratios do not follow full_attention_interval")
        }

        // Linear attention (gated DeltaNet), llama.cpp ssm.* naming.
        let vHeads = try int("ssm.time_step_rank")
        let inner = try int("ssm.inner_size")
        guard vHeads > 0, inner % vHeads == 0 else { throw invalid("ssm.inner_size / time_step_rank") }

        // Indexer K heads are not a key; the K projection's output width is.
        let indexerHeadDim = try int("attention.indexer.key_length")
        let firstFull = try require0(mask.firstIndex(of: 1), invalid("no full-attention layer"))
        let indexerK = try tensor("blk.\(firstFull).indexer.k_proj.weight")
        guard indexerK.ne.count == 2, indexerK.ne[1] % UInt64(indexerHeadDim) == 0 else {
            throw invalid("indexer K projection shape \(indexerK.ne)")
        }
        // Gated attention output: a full-attention layer's Q projection is
        // twice the head width (query and output gate per head, as in Qwen 3.6).
        let q = try tensor("blk.\(firstFull).attn_q.weight")
        guard q.ne.count == 2, q.ne[1] == UInt64(2 * (try int("attention.head_count")) * headDim) else {
            throw invalid("attn_q shape \(q.ne) is not query + output gate")
        }

        let pleLayers = try ints("ple.layers")
        let vocab = try {
            guard case .array(let toks)? = md["tokenizer.ggml.tokens"] else { throw invalid("missing tokens") }
            return toks.count
        }()
        let output = try tensor("output.weight")
        let embd = try tensor("token_embd.weight")
        guard output.ne == [UInt64(try int("embedding_length")), UInt64(vocab)], embd.ne == output.ne else {
            throw invalid("output/token_embd shapes \(output.ne) / \(embd.ne) vs vocab \(vocab)")
        }

        let axes = FlashNextAxes(
            hcCount: try int("hyper_connection.count"),
            hcLowRank: try int("hyper_connection.low_rank"),
            indexerNumHeads: try int("attention.indexer.head_count"),
            indexerHeadDim: indexerHeadDim,
            indexerNumKVHeads: Int(indexerK.ne[1]) / indexerHeadDim,
            indexerBudget: try int("attention.indexer.top_k"),
            indexerCompressRatio: compressRatio,
            pleLayerIDs: pleLayers.map { $0 + 1 },
            pleNgramShardCount: 1,
            pleNgramVocabSizeBase: 0,
            pleConvKernelSize: try int("ple.conv_kernel"))

        let theta = try float("rope.freq_base")
        return ArchInfo(
            hiddenSize: try int("embedding_length"),
            intermediateSize: try int("expert_shared_feed_forward_length"),
            moeIntermediateSize: try int("expert_feed_forward_length"),
            numHeads: try int("attention.head_count"),
            numKVHeads: try int("attention.head_count_kv"),
            numFullKVHeads: try int("attention.head_count_kv"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: vocab,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: Double(rotaryDims) / Double(headDim),
            numLayers: numLayers,
            numExperts: try int("expert_count"),
            topKExperts: try int("expert_used_count"),
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: "silu",
            family: .qwen38flashnextudiq3xxs,
            attnOutputGate: true,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try int("ssm.group_count"),
            linearNumVHeads: vHeads,
            linearKeyHeadDim: try int("ssm.state_size"),
            linearValueHeadDim: inner / vHeads,
            linearConvKernelSize: try int("ssm.conv_kernel"),
            unpaddedVocabSize: 0,
            flashNext: axes)
    }
}

private func require0<T>(_ v: T?, _ e: @autoclosure () -> Error) throws -> T {
    guard let v else { throw e() }
    return v
}
