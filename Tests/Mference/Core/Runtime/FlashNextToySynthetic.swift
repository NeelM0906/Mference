import Foundation
@testable import Mference
@testable import MferenceRepackCore

/// Synthetic Qwen3.8-Flash-Next toy fixture: a tiny `.gturbo/` directory with
/// the family's real tensor-name contract — the vendor container order
/// (`model.language_model.`), a bare `lm_head`, **no** `input_layernorm` /
/// `post_attention_layernorm` / final `norm` (the hyper-connection sites carry
/// the norms), `linear_attn.*` on mask-2 layers, `self_attn.*` plus a QSA
/// indexer on mask-1 layers, a 512-way-shaped router and gated shared expert
/// per layer, and one PLE block with its three I64 hash tables.
///
/// It also writes a real row-lookup pool at `ple/layer_01_ngram_rows.bin` with
/// a deliberately small block stride, so block-boundary and tail-block row
/// addressing are exercised with a 1.5 KB file rather than a 102 GB one.
///
/// Mirrors `Qwen38ToySynthetic`. Deliberately written in this (runtime) test
/// tree rather than reusing the repacker's synthetic-install builder, so the
/// runtime suite owns its own fixture.
enum FlashNextToySynthetic {

    /// Layer index carrying the PLE block: `pleLayerIDs` is one-indexed, so
    /// id 2 means `layers[1]`.
    static let pleLayer = 1

    /// Row-pool geometry of the toy. Tiny on every axis, but every relation
    /// the production pool satisfies holds: rows tile blocks exactly, blocks
    /// are stride-aligned, a record never straddles a block, the last block of
    /// a shard is short, and shard regions are consecutive.
    enum Pool {
        static let ngramHeads = 4
        static let rowDim = 16                       // hidden 64 / 4 heads
        static let rowStride = rowDim * 2            // BF16
        static let blockStride = 256
        static let rowsPerBlock = blockStride / rowStride   // 8
        static let shardCount = 2
        static let rowsPerShard = 20                 // 2 full blocks + a 4-row tail
        static let blocksPerShard = 3
        static let shardBytes = blocksPerShard * blockStride
        static let totalRows = shardCount * rowsPerShard
        static let fileSize = shardCount * shardBytes
        static let file = "ple/layer_01_ngram_rows.bin"

        /// The value stored at `[row, column]`.
        ///
        /// A small integer times a power of two, so every value is **exactly**
        /// representable in BF16's 8-bit significand and a reader can be
        /// checked against it with `==` rather than a tolerance. (A plain
        /// `row + column/64` would round, and the test would then be measuring
        /// BF16 rounding instead of the pool's addressing.)
        static func value(row: Int, column: Int) -> Float {
            Float(row + 1) * (Float(1 << column) / 256.0)
        }
    }

    /// Build the toy directory in a temp dir and return its URL.
    static func write() throws -> URL {
        let toy = ArchConfig.qwen38FlashNextToy()
        let la = toy.linearAttention
        let fn = toy.flashNext
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-flashnext-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("ple"), withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let bundle = fn.hcCount * d          // the 4-stream residual width
        let u16 = MemoryLayout<UInt16>.stride

        func int4AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols / 2),
                                scaleBytes: auxBytes, biasBytes: auxBytes)
        }
        func bf16Spec(_ name: String, shape: [UInt32], count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 1, shape: shape,
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0, biasBytes: 0)
        }
        func int64Spec(_ name: String, count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 4,
                         shape: [UInt32(count), 0, 0, 0],
                         weightBytes: UInt64(count * MemoryLayout<Int64>.stride),
                         scaleBytes: 0, biasBytes: 0)
        }

        let trunk = "model.language_model."
        var specs: [ResidentSpec] = [
            int4AffineSpec("\(trunk)embed_tokens.weight", rows: toy.vocabSize, cols: d),
            int4AffineSpec("lm_head.weight", rows: toy.vocabSize, cols: d),
            // The global mixer stands in for the final norm this family lacks.
            bf16Spec("\(trunk)hyper_connection_mixer.hc_norm.weight",
                     shape: [UInt32(bundle), 0, 0, 0], count: bundle),
            int4AffineSpec("\(trunk)hyper_connection_mixer.input_mix_weight_down.weight",
                           rows: fn.hcLowRank, cols: bundle),
            int4AffineSpec("\(trunk)hyper_connection_mixer.input_mix_weight_up.weight",
                           rows: bundle, cols: fn.hcLowRank),
        ]

        for L in 0..<toy.numLayers {
            let prefix = "\(trunk)layers.\(L)"
            for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
                specs.append(bf16Spec("\(prefix).\(site).hc_norm.weight",
                                      shape: [UInt32(bundle), 0, 0, 0], count: bundle))
                specs.append(int4AffineSpec("\(prefix).\(site).input_mix_weight_down.weight",
                                            rows: fn.hcLowRank, cols: bundle))
                specs.append(int4AffineSpec("\(prefix).\(site).input_mix_weight_up.weight",
                                            rows: bundle, cols: fn.hcLowRank))
                specs.append(int4AffineSpec("\(prefix).\(site).block_inject_weight.weight",
                                            rows: fn.hcCount, cols: bundle))
            }
            specs.append(int4AffineSpec("\(prefix).mlp.gate.weight",
                                        rows: toy.numExperts, cols: d))
            specs.append(int4AffineSpec("\(prefix).mlp.shared_expert.gate_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int4AffineSpec("\(prefix).mlp.shared_expert.up_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int4AffineSpec("\(prefix).mlp.shared_expert.down_proj.weight",
                                        rows: d, cols: toy.intermediateSize))
            specs.append(int4AffineSpec("\(prefix).mlp.shared_expert_gate.weight",
                                        rows: 1, cols: d))

            if toy.layerIsLinear(L) {
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_qkv.weight",
                                            rows: la.qkvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_z.weight",
                                            rows: la.valueDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_a.weight",
                                            rows: la.numVHeads, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_b.weight",
                                            rows: la.numVHeads, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.out_proj.weight",
                                            rows: d, cols: la.valueDim))
                specs.append(bf16Spec("\(prefix).linear_attn.conv1d.weight",
                                      shape: [UInt32(la.qkvDim), 1, UInt32(la.convKernelSize), 0],
                                      count: la.qkvDim * la.convKernelSize))
                specs.append(bf16Spec("\(prefix).linear_attn.A_log",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.dt_bias",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.norm.weight",
                                      shape: [UInt32(la.valueHeadDim), 0, 0, 0],
                                      count: la.valueHeadDim))
            } else {
                let qDim = toy.numHeads * toy.fullHeadDim
                let kvDim = toy.numFullKVHeads * toy.fullHeadDim
                specs.append(int4AffineSpec("\(prefix).self_attn.q_proj.weight",
                                            rows: 2 * qDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.k_proj.weight",
                                            rows: kvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.v_proj.weight",
                                            rows: kvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.o_proj.weight",
                                            rows: d, cols: qDim))
                specs.append(bf16Spec("\(prefix).self_attn.q_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
                specs.append(bf16Spec("\(prefix).self_attn.k_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
                let indexerRows =
                    (fn.indexerNumHeads + fn.indexerNumKVHeads) * fn.indexerHeadDim
                specs.append(int4AffineSpec("\(prefix).self_attn.indexer.index_qk_proj.weight",
                                            rows: indexerRows, cols: d))
                specs.append(bf16Spec("\(prefix).self_attn.indexer.q_layernorm.weight",
                                      shape: [UInt32(fn.indexerHeadDim), 0, 0, 0],
                                      count: fn.indexerHeadDim))
                specs.append(bf16Spec("\(prefix).self_attn.indexer.k_layernorm.weight",
                                      shape: [UInt32(fn.indexerHeadDim), 0, 0, 0],
                                      count: fn.indexerHeadDim))
            }

            if toy.layerIsPLE(L) {
                specs.append(int4AffineSpec("\(prefix).ple.key_proj.weight",
                                            rows: bundle, cols: d))
                specs.append(int4AffineSpec("\(prefix).ple.value_proj.weight",
                                            rows: d, cols: d))
                specs.append(bf16Spec("\(prefix).ple.conv1d.weight",
                                      shape: [UInt32(bundle), 1, UInt32(fn.pleConvKernelSize), 0],
                                      count: bundle * fn.pleConvKernelSize))
                for norm in ["norm_conv", "norm_key", "norm_query"] {
                    specs.append(bf16Spec("\(prefix).ple.\(norm).weight",
                                          shape: [UInt32(bundle), 0, 0, 0], count: bundle))
                }
                specs.append(int64Spec("\(prefix).ple.ple_embedding.layer_multipliers",
                                       count: 3))
                specs.append(int64Spec("\(prefix).ple.ple_embedding.ngram_heads_offsets",
                                       count: Pool.ngramHeads))
                specs.append(int64Spec("\(prefix).ple.ple_embedding.ngram_heads_vocab_sizes",
                                       count: Pool.ngramHeads))
            }
        }

        // 2. Serialize the resident index + payload.
        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes = GTurboBinary.indexEntryBytes
        let stringTableBase = headerBytes + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)

        var entries: [ResidentEntry] = []
        var payloadCursor = indexBytes
        for spec in specs {
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name, dtype: spec.dtype, logicalShape4: spec.shape,
                fileOffset: weightOffset, sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset, scaleSize: spec.scaleBytes,
                biasOffset: biasOffset, biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil, sourceBiases: nil))
            payloadCursor += spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize = payloadCursor - indexBytes
        let totalBytes = Int(indexBytes + residentSize)

        var fileBuf = [UInt8](repeating: 0, count: totalBytes)
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                GTurboBinary.writeIndexEntry(
                    into: base.advanced(by: headerBytes + i * entryBytes),
                    entry: e, nameOffset: nameAbsOffsets[i])
            }
            _ = stringTable.withUnsafeBytes { sb in
                memcpy(base.advanced(by: stringTableBase), sb.baseAddress!, stringTable.count)
            }
            var lcg: UInt64 = 0x9E3779B97F4A7C15
            func nextByte() -> UInt8 {
                lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
                return UInt8(truncatingIfNeeded: lcg >> 33)
            }
            for entry in entries where entry.dtype == 0 {
                let weights = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt8.self)
                for i in 0..<Int(entry.sizeBytes) { weights[i] = nextByte() }
                if entry.scaleSize > 0 {
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / u16) {
                        scales[i] = Quantization.bf16Bits(0.004 + 0.002 * Float(i % 7))
                    }
                }
                if entry.biasSize > 0 {
                    let biases = base.advanced(by: Int(entry.biasOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.biasSize) / u16) {
                        biases[i] = Quantization.bf16Bits(-0.05 + 0.01 * Float(i % 11))
                    }
                }
            }
            // BF16 weights are written **zero-centered**, the convention the
            // checkpoint actually uses: values scatter around 0, not 1. The
            // loader's `(1 + w)` bake is what turns them into full form, and a
            // test that saw values near 1 here could not tell the difference.
            for entry in entries where entry.dtype == 1 {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    dst[i] = Quantization.bf16Bits(0.25 * Float(i % 7) - 0.75)
                }
            }
            for entry in entries where entry.dtype == 4 {
                let count = Int(entry.sizeBytes) / MemoryLayout<Int64>.stride
                let values = int64Table(named: entry.name, count: count)
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: Int64.self)
                for i in 0..<count { dst[i] = values[i].littleEndian }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. Routed-expert blobs: gate | up | down per expert, page-aligned.
        let expertStride: UInt64 = 16384
        let layerBytes = Int(expertStride) * toy.numExperts
        func expertBlob(_ expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]
            func add(_ role: String, rows: Int, cols: Int, seed: Int) {
                let groups = cols / Quantization.groupSize
                let packedOffset = bytes.count
                for i in 0..<(rows * cols / 2) {
                    bytes.append(UInt8(truncatingIfNeeded: expert &* 31 &+ seed &* 7 &+ i))
                }
                tensors[role] = ["offset": packedOffset,
                                 "size": bytes.count - packedOffset,
                                 "dtype": "U32", "shape": [rows, cols], "bits": 4]
                for component in ["_scales", "_biases"] {
                    let start = bytes.count
                    for i in 0..<(rows * groups) {
                        let bits = Quantization.bf16Bits(0.01 + 0.001 * Float(i % 5))
                        bytes.append(UInt8(truncatingIfNeeded: bits))
                        bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
                    }
                    tensors[role + component] = ["offset": start,
                                                 "size": bytes.count - start,
                                                 "dtype": "BF16",
                                                 "shape": [rows, groups]]
                }
            }
            add("gate", rows: toy.moeIntermediateSize, cols: d, seed: 0)
            add("up", rows: toy.moeIntermediateSize, cols: d, seed: 1)
            add("down", rows: d, cols: toy.moeIntermediateSize, seed: 2)
            return (bytes, tensors)
        }

        var layersArr: [[String: Any]] = []
        var layerShaByName: [String: String] = [:]
        for L in 0..<toy.numLayers {
            var payload = Data(count: layerBytes)
            var experts: [[String: Any]] = []
            for E in 0..<toy.numExperts {
                let blob = expertBlob(E)
                precondition(blob.bytes.count <= Int(expertStride))
                let base = E * Int(expertStride)
                for (i, byte) in blob.bytes.enumerated() { payload[base + i] = byte }
                experts.append(["expert": E,
                                "offset": UInt64(E) * expertStride,
                                "size": expertStride,
                                "tensors": blob.tensors])
            }
            let basename = String(format: "layer_%02d.bin", L)
            let url = exp.appendingPathComponent(basename)
            try payload.write(to: url)
            layerShaByName["packed_experts/\(basename)"] =
                try Sha256Verifier.hashFile(at: url)
            layersArr.append(["layer": L, "file": basename, "experts": experts])
        }
        let layoutData = try JSONSerialization.data(
            withJSONObject: ["expertStride": expertStride,
                             "numLayers": toy.numLayers,
                             "expertsPerLayer": toy.numExperts,
                             "layers": layersArr] as [String: Any],
            options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 4. The PLE row pool.
        let poolData = try writeRowPool(directoryURL: dir)

        // 5. manifest.json.
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": totalBytes, "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
            Pool.file: ["size": poolData.size, "sha256": poolData.sha256],
        ]
        for (rel, sha) in layerShaByName { files[rel] = ["size": layerBytes, "sha256": sha] }

        let archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
            "family": toy.family.rawValue,
            "attnOutputGate": toy.attnOutputGate,
            "attentionScale": toy.attentionScale,
            "embeddingScaledBySqrtHidden": toy.embeddingScaledBySqrtHidden,
            "routerScaled": toy.routerScaled,
            "ffnSandwichNorms": toy.ffnSandwichNorms,
            "sharedExpertGated": toy.sharedExpertGated,
            "ropeNeoxSubdim": toy.ropeNeoxSubdim,
            "linearNumKHeads": la.numKHeads, "linearNumVHeads": la.numVHeads,
            "linearKeyHeadDim": la.keyHeadDim, "linearValueHeadDim": la.valueHeadDim,
            "linearConvKernelSize": la.convKernelSize,
            "numSharedExperts": toy.numSharedExperts,
            "hcCount": fn.hcCount, "hcLowRank": fn.hcLowRank,
            "indexerNumHeads": fn.indexerNumHeads,
            "indexerHeadDim": fn.indexerHeadDim,
            "indexerNumKVHeads": fn.indexerNumKVHeads,
            "indexerBudget": fn.indexerBudget,
            "indexerCompressRatio": fn.indexerCompressRatio,
            "pleLayerIDs": fn.pleLayerIDs,
            "pleNgramShardCount": fn.pleNgramShardCount,
            "pleNgramVocabSizeBase": fn.pleNgramVocabSizeBase,
            "pleConvKernelSize": fn.pleConvKernelSize,
            "requiredAxes": ["hyperConnectionsLowRank", "attentionIndexer",
                             "pleNgramEmbedding"],
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "qwen38flashnext-toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
            "plePool": plePoolBlock(),
            "sidecars": ["mtp": ["carried": false, "tensorCount": 0],
                         "vision": ["carried": false, "tensorCount": 0]],
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    /// The three I64 hash tables, as values a test can predict.
    ///
    /// Head `h` owns rows `[offsets[h], offsets[h] + vocabSizes[h])`, and the
    /// heads tile the pool exactly — the production layout, at toy scale.
    static func int64Table(named name: String, count: Int) -> [Int64] {
        if name.hasSuffix("layer_multipliers") {
            // Odd, as splitmix64 derivation guarantees.
            return [0x0000_0000_0000_0BB9, 0x0000_0000_0001_7773, 0x0000_0000_0002_332D]
        }
        if name.hasSuffix("ngram_heads_offsets") {
            return (0..<count).map { Int64($0 * Pool.totalRows / count) }
        }
        if name.hasSuffix("ngram_heads_vocab_sizes") {
            return [Int64](repeating: Int64(Pool.totalRows / count), count: count)
        }
        return [Int64](repeating: 0, count: count)
    }

    static func plePoolBlock() -> [String: Any] {
        var shards: [[String: Any]] = []
        for s in 0..<Pool.shardCount {
            shards.append(["shard": s, "rows": Pool.rowsPerShard,
                           "offset": s * Pool.shardBytes, "size": Pool.shardBytes])
        }
        return [
            "kind": "rowLookupPoolV1",
            "layers": [[
                "layer": pleLayer,
                "file": Pool.file,
                "sourceTensor": "model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
                "rows": Pool.totalRows,
                "rowDim": Pool.rowDim,
                "storage": "bf16",
                "weightBits": 16,
                "scheme": "none", "scaleType": "none", "biasType": "none",
                "groupSize": 0,
                "rowWeightBytes": Pool.rowStride,
                "rowScaleBytes": 0, "rowBiasBytes": 0,
                "rowStride": Pool.rowStride,
                "rowsPerBlock": Pool.rowsPerBlock,
                "blockStride": Pool.blockStride,
                "fileSize": Pool.fileSize,
                "shards": shards,
            ] as [String: Any]],
        ]
    }

    /// Write the toy row pool with `Pool.value(row:column:)` in every row and
    /// a recognizable filler in the block slack, so a reader that mis-computes
    /// the block offset reads the filler rather than a plausible row.
    @discardableResult
    static func writeRowPool(directoryURL dir: URL) throws
        -> (size: Int, sha256: String) {
        var bytes = [UInt8](repeating: 0xAB, count: Pool.fileSize)
        for shard in 0..<Pool.shardCount {
            for local in 0..<Pool.rowsPerShard {
                let row = shard * Pool.rowsPerShard + local
                let offset = shard * Pool.shardBytes
                    + (local / Pool.rowsPerBlock) * Pool.blockStride
                    + (local % Pool.rowsPerBlock) * Pool.rowStride
                for column in 0..<Pool.rowDim {
                    let bits = Quantization.bf16Bits(Pool.value(row: row, column: column))
                    bytes[offset + column * 2] = UInt8(truncatingIfNeeded: bits)
                    bytes[offset + column * 2 + 1] = UInt8(truncatingIfNeeded: bits >> 8)
                }
            }
        }
        let url = dir.appendingPathComponent(Pool.file)
        try Data(bytes).write(to: url)
        return (bytes.count, try Sha256Verifier.hashFile(at: url))
    }
}

extension ArchConfig {
    /// Tiny Qwen3.8-Flash-Next baseline: 4 layers in the production 3:1 mask
    /// shape (linear, linear, linear, full), a 4-stream low-rank
    /// hyper-connected residual, a QSA indexer on the full layer, a PLE block
    /// at one-indexed layer id 2 (`layers[1]`), and 8 routed experts top-2
    /// beside a gated shared expert.
    ///
    /// Numbers are toy but respect the divisibility every path assumes:
    /// `hidden % 64 == 0` and `hcLowRank % 64 == 0` for group-64 columns,
    /// `hidden % rowDim == 0` for the PLE head split, `Hv % Hk == 0` for GDN.
    static func qwen38FlashNextToy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 2, 2, 1],
            hiddenActivation: "silu",
            family: .qwen38flashnext,
            attnOutputGate: true,
            attentionScale: 0.176_776_695_296_636_88,   // 32^-0.5
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 6,
                keyHeadDim: 32, valueHeadDim: 32,
                convKernelSize: 4),
            numSharedExperts: 1,
            flashNext: FlashNextConfig(
                hcCount: 4,
                hcLowRank: 64,
                indexerNumHeads: 2,
                indexerHeadDim: 32,
                indexerNumKVHeads: 1,
                indexerBudget: 32,
                indexerCompressRatio: 4,
                pleLayerIDs: [FlashNextToySynthetic.pleLayer + 1],
                pleNgramShardCount: FlashNextToySynthetic.Pool.shardCount,
                pleNgramVocabSizeBase: 1_000,
                pleConvKernelSize: 4,
                pleEosTokenID: 1_023))
    }
}
