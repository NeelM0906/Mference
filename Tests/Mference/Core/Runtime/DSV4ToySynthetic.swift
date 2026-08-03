import Foundation
@testable import Mference
@testable import MferenceRepackCore

/// Synthetic DeepSeek-V4 Flash toy fixture: a tiny runnable `.gturbo/`
/// directory with the DSV4 tensor-name contract (bare `model.` trunk prefix,
/// low-rank `attn.wq_a`/`wq_b`, shared-KV `attn.wkv`, grouped `attn.wo_a`/
/// `wo_b`, per-layer `attn_hc`/`ffn_hc` mixes, a CSA compressor + lightning
/// indexer, an HCA compressor, hash-routed leading layers, and INT2 routed
/// experts). Mirrors `QwenToySynthetic.write()`.
///
/// The layer mask is `[0, 0, 3, 4]`, so one fixture covers every DSV4 layer
/// flavour: two window-only layers, a CSA layer (compressor + indexer key
/// emission), and an HCA layer. `numHashRoutedLayers = 3` puts layers 0-2 on
/// the frozen `tid2eid` table and leaves layer 3 on the learned router.
enum DSV4ToySynthetic {

    /// Deterministic 64-bit PRNG so the fixture bytes are reproducible.
    private struct Rng {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
        mutating func symmetric(_ scale: Float) -> Float { (unit() * 2 - 1) * scale }
    }

    private enum Fill {
        /// Packed quantized weights + BF16 scales/biases.
        case quantized
        /// BF16 payload, values drawn around `center` +/- `spread`.
        case bf16(center: Float, spread: Float)
        /// FP32 payload, values drawn around `center` +/- `spread`.
        case fp32(center: Float, spread: Float)
        /// UInt32 payload in `0..<bound`.
        case u32(bound: UInt32)
    }

    private struct ResidentSpec {
        let name: String
        let dtype: UInt8
        let shape: [UInt32]
        let weightBytes: UInt64
        let scaleBytes: UInt64
        let biasBytes: UInt64
        let fill: Fill
    }

    // swiftlint:disable:next function_body_length
    static func write() throws -> URL {
        let toy = ArchConfig.deepseekV4Toy()
        let ca = toy.compressedAttention
        let hc = toy.hyperConnections
        let d = toy.hiddenSize
        let u16 = MemoryLayout<UInt16>.stride
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-dsv4-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        func int4Spec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = cols / Quantization.groupSize
            let aux = UInt64(rows * groups * u16)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols / 2),
                                scaleBytes: aux, biasBytes: aux, fill: .quantized)
        }
        func bf16Spec(_ name: String, shape: [UInt32], count: Int,
                      center: Float, spread: Float) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 1, shape: shape,
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0, biasBytes: 0,
                         fill: .bf16(center: center, spread: spread))
        }
        func fp32Spec(_ name: String, shape: [UInt32], count: Int,
                      center: Float, spread: Float) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 3, shape: shape,
                         weightBytes: UInt64(count * MemoryLayout<Float>.stride),
                         scaleBytes: 0, biasBytes: 0,
                         fill: .fp32(center: center, spread: spread))
        }
        func u32Spec(_ name: String, shape: [UInt32], count: Int,
                     bound: UInt32) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 0, shape: shape,
                         weightBytes: UInt64(count * MemoryLayout<UInt32>.stride),
                         scaleBytes: 0, biasBytes: 0, fill: .u32(bound: bound))
        }

        // 1. Resident specs.
        let hcRows = (2 + hc.mult) * hc.mult
        let hcCols = hc.mult * d
        var specs: [ResidentSpec] = [
            int4Spec("model.embed_tokens.weight", rows: toy.vocabSize, cols: d),
            int4Spec("lm_head.weight", rows: toy.vocabSize, cols: d),
            bf16Spec("model.norm.weight", shape: [UInt32(d), 0, 0, 0], count: d,
                     center: 1.0, spread: 0.02),
            fp32Spec("model.hc_head.fn",
                     shape: [UInt32(hcRows), UInt32(hcCols), 0, 0],
                     count: hcRows * hcCols, center: 0, spread: 0.01),
            fp32Spec("model.hc_head.base", shape: [UInt32(hcRows), 0, 0, 0],
                     count: hcRows, center: 0, spread: 0.02),
            fp32Spec("model.hc_head.scale", shape: [3, 0, 0, 0], count: 3,
                     center: 1.0, spread: 0.0),
        ]

        let qDim = toy.numHeads * toy.fullHeadDim
        for L in 0..<toy.numLayers {
            let p = "model.layers.\(L)"
            specs.append(bf16Spec("\(p).attn_norm.weight",
                                  shape: [UInt32(d), 0, 0, 0], count: d,
                                  center: 1.0, spread: 0.02))
            specs.append(bf16Spec("\(p).ffn_norm.weight",
                                  shape: [UInt32(d), 0, 0, 0], count: d,
                                  center: 1.0, spread: 0.02))
            specs.append(int4Spec("\(p).attn.wq_a.weight", rows: ca.qLoraRank, cols: d))
            specs.append(bf16Spec("\(p).attn.q_norm.weight",
                                  shape: [UInt32(ca.qLoraRank), 0, 0, 0],
                                  count: ca.qLoraRank, center: 1.0, spread: 0.02))
            specs.append(int4Spec("\(p).attn.wq_b.weight", rows: qDim, cols: ca.qLoraRank))
            specs.append(int4Spec("\(p).attn.wkv.weight", rows: toy.fullHeadDim, cols: d))
            specs.append(bf16Spec("\(p).attn.kv_norm.weight",
                                  shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                  count: toy.fullHeadDim, center: 1.0, spread: 0.02))
            specs.append(fp32Spec("\(p).attn.attn_sink",
                                  shape: [UInt32(toy.numHeads), 0, 0, 0],
                                  count: toy.numHeads, center: 0, spread: 0.1))
            specs.append(int4Spec("\(p).attn.wo_a.weight",
                                  rows: ca.oGroups * ca.oLoraRank,
                                  cols: qDim / ca.oGroups))
            specs.append(int4Spec("\(p).attn.wo_b.weight",
                                  rows: d, cols: ca.oGroups * ca.oLoraRank))
            for site in ["attn_hc", "ffn_hc"] {
                specs.append(fp32Spec("\(p).\(site).fn",
                                      shape: [UInt32(hcRows), UInt32(hcCols), 0, 0],
                                      count: hcRows * hcCols, center: 0, spread: 0.01))
                specs.append(fp32Spec("\(p).\(site).base",
                                      shape: [UInt32(hcRows), 0, 0, 0],
                                      count: hcRows, center: 0, spread: 0.02))
                specs.append(fp32Spec("\(p).\(site).scale", shape: [3, 0, 0, 0],
                                      count: 3, center: 1.0, spread: 0.0))
            }
            // The DSV4 router gate is unquantized BF16.
            specs.append(bf16Spec("\(p).ffn.gate.weight",
                                  shape: [UInt32(toy.numExperts), UInt32(d), 0, 0],
                                  count: toy.numExperts * d, center: 0, spread: 0.03))
            if toy.layerIsHashRouted(L) {
                specs.append(u32Spec("\(p).ffn.gate.tid2eid",
                                     shape: [UInt32(toy.vocabSize),
                                             UInt32(toy.topKExperts), 0, 0],
                                     count: toy.vocabSize * toy.topKExperts,
                                     bound: UInt32(toy.numExperts)))
            } else {
                specs.append(fp32Spec("\(p).ffn.gate.e_score_correction_bias",
                                      shape: [UInt32(toy.numExperts), 0, 0, 0],
                                      count: toy.numExperts, center: 0, spread: 0.02))
            }
            specs.append(int4Spec("\(p).ffn.shared_experts.gate_proj.weight",
                                  rows: toy.intermediateSize, cols: d))
            specs.append(int4Spec("\(p).ffn.shared_experts.up_proj.weight",
                                  rows: toy.intermediateSize, cols: d))
            specs.append(int4Spec("\(p).ffn.shared_experts.down_proj.weight",
                                  rows: d, cols: toy.intermediateSize))

            let isCSA = toy.layerIsCSA(L)
            let isHCA = toy.layerIsHCA(L)
            if isCSA || isHCA {
                let rate = isCSA ? ca.csaCompressRate : ca.hcaCompressRate
                let rowWidth = isCSA ? 2 * toy.fullHeadDim : toy.fullHeadDim
                specs.append(int4Spec("\(p).attn.compressor.wkv.weight",
                                      rows: rowWidth, cols: d))
                specs.append(int4Spec("\(p).attn.compressor.wgate.weight",
                                      rows: rowWidth, cols: d))
                specs.append(bf16Spec("\(p).attn.compressor.norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim, center: 1.0, spread: 0.02))
                specs.append(bf16Spec("\(p).attn.compressor.ape",
                                      shape: [UInt32(rate), UInt32(rowWidth), 0, 0],
                                      count: rate * rowWidth, center: 0, spread: 0.1))
            }
            if isCSA {
                let idxWidth = 2 * ca.indexHeadDim
                specs.append(int4Spec("\(p).attn.indexer.compressor.wkv.weight",
                                      rows: idxWidth, cols: d))
                specs.append(int4Spec("\(p).attn.indexer.compressor.wgate.weight",
                                      rows: idxWidth, cols: d))
                specs.append(bf16Spec("\(p).attn.indexer.compressor.norm.weight",
                                      shape: [UInt32(ca.indexHeadDim), 0, 0, 0],
                                      count: ca.indexHeadDim, center: 1.0, spread: 0.02))
                specs.append(bf16Spec("\(p).attn.indexer.compressor.ape",
                                      shape: [UInt32(ca.csaCompressRate),
                                              UInt32(idxWidth), 0, 0],
                                      count: ca.csaCompressRate * idxWidth,
                                      center: 0, spread: 0.1))
                specs.append(int4Spec("\(p).attn.indexer.wq_b.weight",
                                      rows: ca.indexNHeads * ca.indexHeadDim,
                                      cols: ca.qLoraRank))
                specs.append(int4Spec("\(p).attn.indexer.weights_proj.weight",
                                      rows: ca.indexNHeads, cols: d))
            }
        }

        // 2. Serialize the resident index + payload.
        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        // The payload base and every sub-tensor offset must be 16-byte
        // aligned: the resident region starts right after a string table whose
        // length depends on the tensor names, and DSV4 binds FP32 tensors
        // (attn_sink, the hyper-connection mixes) straight out of this file.
        // A 2-byte-aligned offset is legal for the BF16/INT4 tensors but
        // silently corrupts a `device const float*` binding.
        func align16(_ value: UInt64) -> UInt64 { (value + 15) & ~15 }
        let indexBytes = align16(UInt64(stringTableBase + stringTable.count))

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = indexBytes
        for spec in specs {
            let weightOffset = align16(payloadCursor)
            let scaleOffset = spec.scaleBytes > 0 ? align16(weightOffset + spec.weightBytes) : 0
            let biasOffset = spec.biasBytes > 0 ? align16(scaleOffset + spec.scaleBytes) : 0
            entries.append(ResidentEntry(
                name: spec.name, dtype: spec.dtype, logicalShape4: spec.shape,
                fileOffset: weightOffset, sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset, scaleSize: spec.scaleBytes,
                biasOffset: biasOffset, biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil, sourceBiases: nil))
            payloadCursor = biasOffset > 0
                ? biasOffset + spec.biasBytes
                : (scaleOffset > 0 ? scaleOffset + spec.scaleBytes
                                   : weightOffset + spec.weightBytes)
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
                    into: base.advanced(by: entriesBase + i * entryBytes),
                    entry: e, nameOffset: nameAbsOffsets[i])
            }
            _ = stringTable.withUnsafeBytes { sb in
                memcpy(base.advanced(by: stringTableBase), sb.baseAddress!,
                       stringTable.count)
            }
            var rng = Rng(state: 0x5DEE_CE66_D00D_1234)
            for (i, spec) in specs.enumerated() {
                let entry = entries[i]
                let dst = base.advanced(by: Int(entry.fileOffset))
                switch spec.fill {
                case .quantized:
                    let bytes = dst.assumingMemoryBound(to: UInt8.self)
                    for j in 0..<Int(entry.sizeBytes) {
                        bytes[j] = UInt8(truncatingIfNeeded: rng.next())
                    }
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    let biases = base.advanced(by: Int(entry.biasOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    // Zero-centred, small-magnitude affine dequant: nibble
                    // codes 0...15 map to roughly +/- 8 * scale, so activations
                    // stay well inside FP16 range through every layer.
                    for j in 0..<(Int(entry.scaleSize) / u16) {
                        let scale = 0.003 + rng.unit() * 0.003
                        scales[j] = Quantization.bf16Bits(scale)
                        biases[j] = Quantization.bf16Bits(-7.5 * scale)
                    }
                case .bf16(let center, let spread):
                    let out = dst.assumingMemoryBound(to: UInt16.self)
                    for j in 0..<(Int(entry.sizeBytes) / u16) {
                        out[j] = Quantization.bf16Bits(center + rng.symmetric(spread))
                    }
                case .fp32(let center, let spread):
                    let out = dst.assumingMemoryBound(to: Float.self)
                    for j in 0..<(Int(entry.sizeBytes) / 4) {
                        out[j] = center + rng.symmetric(spread)
                    }
                case .u32(let bound):
                    let out = dst.assumingMemoryBound(to: UInt32.self)
                    for j in 0..<(Int(entry.sizeBytes) / 4) {
                        out[j] = UInt32(rng.next() % UInt64(bound))
                    }
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. Packed experts: INT2 affine gate/up/down, page-aligned stride.
        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        /// One INT2 affine row: four 2-bit codes per byte, low bits first,
        /// with a BF16 scale/bias per group of `Quantization.groupSize`.
        func int2Row(cols: Int, rng: inout Rng)
            -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            var packed = [UInt8](repeating: 0, count: cols / 4)
            for i in 0..<cols {
                let q = UInt8(rng.next() % 4)
                packed[i / 4] |= q << UInt8(2 * (i % 4))
            }
            var scales: [UInt16] = []
            var biases: [UInt16] = []
            for _ in 0..<(cols / Quantization.groupSize) {
                let scale = 0.006 + rng.unit() * 0.006
                scales.append(Quantization.bf16Bits(scale))
                biases.append(Quantization.bf16Bits(-1.5 * scale))
            }
            return (packed, scales, biases)
        }

        func toyExpertBlob(layer: Int, expert: Int)
            -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var rng = Rng(state: UInt64(layer &* 1_000 &+ expert &+ 7) &* 0x2545_F491_4F6C_DD1D)
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]
            func addProjection(prefix: String, rows: Int, cols: Int) {
                let quantized = (0..<rows).map { _ in int2Row(cols: cols, rng: &rng) }
                let packedOffset = bytes.count
                for row in quantized { bytes.append(contentsOf: row.packed) }
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols], "bits": 2,
                ]
                let scalesOffset = bytes.count
                for row in quantized { appendU16(row.scales, to: &bytes) }
                tensors["\(prefix)_scales"] = [
                    "offset": scalesOffset, "size": bytes.count - scalesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
                let biasesOffset = bytes.count
                for row in quantized { appendU16(row.biases, to: &bytes) }
                tensors["\(prefix)_biases"] = [
                    "offset": biasesOffset, "size": bytes.count - biasesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
            }
            addProjection(prefix: "gate", rows: toy.moeIntermediateSize, cols: d)
            addProjection(prefix: "up", rows: toy.moeIntermediateSize, cols: d)
            addProjection(prefix: "down", rows: d, cols: toy.moeIntermediateSize)
            return (bytes, tensors)
        }

        let expertStride = UInt64(getpagesize())
        let layerBytes = Int(expertStride) * toy.numExperts
        var layoutLayers: [[String: Any]] = []
        for L in 0..<toy.numLayers {
            var payload = Data(count: layerBytes)
            var experts: [[String: Any]] = []
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(layer: L, expert: E)
                precondition(blob.bytes.count <= Int(expertStride),
                             "toy expert blob (\(blob.bytes.count) B) exceeds stride \(expertStride)")
                let baseB = E * Int(expertStride)
                for (i, byte) in blob.bytes.enumerated() { payload[baseB + i] = byte }
                experts.append([
                    "expert": E,
                    "offset": UInt64(E) * expertStride,
                    "size": expertStride,
                    "tensors": blob.tensors,
                ])
            }
            let basename = String(format: "layer_%02d.bin", L)
            try payload.write(to: exp.appendingPathComponent(basename))
            layoutLayers.append(["layer": L, "file": basename, "experts": experts])
        }
        var layerShaByName: [String: String] = [:]
        for L in 0..<toy.numLayers {
            let basename = String(format: "layer_%02d.bin", L)
            layerShaByName["packed_experts/\(basename)"] =
                try Sha256Verifier.hashFile(at: exp.appendingPathComponent(basename))
        }

        // 4. layout.json
        let layoutData = try JSONSerialization.data(
            withJSONObject: [
                "expertStride": expertStride,
                "numLayers": toy.numLayers,
                "expertsPerLayer": toy.numExperts,
                "layers": layoutLayers,
            ] as [String: Any], options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 5. manifest.json
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": totalBytes, "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
        ]
        for (rel, sha) in layerShaByName {
            files[rel] = ["size": layerBytes, "sha256": sha]
        }
        func quantSlot(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": "affine", "scaleType": "bf16",
             "biasType": "bf16", "groupSize": Quantization.groupSize]
        }
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
            "caQLoraRank": ca.qLoraRank, "caOLoraRank": ca.oLoraRank,
            "caOGroups": ca.oGroups, "caRopeHeadDim": ca.ropeHeadDim,
            "caIndexNHeads": ca.indexNHeads, "caIndexHeadDim": ca.indexHeadDim,
            "caIndexTopK": ca.indexTopK,
            "caCSACompressRate": ca.csaCompressRate,
            "caHCACompressRate": ca.hcaCompressRate,
            "caCompressRopeTheta": ca.compressRopeTheta,
            "caRopeScalingFactor": ca.ropeScalingFactor,
            "caRopeScalingOriginalMax": ca.ropeScalingOriginalMax,
            "caRopeScalingBetaFast": ca.ropeScalingBetaFast,
            "caRopeScalingBetaSlow": ca.ropeScalingBetaSlow,
            "hcMult": hc.mult, "hcSinkhornIters": hc.sinkhornIters, "hcEps": hc.eps,
            "numHashRoutedLayers": toy.numHashRoutedLayers,
            "routerScoringFunc": toy.routerScoringFunc,
            "routedScalingFactor": toy.routedScalingFactor,
            "swigluLimit": toy.swigluLimit,
            "linearNumKHeads": 0, "linearNumVHeads": 0,
            "linearKeyHeadDim": 0, "linearValueHeadDim": 0,
            "linearConvKernelSize": 0,
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "dsv4-toy",
            "arch": archDict,
            "quant": [
                "embedding": quantSlot(4), "attention": quantSlot(4),
                "router": quantSlot(8), "sharedExpert": quantSlot(4),
                "routedExpert": quantSlot(2),
            ],
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}

extension ArchConfig {
    /// Tiny DeepSeek-V4 Flash baseline. Toy sizes everywhere they are free,
    /// but every hard kernel constraint is respected:
    ///
    ///  * `fullHeadDim == 512` — `dsv4_attention_decode` is written for the
    ///    absorbed-MLA KV width and `DSV4Kernels` preconditions on it.
    ///  * `numHeads` a multiple of `DSV4Kernels.attentionHeadsPerThreadgroup`.
    ///  * `hyperConnections.mult <= 4` — `dsv4_hc_weights` sizes its mix
    ///    scratch for that.
    ///  * `numHeads * fullHeadDim / oGroups` a multiple of the quant group.
    ///  * `topKExperts == MoEDeepseekV4.topK` — the routed kernels are k6.
    ///  * `numExperts > 8` so a chunk's live experts span more than one
    ///    routed tile.
    ///  * `indexTopK == 12` with `csaCompressRate == 4` puts the chunked
    ///    prefill's lightning-selection cutover at absolute position 48, so a
    ///    single fixture exercises both the batched path and the
    ///    token-by-token fallback.
    static func deepseekV4Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 256,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 8,
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: 512,
            fullHeadDim: 512,
            vocabSize: 256,
            slidingWindow: 16,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 10_000.0,
            partialRotaryFactor: 0.125,
            numLayers: 4,
            numExperts: 16,
            topKExperts: 6,
            tieWordEmbeddings: false,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 0, 3, 4],
            hiddenActivation: "silu",
            family: .deepseekV4Flash,
            attnOutputGate: false,
            attentionScale: 0.044194173824159216,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            compressedAttention: CompressedAttentionConfig(
                qLoraRank: 128, oLoraRank: 64, oGroups: 2,
                ropeHeadDim: 64,
                indexNHeads: 2, indexHeadDim: 64, indexTopK: 12,
                csaCompressRate: 4, hcaCompressRate: 8,
                compressRopeTheta: 160_000.0),
            hyperConnections: HyperConnectionConfig(mult: 2, sinkhornIters: 4,
                                                    eps: 1e-6),
            numHashRoutedLayers: 3,
            routerScoringFunc: "sqrtsoftplus",
            routedScalingFactor: 1.5,
            swigluLimit: 10.0)
    }
}
