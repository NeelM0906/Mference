import Foundation
@testable import Mference
@testable import MferenceRepackCore

/// Synthetic Qwen 3.8 toy fixture: a tiny runnable `.gturbo/` directory with
/// the qwen38 tensor-name contract (linear_attn.* on mask-2 layers,
/// self_attn.* with packed [query ; gate] q_proj on mask-1 layers, a dense
/// mlp.{gate,up,down}_proj per layer, untied lm_head, NO router, NO experts,
/// NO shared-expert tensors). Mirrors `QwenToySynthetic` for the qwen36 toy.
enum Qwen38ToySynthetic {

    /// Build the toy directory in a temp dir and return its URL.
    static func write() throws -> URL {
        let toy = ArchConfig.qwen38Toy()
        let la = toy.linearAttention
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-qwen38-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let u16 = MemoryLayout<UInt16>.stride

        func int4AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name,
                                dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols / 2),
                                scaleBytes: auxBytes,
                                biasBytes: auxBytes)
        }

        func int8AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name,
                                dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols),
                                scaleBytes: auxBytes,
                                biasBytes: auxBytes)
        }

        func bf16Spec(_ name: String, shape: [UInt32], count: Int) -> ResidentSpec {
            ResidentSpec(name: name,
                         dtype: 1,
                         shape: shape,
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0,
                         biasBytes: 0)
        }

        // 1. Resident specs. The dense MLP goes through the shared-expert
        // runtime, whose weight bits default to 8 when the manifest carries
        // no quant block (toy manifests may omit it), so the MLP tensors are
        // int8 affine here — exactly like the qwen36 toy's shared expert.
        var specs: [ResidentSpec] = [
            int4AffineSpec("language_model.model.embed_tokens.weight",
                           rows: toy.vocabSize, cols: d),
            int4AffineSpec("language_model.lm_head.weight",
                           rows: toy.vocabSize, cols: d),
            bf16Spec("language_model.model.norm.weight",
                     shape: [UInt32(d), 0, 0, 0], count: d),
        ]
        for L in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            specs.append(bf16Spec("\(prefix).input_layernorm.weight",
                                  shape: [UInt32(d), 0, 0, 0], count: d))
            specs.append(bf16Spec("\(prefix).post_attention_layernorm.weight",
                                  shape: [UInt32(d), 0, 0, 0], count: d))
            specs.append(int8AffineSpec("\(prefix).mlp.gate_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.up_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.down_proj.weight",
                                        rows: d, cols: toy.intermediateSize))
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
                                      shape: [UInt32(la.qkvDim), UInt32(la.convKernelSize), 1, 0],
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
            }
        }

        // 2. Serialize the resident index + payload.
        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes  = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = indexBytes
        for spec in specs {
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name,
                dtype: spec.dtype,
                logicalShape4: spec.shape,
                fileOffset: weightOffset,
                sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset,
                scaleSize: spec.scaleBytes,
                biasOffset: biasOffset,
                biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil,
                sourceBiases: nil))
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
                let dst = base.advanced(by: entriesBase + i * entryBytes)
                GTurboBinary.writeIndexEntry(into: dst, entry: e,
                                             nameOffset: nameAbsOffsets[i])
            }
            _ = stringTable.withUnsafeBytes { sb in
                memcpy(base.advanced(by: stringTableBase), sb.baseAddress!, stringTable.count)
            }
            // Quantized tensors: deterministic varied nibbles (splitmix-style
            // LCG over the byte index) with varied small scales and biases, so
            // dot products exercise real, order-sensitive FP accumulation
            // (uniform fills would make every reduction order agree by
            // construction and mask parity bugs). BF16 tensors: 1.0 plus a
            // small deterministic ripple.
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
            for entry in entries where entry.dtype == 1 {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    dst[i] = Quantization.bf16Bits(1.0 + 0.03 * Float(i % 5) - 0.06)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. layout.json: zero experts per layer, so every layer entry is
        // empty and no blob file is written (the dense-layer contract the
        // Inkling toys already exercise).
        let expertStride: UInt64 = 16384
        var layersArr: [[String: Any]] = []
        for L in 0..<toy.numLayers {
            layersArr.append([
                "layer": L,
                "file": String(format: "layer_%02d.bin", L),
                "experts": [[String: Any]](),
            ])
        }
        let layoutRoot: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": layersArr,
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layoutRoot, options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 4. manifest.json (arch v2 with the qwen38 family fields).
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": Int(totalBytes), "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
        ]

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
            "linearNumKHeads": la.numKHeads,
            "linearNumVHeads": la.numVHeads,
            "linearKeyHeadDim": la.keyHeadDim,
            "linearValueHeadDim": la.valueHeadDim,
            "linearConvKernelSize": la.convKernelSize,
            "numSharedExperts": toy.numSharedExperts,
            "numDenseLayers": toy.numDenseLayers,
            "denseIntermediateSize": toy.denseIntermediateSize,
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "qwen38-toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}

extension Qwen38ToySynthetic {

    /// Write a toy BF16 MTP shard (the 15 `mtp.*` tensors of the HF layout,
    /// sized for `ArchConfig.qwen38Toy`) for `MTPAttachTool`. Norm vectors
    /// are written zero-centered, matching the HF convention the attach
    /// step's `+1` conversion expects. Values are deterministic and varied.
    static func writeMTPShard() throws -> URL {
        let toy = ArchConfig.qwen38Toy()
        let d = toy.hiddenSize
        let f = toy.intermediateSize
        let qDim = toy.numHeads * toy.fullHeadDim
        let kvDim = toy.numFullKVHeads * toy.fullHeadDim

        // (name, shape, zeroCenteredNorm)
        let tensors: [(String, [Int], Bool)] = [
            ("mtp.fc.weight", [d, 2 * d], false),
            ("mtp.pre_fc_norm_embedding.weight", [d], true),
            ("mtp.pre_fc_norm_hidden.weight", [d], true),
            ("mtp.norm.weight", [d], true),
            ("mtp.layers.0.input_layernorm.weight", [d], true),
            ("mtp.layers.0.post_attention_layernorm.weight", [d], true),
            ("mtp.layers.0.self_attn.q_proj.weight", [2 * qDim, d], false),
            ("mtp.layers.0.self_attn.k_proj.weight", [kvDim, d], false),
            ("mtp.layers.0.self_attn.v_proj.weight", [kvDim, d], false),
            ("mtp.layers.0.self_attn.o_proj.weight", [d, qDim], false),
            ("mtp.layers.0.self_attn.q_norm.weight", [toy.fullHeadDim], true),
            ("mtp.layers.0.self_attn.k_norm.weight", [toy.fullHeadDim], true),
            ("mtp.layers.0.mlp.gate_proj.weight", [f, d], false),
            ("mtp.layers.0.mlp.up_proj.weight", [f, d], false),
            ("mtp.layers.0.mlp.down_proj.weight", [d, f], false),
        ]

        var headerEntries: [String] = []
        var payload = Data()
        var lcg: UInt64 = 0x5DEECE66D
        func nextFloat() -> Float {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            return Float(lcg >> 40) / Float(1 << 24) - 0.5
        }
        for (name, shape, isNorm) in tensors {
            let count = shape.reduce(1, *)
            let begin = payload.count
            var bits = [UInt16](repeating: 0, count: count)
            for i in 0..<count {
                let value = isNorm ? 0.3 * nextFloat() : 0.25 * nextFloat()
                bits[i] = Quantization.bf16Bits(value)
            }
            bits.withUnsafeBufferPointer {
                payload.append(UnsafeRawBufferPointer($0).bindMemory(to: UInt8.self))
            }
            let dims = shape.map(String.init).joined(separator: ",")
            headerEntries.append(
                "\"\(name)\":{\"dtype\":\"BF16\",\"shape\":[\(dims)],"
                + "\"data_offsets\":[\(begin),\(payload.count)]}")
        }
        let headerJSON = "{" + headerEntries.joined(separator: ",") + "}"
        let headerBytes = Data(headerJSON.utf8)
        var file = Data()
        var headerLen = UInt64(headerBytes.count).littleEndian
        withUnsafeBytes(of: &headerLen) { file.append(contentsOf: $0) }
        file.append(headerBytes)
        file.append(payload)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen38-toy-mtp-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }
}

extension ArchConfig {
    /// Tiny Qwen 3.8 baseline: 4 dense layers with the production 3:1 mask
    /// shape (linear, linear, linear, full), an attn-gated full layer, GDN
    /// with Hv/Hk ratio 3 (Hv 6, Hk 2 — off the fused Hv-32 kernels, like
    /// production's 48/16), one dense SwiGLU MLP per layer, untied lm_head,
    /// no experts. Numbers are intentionally toy but respect every kernel
    /// divisibility constraint (D % 64, keyHeadDim % 32, even rotaryDim,
    /// Hv % Hk == 0).
    static func qwen38Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 0,
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
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 2, 2, 1],
            hiddenActivation: "silu",
            family: .qwen38,
            attnOutputGate: true,
            attentionScale: 0.125,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 6,
                keyHeadDim: 32, valueHeadDim: 32,
                convKernelSize: 4),
            numSharedExperts: 0,
            numDenseLayers: 4,
            denseIntermediateSize: 128
        )
    }
}
