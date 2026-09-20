import Foundation
@testable import Mference
@testable import MferenceRepackCore

/// Nonzero, deterministic Inkling install for ordinary CI. Exercises both
/// dense/MoE layers, local/global attention, four convolution states, shared
/// sinks, BF16/FP32 scalar gains and padded vocabulary without real weights.
enum InklingToySynthetic {
    static let config = ArchConfig(
        hiddenSize: 64, intermediateSize: 64, moeIntermediateSize: 64,
        numHeads: 4, numKVHeads: 2, numFullKVHeads: 2,
        headDim: 32, fullHeadDim: 32, vocabSize: 256, slidingWindow: 32,
        finalLogitSoftcap: 0, ropeTheta: 0, fullRopeTheta: 0,
        partialRotaryFactor: 0, numLayers: 4, numExperts: 8, topKExperts: 6,
        tieWordEmbeddings: false, attentionKEqV: false,
        fullAttentionLayerMask: [0, 1, 0, 1], hiddenActivation: "silu",
        family: .inklingSmall, attentionScale: 1.0 / 32,
        embeddingScaledBySqrtHidden: false, routerScaled: false,
        ffnSandwichNorms: false, routerScoringFunc: "sigmoid", routedScalingFactor: 8,
        relativePosition: RelativePositionConfig(dRel: 16, extent: 64, projDim: 64,
            logScalingFloor: 40, logScalingAlpha: 0.1),
        sconvKernelSize: 4, numSharedExperts: 2, numDenseLayers: 2,
        denseIntermediateSize: 128, sharedExpertSink: true, embedNormEnabled: true,
        logitsWidthMultiplier: 16, routerGateBias: true, routerNormAfterTopK: true,
        routerGlobalScale: true, unpaddedVocabSize: 250)

    private struct Tensor {
        let name: String
        let dtype: UInt8
        let shape: [UInt32]
        let weights: Data
        let scales: Data
        let biases: Data
    }

    static func write() throws -> URL {
        let cfg = config
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-inkling-toy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("packed_experts"),
                                               withIntermediateDirectories: true)
        var succeeded = false
        defer { if !succeeded { try? FileManager.default.removeItem(at: directory) } }
        var rng: UInt64 = 0x62781A3C
        func sample(_ scale: Float) -> Float {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return (Float(rng >> 40) / Float(1 << 24) * 2 - 1) * scale
        }
        func shorts(_ values: [UInt16]) -> Data {
            var bytes = Data()
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
            return bytes
        }
        func quantized(_ name: String, rows: Int, cols: Int, stack: Int = 1) -> Tensor {
            var weights = Data(), scales = Data(), biases = Data()
            for _ in 0..<(rows * stack) {
                let row = Quantization.quantizeInt4Affine((0..<cols).map { _ in sample(0.04) })
                weights.append(contentsOf: row.packed)
                scales.append(shorts(row.scales))
                biases.append(shorts(row.biases))
            }
            let shape = stack == 1 ? [rows, cols, 0, 0] : [stack, rows, cols, 0]
            return Tensor(name: name, dtype: 0, shape: shape.map(UInt32.init),
                          weights: weights, scales: scales, biases: biases)
        }
        func dense(_ name: String, shape: [Int], center: Float = 0,
                   spread: Float = 0.02, fp32: Bool = false) -> Tensor {
            let count = shape.reduce(1, *)
            var bytes = Data()
            for _ in 0..<count {
                let value = center + sample(spread)
                if fp32 {
                    let bits = value.bitPattern
                    for shift in [0, 8, 16, 24] { bytes.append(UInt8(truncatingIfNeeded: bits >> shift)) }
                } else { bytes.append(shorts([Quantization.bf16Bits(value)])) }
            }
            return Tensor(name: name, dtype: fp32 ? 3 : 1,
                shape: (shape + Array(repeating: 0, count: 4 - shape.count)).map(UInt32.init),
                weights: bytes, scales: Data(), biases: Data())
        }
        var tensors = [quantized("model.llm.embed.weight", rows: cfg.vocabSize, cols: 64),
                       quantized("model.llm.unembed.weight", rows: cfg.vocabSize, cols: 64),
                       dense("model.llm.embed_norm.weight", shape: [64], center: 1),
                       dense("model.llm.norm.weight", shape: [64], center: 1)]
        for layer in 0..<cfg.numLayers {
            let p = "model.llm.layers.\(layer)"
            for site in ["attn_norm", "mlp_norm"] {
                tensors.append(dense("\(p).\(site).weight", shape: [64], center: 1))
            }
            for (name, rows, cols) in [("wq_du", 128, 64), ("wk_dv", 64, 64),
                                        ("wv_dv", 64, 64), ("wo_ud", 64, 128), ("wr_du", 64, 64)] {
                tensors.append(quantized("\(p).attn.\(name).weight", rows: rows, cols: cols))
            }
            for site in ["q_norm", "k_norm"] {
                tensors.append(dense("\(p).attn.\(site).weight", shape: [32], center: 1))
            }
            let extent = cfg.fullAttentionLayerMask[layer] == 0 ? cfg.slidingWindow : cfg.relativePosition.extent
            tensors.append(dense("\(p).attn.rel_logits_proj.proj", shape: [16, extent]))
            for site in ["attn.k_sconv", "attn.v_sconv", "attn_sconv", "mlp_sconv"] {
                tensors.append(dense("\(p).\(site).weight", shape: [64, 4, 1], spread: 0.1))
            }
            if layer < cfg.numDenseLayers {
                for role in ["gate", "up", "down"] {
                    tensors.append(quantized("\(p).mlp.\(role)_proj.weight",
                        rows: role == "down" ? 64 : 128, cols: role == "down" ? 128 : 64))
                }
                tensors.append(dense("\(p).mlp.global_scale", shape: [1], center: 0.75, spread: 0))
            } else {
                tensors.append(dense("\(p).mlp.gate.weight", shape: [10, 64]))
                tensors.append(dense("\(p).mlp.gate.bias", shape: [8], fp32: true))
                tensors.append(dense("\(p).mlp.gate.global_scale", shape: [1], center: 0.5,
                                     spread: 0, fp32: true))
                for role in ["gate", "up", "down"] {
                    tensors.append(quantized("\(p).mlp.shared_experts.\(role)_proj.weight",
                                             rows: 64, cols: 64, stack: 2))
                }
            }
        }
        let names = tensors.map(\.name)
        let strings = Data(names.joined().utf8)
        let stringBase = GTurboBinary.indexHeaderBytes + tensors.count * GTurboBinary.indexEntryBytes
        let indexBytes = stringBase + strings.count
        var cursor = indexBytes
        var entries: [ResidentEntry] = []
        for tensor in tensors {
            // Keep scalar FP32 and BF16 loads naturally aligned.
            cursor = (cursor + 3) / 4 * 4
            let weight = cursor
            let scale = tensor.scales.isEmpty ? 0 : weight + tensor.weights.count
            let bias = tensor.biases.isEmpty ? 0 : scale + tensor.scales.count
            entries.append(ResidentEntry(name: tensor.name, dtype: tensor.dtype, logicalShape4: tensor.shape,
                fileOffset: UInt64(weight), sizeBytes: UInt64(tensor.weights.count),
                scaleOffset: UInt64(scale), scaleSize: UInt64(tensor.scales.count),
                biasOffset: UInt64(bias), biasSize: UInt64(tensor.biases.count), quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(tensor.name), sourceScales: nil, sourceBiases: nil))
            cursor += tensor.weights.count + tensor.scales.count + tensor.biases.count
        }
        var resident = Data(count: cursor)
        resident.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base, indexSize: UInt64(indexBytes),
                residentSize: UInt64(cursor - indexBytes), entryCount: UInt64(entries.count))
            var nameOffset = stringBase
            for (i, entry) in entries.enumerated() {
                GTurboBinary.writeIndexEntry(into: base.advanced(by: GTurboBinary.indexHeaderBytes + i * GTurboBinary.indexEntryBytes),
                                             entry: entry, nameOffset: UInt32(nameOffset))
                nameOffset += entry.name.utf8.count
            }
        }
        resident.replaceSubrange(stringBase..<indexBytes, with: strings)
        for (tensor, entry) in zip(tensors, entries) {
            for (bytes, offset) in [(tensor.weights, entry.fileOffset), (tensor.scales, entry.scaleOffset),
                                     (tensor.biases, entry.biasOffset)] where !bytes.isEmpty {
                resident.replaceSubrange(Int(offset)..<(Int(offset) + bytes.count), with: bytes)
            }
        }
        var files: [String: [String: Any]] = [:]
        func write(_ data: Data, path: String) throws {
            let url = directory.appendingPathComponent(path)
            try data.write(to: url)
            files[path] = ["size": data.count, "sha256": try Sha256Verifier.hashFile(at: url)]
        }
        try write(resident, path: "model_weights.bin")
        let stride = 16384
        var layers: [[String: Any]] = []
        for layer in 0..<cfg.numLayers {
            let file = String(format: "layer_%02d.bin", layer)
            var experts: [[String: Any]] = []
            var payload = Data(count: stride * cfg.numExperts)
            if layer >= cfg.numDenseLayers {
                for expert in 0..<cfg.numExperts {
                    var offset = 0
                    var parts: [String: [String: Any]] = [:]
                    for role in ["gate", "up", "down"] {
                        let tensor = quantized(role, rows: 64, cols: 64)
                        for (suffix, data) in [("", tensor.weights), ("_scales", tensor.scales), ("_biases", tensor.biases)] {
                            var part: [String: Any] = ["offset": offset, "size": data.count,
                                "dtype": suffix.isEmpty ? "U32" : "BF16", "shape": suffix.isEmpty ? [64, 64] : [64, 1]]
                            if suffix.isEmpty { part["bits"] = 4 }
                            parts[role + suffix] = part
                            let start = expert * stride + offset
                            payload.replaceSubrange(start..<(start + data.count), with: data)
                            offset += data.count
                        }
                    }
                    precondition(offset <= stride)
                    experts.append(["expert": expert, "offset": expert * stride, "size": stride, "tensors": parts])
                }
                try write(payload, path: "packed_experts/\(file)")
            }
            layers.append(["layer": layer, "file": file, "experts": experts])
        }
        let layout: [String: Any] = ["expertStride": stride, "numLayers": cfg.numLayers,
                                   "expertsPerLayer": cfg.numExperts, "layers": layers]
        try write(JSONSerialization.data(withJSONObject: layout, options: [.sortedKeys]),
                  path: "packed_experts/layout.json")
        let arch: [String: Any] = [
            "hiddenSize": 64, "ffnIntermediate": 64, "moeIntermediate": 64,
            "numHeads": 4, "numKVHeads": 2, "numFullKVHeads": 2, "headDim": 32, "fullHeadDim": 32,
            "vocabSize": 256, "slidingWindow": 32, "finalLogitSoftcap": 0,
            "ropeTheta": 0, "fullRopeTheta": 0, "partialRotaryFactor": 0,
            "numLayers": 4, "numExperts": 8, "topKExperts": 6, "tieWordEmbeddings": false,
            "attentionKEqV": false, "fullAttentionLayerMask": [0, 1, 0, 1], "hiddenActivation": "silu",
            "family": "inklingSmall", "attnOutputGate": false, "attentionScale": 1.0 / 32,
            "embeddingScaledBySqrtHidden": false, "routerScaled": false, "ffnSandwichNorms": false,
            "sharedExpertGated": false, "ropeNeoxSubdim": false, "routerScoringFunc": "sigmoid",
            "routedScalingFactor": 8, "relDRel": 16, "relExtent": 64, "relProjDim": 64,
            "relLogScalingFloor": 40, "relLogScalingAlpha": 0.1, "sconvKernelSize": 4,
            "numSharedExperts": 2, "numDenseLayers": 2, "denseIntermediateSize": 128,
            "sharedExpertSink": true, "embedNormEnabled": true, "logitsWidthMultiplier": 16,
            "routerGateBias": true, "routerNormAfterTopK": true, "routerGlobalScale": true,
            "unpaddedVocabSize": 250,
        ]
        let manifest: [String: Any] = ["magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "inkling-nonzero-toy", "arch": arch, "files": files,
            "expertsPerLayer": cfg.numExperts, "numLayers": cfg.numLayers, "expertStride": stride]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
        succeeded = true
        return directory
    }
}
