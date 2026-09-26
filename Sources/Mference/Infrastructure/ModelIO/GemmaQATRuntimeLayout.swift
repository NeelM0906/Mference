import Foundation

extension GemmaQATCheckpoint {
    /// Validate the byte geometry consumed by the kernels before mapping or
    /// allocating GPU buffers. A file checksum alone does not establish that
    /// its tensor index agrees with the checkpoint's native format.
    static func validateRuntimeLayout(residentIndex index: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      expected arch: ArchConfig) throws {
        func invalid(_ detail: String) -> ModelError {
            .indexCorrupt(detail: "Gemma QAT runtime layout: " + detail)
        }
        var names = Set<String>()
        var spans: [(start: UInt64, end: UInt64)] = []
        let region = index.header
        guard region.indexSize <= UInt64.max - region.residentSize else {
            throw invalid("resident region overflows")
        }
        let end = region.indexSize + region.residentSize
        func span(_ offset: UInt64, _ size: UInt64, name: String) throws {
            guard offset.isMultiple(of: 2), offset >= region.indexSize,
                  offset <= end, size > 0, size <= end - offset else {
                throw invalid("invalid resident span for \(name)")
            }
            spans.append((offset, offset + size))
        }
        func tensor(_ name: String, _ shape: [Int], packed: Bool = false) throws {
            names.insert(name)
            guard let entry = index.entries[name] else { throw invalid("missing \(name)") }
            let actual = [entry.shape.0, entry.shape.1, entry.shape.2, entry.shape.3]
            let padded = shape.map(UInt32.init) + Array(repeating: UInt32(0), count: 4 - shape.count)
            let elements = shape.reduce(UInt64(1)) { $0 * UInt64($1) }
            guard actual == padded, entry.dtype == (packed ? 0 : 1),
                  entry.sizeBytes == (packed ? elements / 2 : elements * 2) else {
                throw invalid("wrong shape/dtype/size for \(name)")
            }
            try span(entry.fileOffset, entry.sizeBytes, name: name)
            if packed {
                guard shape.last!.isMultiple(of: 32),
                      entry.scaleSize == elements / 32 * 2,
                      entry.biasSize == entry.scaleSize else {
                    throw invalid("wrong group-32 companion size for \(name)")
                }
                try span(entry.scaleOffset, entry.scaleSize, name: name + " scales")
                try span(entry.biasOffset, entry.biasSize, name: name + " biases")
            } else {
                guard entry.scaleOffset == 0, entry.scaleSize == 0,
                      entry.biasOffset == 0, entry.biasSize == 0 else {
                    throw invalid("BF16 tensor has quantization companions: \(name)")
                }
            }
        }
        let root = "language_model.model"
        let d = arch.hiddenSize
        try tensor(root + ".embed_tokens.weight", [arch.vocabSize, d], packed: true)
        try tensor(root + ".norm.weight", [d])
        for layer in 0..<arch.numLayers {
            let p = root + ".layers.\(layer)"
            let full = arch.layerIsFull(layer)
            let head = full ? arch.fullHeadDim : arch.headDim
            let kv = full ? arch.numFullKVHeads : arch.numKVHeads
            try tensor(p + ".self_attn.q_proj.weight", [arch.numHeads * head, d], packed: true)
            try tensor(p + ".self_attn.k_proj.weight", [kv * head, d], packed: true)
            if !full || !arch.attentionKEqV {
                try tensor(p + ".self_attn.v_proj.weight", [kv * head, d], packed: true)
            }
            try tensor(p + ".self_attn.o_proj.weight", [d, arch.numHeads * head], packed: true)
            try tensor(p + ".self_attn.q_norm.weight", [head])
            try tensor(p + ".self_attn.k_norm.weight", [head])
            try tensor(p + ".router.proj.weight", [arch.numExperts, d])
            try tensor(p + ".router.scale", [d])
            try tensor(p + ".router.per_expert_scale", [arch.numExperts])
            try tensor(p + ".layer_scalar", [1])
            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
                         "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                         "post_feedforward_layernorm_2"] {
                try tensor(p + "." + norm + ".weight", [d])
            }
            for role in ["gate", "up", "down"] {
                let shape = role == "down" ? [d, arch.intermediateSize] : [arch.intermediateSize, d]
                try tensor(p + ".mlp.\(role)_proj.weight", shape, packed: true)
            }
        }
        guard Set(index.entries.keys) == names,
              region.entryCount == UInt64(names.count) else {
            throw invalid("resident inventory differs from text architecture")
        }
        let sorted = spans.sorted { $0.start < $1.start }
        for (left, right) in zip(sorted, sorted.dropFirst()) where left.end > right.start {
            throw invalid("overlapping resident tensors or companions")
        }

        guard layout.numLayers == arch.numLayers, layout.layers.count == arch.numLayers,
              layout.expertsPerLayer == arch.numExperts,
              layout.expertStride == manifest.expertStride,
              layout.expertStride > 0, layout.expertStride <= UInt32.max,
              layout.expertStride.isMultiple(of: UInt64(getpagesize())),
              layout.storedExpertStride > 0,
              layout.storedExpertStride.isMultiple(of: UInt64(getpagesize())) else {
            throw invalid("expert dimensions or stride disagree with manifest")
        }
        // The manifest and the layout must both say whether biases are implied;
        // neither may be inferred from the other or from missing keys.
        let impliedBiases = manifest.quant?.routedExpert.biasType.lowercased()
            == GemmaQATCheckpoint.impliedRoutedBiasType.lowercased()
        guard impliedBiases == (layout.storage != nil) else {
            throw invalid("manifest bias type and layout expert storage disagree")
        }
        let roles = Set(["gate", "gate_scales", "gate_biases", "up", "up_scales",
                         "up_biases", "down", "down_scales", "down_biases"])
        for (layerID, layer) in layout.layers.enumerated() {
            let file = "packed_experts/" + layer.file
            guard layer.layer == layerID, layer.experts.count == arch.numExperts,
                  URL(fileURLWithPath: layer.file).lastPathComponent == layer.file,
                  manifest.files[file]?.size == layout.storedExpertStride * UInt64(arch.numExperts) else {
                throw invalid("invalid expert layer \(layerID)")
            }
            var physicalOffsets = Set<UInt64>()
            for (expertID, expert) in layer.experts.enumerated() {
                guard expert.expert == expertID, expert.size == layout.storedExpertStride,
                      expert.offset.isMultiple(of: layout.storedExpertStride),
                      expert.offset / layout.storedExpertStride < UInt64(arch.numExperts),
                      physicalOffsets.insert(expert.offset).inserted,
                      Set(expert.subTensors.keys) == roles,
                      expert.subTensors == layer.experts[0].subTensors else {
                    throw invalid("invalid or nonuniform expert \(layerID)/\(expertID)")
                }
                var expertSpans: [(start: UInt64, end: UInt64)] = []
                for role in ["gate", "up", "down"] {
                    let shape = role == "down" ? [d, arch.moeIntermediateSize] : [arch.moeIntermediateSize, d]
                    let elements = UInt64(shape[0]) * UInt64(shape[1])
                    for suffix in ["", "_scales", "_biases"] {
                        let name = role + suffix
                        let entry = expert.subTensors[name]!
                        let packed = suffix.isEmpty
                        let expectedShape = packed ? shape : [shape[0], shape[1] / 32]
                        // The down kernel reads uint words; gate/up use ushort.
                        let alignment: UInt64 = packed && role == "down" ? 4 : 2
                        guard entry.dtype == (packed ? "U32" : "BF16"),
                              entry.shape == expectedShape,
                              packed ? entry.bits == 4 : entry.bits == nil,
                              entry.size == (packed ? elements / 2 : elements / 32 * 2),
                              entry.offset.isMultiple(of: alignment),
                              entry.offset <= layout.expertStride,
                              entry.size <= layout.expertStride - entry.offset else {
                            throw invalid("invalid \(name) in expert \(layerID)/\(expertID)")
                        }
                        expertSpans.append((entry.offset, entry.offset + entry.size))
                    }
                }
                let ordered = expertSpans.sorted { $0.start < $1.start }
                for (left, right) in zip(ordered, ordered.dropFirst()) where left.end > right.start {
                    throw invalid("overlapping expert tensors in \(layerID)/\(expertID)")
                }
            }
        }
    }
}
