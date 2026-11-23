import Foundation

/// `layout.json` and manifest additions for an install copied from a GGUF
/// (Inkling 2 Mini). The schema is the existing one plus per-layer keys an
/// older reader ignores; the family itself is unknown to an older runtime, so
/// such an install cannot be mistaken for another format.
extension GTurboJSON {
    /// The GGML type a planned slice or resident entry carries, by its dtype code.
    static func ggmlType(dtypeCode: UInt8) -> GGMLType? {
        GGMLType.allCases.first { $0.residentDtype == dtypeCode }
    }

    /// Per-layer expert stride and per-tensor GGML format. Each layer lists
    /// its own `expertStride` and block `format`; experts sit at
    /// `expert * expertStride` in their layer file. The top-level
    /// `expertStride` is the largest layer stride (the bound a reader that
    /// sizes one slot for every layer needs).
    static func encodeInkling2Layout(plan: RepackPlan) throws -> Data {
        var layersArr: [[String: Any]] = []
        layersArr.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            var tensors: [String: Any] = [:]
            var formats = Set<String>()
            for slice in lp.subTensors {
                guard let type = ggmlType(dtypeCode: slice.dtype) else {
                    throw RepackError.configurationInvalid(detail: "expert slice \(slice.role) has no GGML type")
                }
                formats.insert(type.name)
                tensors[slice.role] = [
                    "offset": slice.offsetInExpertBlob,
                    "size": slice.sizeInExpertBlob,
                    "dtype": "GGML",
                    "format": type.name,
                    "shape": slice.logicalShape.map { Int($0) },
                ] as [String: Any]
            }
            guard formats.count == 1, let format = formats.first else {
                throw RepackError.configurationInvalid(detail: "layer \(lp.layerIndex) mixes expert formats")
            }
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                experts.append(["expert": e, "offset": UInt64(e) * lp.expertStride,
                                "size": lp.expertStride, "tensors": tensors])
            }
            layersArr.append(["layer": lp.layerIndex, "file": (lp.path as NSString).lastPathComponent,
                              "expertStride": lp.expertStride, "format": format, "experts": experts])
        }
        let obj: [String: Any] = [
            "expertStride": plan.layers.map(\.expertStride).max() ?? 0,
            "numLayers": plan.arch.numLayers,
            "expertsPerLayer": plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            "layers": layersArr,
        ]
        return try JSONSerialization.data(withJSONObject: obj,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    /// Adds the GGUF source block to an encoded manifest:
    /// `source{kind:"gguf", repo, commit, pinFingerprint, shards[]}` and the
    /// per-layer `expertStrides`. The runtime's decoder ignores both; they
    /// record what the install was copied from.
    static func addGGUFSourceFields(_ data: Data, plan: RepackPlan, repoID: String,
                                    pin: GGUFSourcePin) throws -> Data {
        guard var m = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "manifest is not a JSON object")
        }
        m["source"] = [
            "kind": "gguf", "repo": repoID, "commit": pin.commit, "pinFingerprint": pin.fingerprint,
            "shards": pin.shards.map { shard -> [String: Any] in
                var entry: [String: Any] = ["path": shard.path, "sha256": shard.sha256, "size": shard.size]
                if let xet = shard.xetHash { entry["xetHash"] = xet }
                return entry
            },
        ] as [String: Any]
        m["expertStrides"] = plan.layers.map(\.expertStride)
        return try JSONSerialization.data(withJSONObject: m,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
