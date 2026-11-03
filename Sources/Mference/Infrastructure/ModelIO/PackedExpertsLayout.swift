import Foundation

struct SubTensorEntry: Sendable, Equatable {
    let offset: UInt64    // relative to the expanded expert's start (the kernel view)
    let size: UInt64      // bytes; scale slices encode the group count
    let dtype: String?
    let shape: [Int]?
    let bits: Int?

    init(offset: UInt64, size: UInt64, dtype: String? = nil,
         shape: [Int]? = nil, bits: Int? = nil) {
        self.offset = offset
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.bits = bits
    }
}

struct ExpertEntry: Sendable {
    /// Logical routed-expert id used by the model/router.
    let expert: Int
    /// Absolute byte offset of this expert blob's start inside its layer file.
    let offset: UInt64
    /// Bytes this expert occupies in its layer file (`storedExpertStride`).
    let size: UInt64
    /// Sub-tensors keyed by role (gate / up / down / shared) and component
    /// (raw, `_scales`, `_biases`).
    let subTensors: [String: SubTensorEntry]

    init(expert: Int,
                offset: UInt64,
                size: UInt64,
                subTensors: [String: SubTensorEntry]) {
        self.expert = expert
        self.offset = offset
        self.size = size
        self.subTensors = subTensors
    }
}

struct LayerLayout: Sendable {
    let layer: Int
    let file: String          // basename, e.g. "layer_00.bin"
    let experts: [ExpertEntry]
}

struct PackedExpertsLayout: Sendable {
    /// Bytes of one expanded expert; the tensor offsets are relative to it.
    let expertStride: UInt64
    let numLayers: Int
    let expertsPerLayer: Int
    let layers: [LayerLayout]
    /// How experts are stored when the file omits bytes the kernels read.
    let storage: ExpertStorage?

    init(expertStride: UInt64, numLayers: Int, expertsPerLayer: Int,
         layers: [LayerLayout], storage: ExpertStorage? = nil) {
        self.expertStride = expertStride
        self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer
        self.layers = layers
        self.storage = storage
    }

    /// Bytes of one expert in its layer file; expert offsets and sizes in the
    /// layout are file values.
    var storedExpertStride: UInt64 { storage?.storedExpertStride ?? expertStride }

    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    func expert(layer: Int, expert: Int) -> ExpertEntry {
        return layers[layer].experts[expert]
    }
}

enum PackedExpertsLayoutReader {
    // 64 MiB: Qwen 3.6's 40 layers x 256 experts x 9 sub-tensors produce a
    // ~22 MB layout.json; Gemma's is ~5 MB.
    static let defaultMaxBytes: UInt64 = 64 * 1024 * 1024

    static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let size = try metadataFileSize(url)
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "layout.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: url)
        let root: [String: Any]
        do {
            root = try (JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        } catch {
            throw ModelError.indexCorrupt(detail: "layout.json: \(error)")
        }
        guard
            let expertStride = (root["expertStride"] as? NSNumber)?.uint64Value,
            let numLayers = root["numLayers"] as? Int,
            let expertsPerLayer = root["expertsPerLayer"] as? Int,
            let layersArr = root["layers"] as? [[String: Any]]
        else {
            throw ModelError.indexCorrupt(detail: "layout.json: missing top-level keys")
        }

        var layers: [LayerLayout] = []
        layers.reserveCapacity(layersArr.count)
        for layerObj in layersArr {
            guard
                let layerIdx = layerObj["layer"] as? Int,
                let file = layerObj["file"] as? String,
                let expertsArr = layerObj["experts"] as? [[String: Any]]
            else {
                throw ModelError.indexCorrupt(detail: "layout.json: malformed layer entry")
            }
            // Leading dense-FFN layers carry no routed experts; the writer
            // emits an empty entry and no blob file (Inkling's layers 0-1).
            if expertsArr.isEmpty {
                layers.append(LayerLayout(layer: layerIdx, file: file, experts: []))
                continue
            }
            var experts = [ExpertEntry?](repeating: nil, count: expertsPerLayer)
            for expertObj in expertsArr {
                guard
                    let offset = (expertObj["offset"] as? NSNumber)?.uint64Value,
                    let size = (expertObj["size"] as? NSNumber)?.uint64Value,
                    let tensorsObj = expertObj["tensors"] as? [String: [String: Any]]
                else {
                    throw ModelError.indexCorrupt(detail: "layout.json: malformed expert entry")
                }
                var subTensors: [String: SubTensorEntry] = [:]
                for (role, t) in tensorsObj {
                    guard
                        let toff = (t["offset"] as? NSNumber)?.uint64Value,
                        let tsize = (t["size"] as? NSNumber)?.uint64Value,
                        t["dtype"] is String,
                        t["shape"] is [Int]
                    else {
                        throw ModelError.indexCorrupt(detail: "layout.json: malformed tensor \(role)")
                    }
                    if let bits = t["bits"], !(bits is Int) {
                        throw ModelError.indexCorrupt(detail: "layout.json: malformed tensor bits \(role)")
                    }
                    subTensors[role] = SubTensorEntry(offset: toff, size: tsize,
                        dtype: t["dtype"] as? String, shape: t["shape"] as? [Int],
                        bits: t["bits"] as? Int)
                }
                let expertID = expertObj["expert"] as? Int ?? experts.compactMap { $0 }.count
                guard expertID >= 0 && expertID < expertsPerLayer else {
                    throw ModelError.indexCorrupt(detail: "layout.json: expert id out of range")
                }
                let physicalRank = expertObj["physicalRank"] as? Int
                if let physicalRank,
                   (physicalRank < 0 || physicalRank >= expertsPerLayer) {
                    throw ModelError.indexCorrupt(detail: "layout.json: physicalRank out of range")
                }
                experts[expertID] = ExpertEntry(expert: expertID,
                                                offset: offset,
                                                size: size,
                                                subTensors: subTensors)
            }
            guard experts.allSatisfy({ $0 != nil }) else {
                throw ModelError.indexCorrupt(detail: "layout.json: missing expert entries")
            }
            layers.append(LayerLayout(layer: layerIdx,
                                      file: file,
                                      experts: experts.map { $0! }))
        }

        var storage: ExpertStorage?
        if let storageObj = root["expertStorage"] {
            guard let storageObj = storageObj as? [String: Any] else {
                throw ModelError.indexCorrupt(detail: "layout.json: malformed expertStorage")
            }
            storage = try parseStorage(storageObj, expertStride: expertStride, layers: layers)
        }
        return PackedExpertsLayout(expertStride: expertStride,
                                   numLayers: numLayers,
                                   expertsPerLayer: expertsPerLayer,
                                   layers: layers,
                                   storage: storage)
    }

    /// Parses `expertStorage` and proves it reconstructs every byte the
    /// kernels read: stored segments cover each tensor that is not an implied
    /// bias, and each implied bias sits outside the segments beside scales
    /// that a segment loads.
    static func parseStorage(_ obj: [String: Any],
                             expertStride: UInt64,
                             layers: [LayerLayout]) throws -> ExpertStorage {
        func invalid(_ detail: String) -> ModelError {
            .indexCorrupt(detail: "layout.json: expertStorage " + detail)
        }
        guard obj["format"] as? String == ExpertStorage.impliedNeg8ScaleBiasesFormat else {
            throw invalid("has an unknown format")
        }
        guard let stored = (obj["storedExpertStride"] as? NSNumber)?.uint64Value, stored > 0,
              let segmentObjs = obj["segments"] as? [[String: Any]], !segmentObjs.isEmpty,
              let impliedObjs = obj["impliedBiases"] as? [[String: String]], !impliedObjs.isEmpty else {
            throw invalid("is missing its stride, segments or implied biases")
        }
        let routed = layers.flatMap(\.experts)
        guard let tensors = routed.first?.subTensors,
              routed.allSatisfy({ $0.subTensors == tensors }) else {
            throw invalid("needs experts with one uniform tensor layout")
        }
        var segments: [ExpertStorage.Segment] = []
        for segmentObj in segmentObjs {
            guard let storedOffset = (segmentObj["storedOffset"] as? NSNumber)?.uint64Value,
                  let memoryOffset = (segmentObj["memoryOffset"] as? NSNumber)?.uint64Value,
                  let size = (segmentObj["size"] as? NSNumber)?.uint64Value, size > 0,
                  storedOffset <= stored, size <= stored - storedOffset,
                  memoryOffset <= expertStride, size <= expertStride - memoryOffset else {
                throw invalid("has a malformed or out-of-range segment")
            }
            segments.append(.init(storedOffset: storedOffset, memoryOffset: memoryOffset, size: size))
        }
        func overlaps(_ ranges: [(UInt64, UInt64)]) -> Bool {
            let sorted = ranges.sorted { $0.0 < $1.0 }
            return zip(sorted, sorted.dropFirst()).contains { $0.0 + $0.1 > $1.0 }
        }
        guard !overlaps(segments.map { ($0.storedOffset, $0.size) }),
              !overlaps(segments.map { ($0.memoryOffset, $0.size) }) else {
            throw invalid("has overlapping segments")
        }
        func covered(_ offset: UInt64, _ size: UInt64) -> Bool {
            segments.contains { offset >= $0.memoryOffset && offset + size <= $0.memoryOffset + $0.size }
        }
        func touched(_ offset: UInt64, _ size: UInt64) -> Bool {
            segments.contains { offset < $0.memoryOffset + $0.size && $0.memoryOffset < offset + size }
        }
        var implied: [ExpertStorage.ImpliedBias] = []
        var impliedNames = Set<String>()
        for impliedObj in impliedObjs {
            guard let biasesName = impliedObj["biases"], let scalesName = impliedObj["scales"],
                  let biases = tensors[biasesName], let scales = tensors[scalesName],
                  biases.dtype == "BF16", scales.dtype == "BF16",
                  biases.size == scales.size, biases.size > 0, biases.size.isMultiple(of: 2),
                  biases.offset.isMultiple(of: 2), scales.offset.isMultiple(of: 2),
                  biases.offset + biases.size <= expertStride,
                  covered(scales.offset, scales.size), !touched(biases.offset, biases.size),
                  impliedNames.insert(biasesName).inserted else {
                throw invalid("has an invalid implied bias")
            }
            implied.append(.init(biasesOffset: biases.offset, scalesOffset: scales.offset,
                                 size: biases.size))
        }
        for (name, tensor) in tensors where !impliedNames.contains(name) {
            guard covered(tensor.offset, tensor.size) else {
                throw invalid("does not store \(name)")
            }
        }
        return ExpertStorage(storedExpertStride: stored, segments: segments, impliedBiases: implied)
    }

    private static func metadataFileSize(_ url: URL) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attrs[.size] as? NSNumber else {
            throw ModelError.indexCorrupt(detail: "layout.json: file size unavailable")
        }
        return number.uint64Value
    }
}
