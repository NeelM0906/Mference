import Foundation

/// Pre-load sizing for the expert-cache auto profile: how many bytes the
/// routed-expert pool and core weights occupy on disk, read from
/// `packed_experts/layout.json` and `model_weights.bin` without loading
/// the model.
public enum ExpertPoolInspector {

    /// Total routed-expert bytes across all layers. Dense-FFN layers with no
    /// routed experts contribute nothing.
    public static func poolByteSize(directoryURL: URL) throws -> UInt64 {
        let layout = try PackedExpertsLayoutReader.load(directoryURL: directoryURL)
        return layout.layers.reduce(UInt64(0)) { total, layer in
            total + UInt64(layer.experts.count) * layout.expertStride
        }
    }

    /// On-disk size of `model_weights.bin` (resident core weights + index).
    public static func coreWeightsByteSize(directoryURL: URL) throws -> UInt64 {
        let url = directoryURL.appendingPathComponent("model_weights.bin")
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = attrs[.size] as? UInt64 else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        return size
    }
}
