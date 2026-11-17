import Foundation

/// Diagnostic only, never a control: replays an external top-k expert set per
/// (position, layer) inside `FlashNextForwardRunner`'s decode path, so a
/// fidelity comparison can separate routing near-tie flips from everything
/// else. The gate weights still come from the runner's own router logits
/// (softmax over the forced set, as the shipped select kernel computes over
/// its own set).
///
/// Off unless `MFERENCE_U5_FORCE_ROUTES` names a file of little-endian `Int32`
/// expert ids laid out `[positions][layers][topK]`. Positions past the file's
/// end decode unforced. Prefill is never forced.
struct FlashNextForcedRoutes: Sendable {
    static let environmentKey = "MFERENCE_U5_FORCE_ROUTES"

    let positions: Int
    let layers: Int
    let topK: Int
    private let ids: [Int32]

    enum LoadError: Error, Equatable {
        case sizeMismatch(bytes: Int, rowBytes: Int)
        case expertOutOfRange(Int32)
        case duplicateExpert(position: Int, layer: Int)
    }

    init(data: Data, layers: Int, topK: Int, numExperts: Int) throws {
        let rowBytes = layers * topK * MemoryLayout<Int32>.size
        guard rowBytes > 0, !data.isEmpty, data.count % rowBytes == 0 else {
            throw LoadError.sizeMismatch(bytes: data.count, rowBytes: rowBytes)
        }
        let values = data.withUnsafeBytes { raw in
            (0..<(data.count / 4)).map { Int32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) }
        }
        if let bad = values.first(where: { $0 < 0 || Int($0) >= numExperts }) {
            throw LoadError.expertOutOfRange(bad)
        }
        self.positions = data.count / rowBytes
        self.layers = layers
        self.topK = topK
        self.ids = values
        for p in 0..<positions {
            for L in 0..<layers where Set(self.ids(position: p, layer: L) ?? []).count != topK {
                throw LoadError.duplicateExpert(position: p, layer: L)
            }
        }
    }

    /// Nil when the variable is unset; throws when it names an unusable file.
    static func fromEnvironment(layers: Int, topK: Int, numExperts: Int,
                                environment: [String: String] = ProcessInfo.processInfo.environment) throws
        -> FlashNextForcedRoutes? {
        guard let path = environment[environmentKey], !path.isEmpty else { return nil }
        return try FlashNextForcedRoutes(data: Data(contentsOf: URL(fileURLWithPath: path)),
                                         layers: layers, topK: topK, numExperts: numExperts)
    }

    func covers(position p: Int) -> Bool { p >= 0 && p < positions }

    func ids(position p: Int, layer L: Int) -> [Int32]? {
        guard covers(position: p), L >= 0, L < layers else { return nil }
        let base = (p * layers + L) * topK
        return Array(ids[base ..< base + topK])
    }

    /// `softmax(logits[ids])` in Double, the select kernel's renormalized
    /// weights over the forced set (its per-expert scale is 1 for this family).
    static func weights(logits: UnsafePointer<Float>, ids: [Int32]) -> [Float] {
        let selected = ids.map { Double(logits[Int($0)]) }
        let m = selected.max() ?? 0
        let e = selected.map { Foundation.exp($0 - m) }
        let s = e.reduce(0, +)
        return e.map { Float($0 / s) }
    }
}
