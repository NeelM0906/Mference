import Foundation

enum StreamerError: Error, CustomStringConvertible {
    case openFailed(path: String, errno: Int32)
    case sizeMismatch(expected: UInt64, actual: UInt64)
    case offsetOutOfRange(UInt64)
    case bufferWrapFailed
    case preadFailed(errno: Int32)
    case allocFailed(errno: Int32)
    case slotOutOfRange(Int)
    case noSlotCache

    public var description: String {
        switch self {
        case .openFailed(let path, let error):
            return "open(\(path)) failed: errno \(error)"
        case .sizeMismatch(let expected, let actual):
            return "file size mismatch: expected \(expected), got \(actual)"
        case .offsetOutOfRange(let offset):
            return "offset \(offset) is outside the streamed range"
        case .bufferWrapFailed:
            return "failed to wrap expert cache memory in an MTLBuffer"
        case .preadFailed(let error):
            return "pread failed: errno \(error)"
        case .allocFailed(let error):
            return "posix_memalign failed: errno \(error)"
        case .slotOutOfRange(let slot):
            return "expert cache slot \(slot) is out of range"
        case .noSlotCache:
            return "resident expert mode has no slot cache"
        }
    }
}

/// Byte layout of one routed-expert layer file.
public struct StreamLayout: Sendable {
    public let path: String
    public let streamOffset: UInt64
    public let streamSize: UInt64
    public let expertsPerLayer: Int
    /// Bytes of one expanded expert, as the kernels address it.
    public let expertStride: UInt64
    public let expertOffsets: [UInt64]?
    /// Set when the file stores experts without bytes the kernels read;
    /// see `ExpertStorage`. Nil means each expert is stored as addressed.
    public let storage: ExpertStorage?

    public init(path: String,
                streamOffset: UInt64,
                streamSize: UInt64,
                expertsPerLayer: Int,
                expertStride: UInt64,
                expertOffsets: [UInt64]? = nil,
                storage: ExpertStorage? = nil) {
        self.path = path
        self.streamOffset = streamOffset
        self.streamSize = streamSize
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.expertOffsets = expertOffsets
        self.storage = storage
    }

    /// Bytes of one expert in the file.
    @inline(__always)
    public var storedExpertStride: UInt64 { storage?.storedExpertStride ?? expertStride }

    @inline(__always)
    public func expertOffset(layer: Int, expert: Int) -> UInt64 {
        if layer == 0, let expertOffsets, expert >= 0, expert < expertOffsets.count {
            return expertOffsets[expert]
        }
        let perLayer = UInt64(expertsPerLayer) * storedExpertStride
        return UInt64(layer) * perLayer + UInt64(expert) * storedExpertStride
    }
}
