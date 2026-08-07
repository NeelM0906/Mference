import Darwin
import Foundation
import Metal

/// All-resident routed-expert backend. Maps the entire layer file once and
/// serves page-aligned subregion views of one shared MTLBuffer. No slots,
/// no bookkeeping, no reads on the hot path.
public final class ResidentExpertStreamer: @unchecked Sendable {
    public let layout: StreamLayout
    private let resident: ResidentBuffer

    public init(layout: StreamLayout, device: MTLDevice) throws {
        self.layout = layout
        self.resident = try ResidentBuffer(
            fileURL: URL(fileURLWithPath: layout.path),
            fileOffset: layout.streamOffset,
            residentSize: layout.streamSize,
            device: device)
    }

    public func expertBuffer(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard expert >= 0, expert < layout.expertsPerLayer else {
            throw StreamerError.slotOutOfRange(expert)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        return (resident.buffer, regionOffset, layout.expertStride)
    }

    /// Touch the mapping sequentially so first-token decode does not pay
    /// the page-in cost. Called at load time; counts as model load, not
    /// decode.
    public func warmUp() {
        let contents = resident.buffer.contents()
        let pageSize = Int(getpagesize())
        var checksum: UInt8 = 0
        var offset = 0
        while offset < Int(layout.streamSize) {
            checksum ^= contents.load(fromByteOffset: offset, as: UInt8.self)
            offset += pageSize
        }
        _ = checksum
    }
}
