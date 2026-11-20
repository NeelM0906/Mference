import Foundation
import Metal
import Testing
@testable import Mference

/// Prefix state snapshots behind the server's multi-prefix cache: KV row
/// images and recurrent checkpoints round-trip byte for byte, and layouts
/// without a complete row image refuse.
@Suite struct QwenPrefixSnapshotTests {

    // MARK: KV row images

    private func filledKV(_ device: MTLDevice, positions: Int, seed: UInt8,
                          growthStep: Int? = 16) throws -> KVCacheManager {
        let kv = try KVCacheManager(device: device, config: .qwen36Toy(), maxContext: 64,
                                    fullAttentionGrowthStep: growthStep)
        try write(kv, from: 0, count: positions, seed: seed)
        return kv
    }

    private func write(_ kv: KVCacheManager, from start: Int, count: Int, seed: UInt8) throws {
        try kv.ensureCapacity(for: start + count)
        for position in start..<(start + count) {
            for layer in 0..<kv.config.numLayers where kv.layerKind(layer) == .full {
                for slot in [kv.kSlot(layer: layer, position: position),
                             kv.vSlot(layer: layer, position: position)] {
                    let bytes = slot.buffer.contents().advanced(by: slot.offset)
                        .assumingMemoryBound(to: UInt8.self)
                    for i in 0..<kv.stride(layer: layer) {
                        bytes[i] = UInt8(truncatingIfNeeded: Int(seed) &* 31 &+ position &* 7 &+ layer &* 3 &+ i)
                    }
                }
            }
            kv.advance()
        }
    }

    private func rows(_ kv: KVCacheManager, count: Int) -> [UInt8] {
        var out: [UInt8] = []
        for layer in 0..<kv.config.numLayers where kv.layerKind(layer) == .full {
            for view in [kv.keyView(layer: layer, validTokenCount: count),
                         kv.valueView(layer: layer, validTokenCount: count)] {
                out += UnsafeBufferPointer(start: view.buffer.contents().assumingMemoryBound(to: UInt8.self),
                                           count: count * view.stride)
            }
        }
        return out
    }

    @Test(arguments: [1, 15, 16, 17, 40])
    func kvRowImageRoundTripsAcrossGrowthAndReset(count: Int) throws {
        let device = try MetalContext().device
        let kv = try filledKV(device, positions: count, seed: 1)
        #expect(kv.supportsPrefixSnapshots)
        let expected = rows(kv, count: count)
        let region = try PrefixSnapshotRegion(byteCount: kv.prefixSnapshotBytes(tokens: count))
        #expect(region.byteCount == expected.count)
        try kv.copyPrefixRows(count: count, into: region)

        // Clobber the live rows, then drop them entirely: reset releases the
        // grown regions, so the restore has to grow again.
        kv.rewind(to: 0)
        try write(kv, from: 0, count: count, seed: 2)
        #expect(rows(kv, count: count) != expected)
        kv.reset()

        try kv.restorePrefixRows(count: count, from: region, position: count)
        #expect(kv.position == count)
        #expect(rows(kv, count: count) == expected)
    }

    @Test func kvRestoreMovesTheCursorOnlyToTheRequestedPosition() throws {
        let device = try MetalContext().device
        let kv = try filledKV(device, positions: 20, seed: 3)
        let expected = rows(kv, count: 20)
        let region = try PrefixSnapshotRegion(byteCount: kv.prefixSnapshotBytes(tokens: 20))
        try kv.copyPrefixRows(count: 20, into: region)
        kv.reset()
        try kv.restorePrefixRows(count: 20, from: region, position: 12)
        #expect(kv.position == 12)
        #expect(rows(kv, count: 20) == expected)
    }

    @Test func kvSnapshotRefusesBadRangesAndSmallRegions() throws {
        let device = try MetalContext().device
        let kv = try filledKV(device, positions: 8, seed: 4)
        let small = try PrefixSnapshotRegion(byteCount: kv.prefixSnapshotBytes(tokens: 8) - 1)
        #expect(throws: PrefixSnapshotError.self) { try kv.copyPrefixRows(count: 8, into: small) }
        let region = try PrefixSnapshotRegion(byteCount: kv.prefixSnapshotBytes(tokens: 9))
        #expect(throws: PrefixSnapshotError.self) { try kv.copyPrefixRows(count: 9, into: region) }
        #expect(throws: PrefixSnapshotError.self) { try kv.copyPrefixRows(count: 0, into: region) }
        #expect(throws: PrefixSnapshotError.self) {
            try kv.restorePrefixRows(count: 8, from: region, position: 9)
        }
        #expect(kv.position == 8)
    }

    @Test func layoutsWithoutAFullRowImageAreUnsupported() throws {
        let device = try MetalContext().device
        // Gemma 4 keeps sliding-window rings.
        let gemma = try KVCacheManager(device: device, config: .retiredGemma4Shape, maxContext: 64,
                                       fp16RingEnabled: true)
        #expect(!gemma.supportsPrefixSnapshots)
        // Paged full attention keeps its rows in a KVPageStore.
        let paged = try KVCacheManager(device: device, config: .qwen36Toy(), maxContext: 64,
                                       pagedFullAttention: true)
        #expect(!paged.supportsPrefixSnapshots)
        let region = try PrefixSnapshotRegion(byteCount: 4096)
        #expect(throws: PrefixSnapshotError.self) { try gemma.copyPrefixRows(count: 1, into: region) }
    }

    // MARK: Recurrent checkpoints

    @Test func recurrentCheckpointRoundTrips() throws {
        let device = try MetalContext().device
        let cfg = ArchConfig.qwen36Toy()
        let gdn = try GDNStateManager(device: device, config: cfg)
        func fill(_ seed: UInt8) {
            for layer in 0..<cfg.numLayers where gdn.isLinear(layer: layer) {
                for buffer in [gdn.stateBuffer(layer: layer), gdn.convTailBuffer(layer: layer)] {
                    let bytes = buffer.contents().assumingMemoryBound(to: UInt8.self)
                    for i in 0..<buffer.length { bytes[i] = UInt8(truncatingIfNeeded: Int(seed) &+ i &* 13 &+ layer) }
                }
            }
        }
        func dump() -> [UInt8] {
            (0..<cfg.numLayers).filter { gdn.isLinear(layer: $0) }.flatMap { layer in
                [gdn.stateBuffer(layer: layer), gdn.convTailBuffer(layer: layer)].flatMap { buffer in
                    Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self),
                                              count: buffer.length))
                }
            }
        }
        fill(5)
        let expected = dump()
        #expect(gdn.recurrentCheckpointBytes == expected.count)
        let region = try PrefixSnapshotRegion(byteCount: gdn.recurrentCheckpointBytes)
        try gdn.copyRecurrentState(into: region)
        fill(9)
        #expect(dump() != expected)
        try gdn.restoreRecurrentState(from: region)
        #expect(dump() == expected)
        let small = try PrefixSnapshotRegion(byteCount: gdn.recurrentCheckpointBytes - 1)
        #expect(throws: PrefixSnapshotError.self) { try gdn.restoreRecurrentState(from: small) }
    }
}
