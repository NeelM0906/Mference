import Testing
import Foundation
import Darwin
import Metal
@testable import Mference

/// Tests `KVPageStore` page geometry, unsealed-slot addressing, seal +
/// write-behind spill, LRU eviction, fetch round-trips, pinning, and the
/// layer-major spill-file layout against the Qwen 3.8 config.
@Suite struct KVPageStoreTests {

    private let config = ArchConfig.qwen38_27B

    /// Qwen 3.8 full-attn stride: 4 kv-heads * 256 head_dim * FP16 = 2048 B.
    private static let tokenStride = 4 * 256 * 2
    private static let pageTokens = KVPageGeometry.tokensPerPage

    private func makeStore(maxContext: Int = 512,
                           poolPagesPerLayer: Int = 8) throws -> (MetalContext, KVPageStore, URL) {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("kvpage-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let store = try KVPageStore(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    poolPagesPerLayer: poolPagesPerLayer,
                                    spillDirectory: dir)
        return (ctx, store, dir)
    }

    // MARK: geometry

    @Test func geometry_matchesQwen38Config() throws {
        let (_, store, _) = try makeStore(maxContext: 512, poolPagesPerLayer: 8)
        #expect(store.geometry.fullLayerOrdinals.count == 16)
        // mask 1 at layers 3, 7, 11, ...
        #expect(store.geometry.fullLayerOrdinals.first == 3)
        #expect(store.geometry.fullLayerOrdinals.last == 63)
        #expect(store.geometry.tokenStrideBytes == Self.tokenStride)
        #expect(store.geometry.pagesPerLayer == 512 / Self.pageTokens)
        #expect(store.geometry.kPageBytes == Self.pageTokens * Self.tokenStride)
    }

    @Test func geometry_rejectsNonFullAttentionLayer() throws {
        let (_, store, _) = try makeStore()
        // Layer 0 is a linear-attention layer in Qwen 3.8.
        #expect(store.fullLayerOrdinal(forLayer: 0) == nil)
        #expect(store.fullLayerOrdinal(forLayer: 3) == 0)
        #expect(store.fullLayerOrdinal(forLayer: 7) == 1)
    }

    // MARK: unsealed writes

    @Test func kSlotAndVSlot_addressWithinUnsealedPage() throws {
        let (_, store, _) = try makeStore()
        // Position 0 lands in page 0 at offset 0.
        let k0 = try store.kSlot(layer: 3, position: 0)
        #expect(k0.offset % Self.tokenStride == 0)
        // Position 65 lands in page 1 at within-page row 1.
        for p in 0...65 { _ = try store.kSlot(layer: 3, position: p) }
        let k65 = try store.kSlot(layer: 3, position: 65)
        let v65 = try store.vSlot(layer: 3, position: 65)
        #expect(k65.offset % Self.tokenStride == 0)
        // K and V use the same slot index in distinct pool buffers.
        #expect(k65.offset == v65.offset)
        #expect(k65.buffer !== v65.buffer)
    }

    @Test func advance_sealsCrossedPagesAcrossAllLayers() throws {
        let (_, store, _) = try makeStore()
        for p in 0..<Self.pageTokens { _ = try store.kSlot(layer: 3, position: p) }
        #expect(store.sealedPageCount == 0)
        store.advance(by: Self.pageTokens)      // position now 64: page 0 sealed
        store.flushSpills()
        #expect(store.sealedPageCount == 1)
        #expect(store.position == Self.pageTokens)
    }

    // MARK: spill + fetch round trip

    /// Fill page 0 of one layer with a marker pattern, seal, flush, then read
    /// the spill file directly at the computed layer-major offset.
    @Test func seal_writesPageToSpillFileAtLayerMajorOffset() throws {
        let (_, store, dir) = try makeStore()
        let ordinal = try #require(store.fullLayerOrdinal(forLayer: 7))

        let k = try store.kSlot(layer: 7, position: 0)
        let pageBytes = store.geometry.kPageBytes
        memset(k.buffer.contents() + k.offset, 0xAB, pageBytes)
        let v = try store.vSlot(layer: 7, position: 0)
        memset(v.buffer.contents() + v.offset, 0xCD, pageBytes)

        store.advance(by: Self.pageTokens)
        store.flushSpills()

        let fileURL = dir.appendingPathComponent(store.spillFileName)
        let data = try Data(contentsOf: fileURL)
        let off = store.geometry.fileOffset(layerOrdinal: ordinal, pageIndex: 0)
        #expect(data[off] == 0xAB)
        #expect(data[off + pageBytes - 1] == 0xAB)
        #expect(data[off + pageBytes] == 0xCD)
        #expect(data[off + 2 * pageBytes - 1] == 0xCD)
    }

    @Test func fileOffsets_layerRegionsDoNotOverlap() throws {
        let (_, store, _) = try makeStore(maxContext: 512)
        let g = store.geometry
        let regionBytes = g.pagesPerLayer * 2 * g.kPageBytes
        for ord in 0..<g.fullLayerOrdinals.count {
            let start = g.fileOffset(layerOrdinal: ord, pageIndex: 0)
            #expect(start == ord * regionBytes)
            let lastEnd = g.fileOffset(layerOrdinal: ord, pageIndex: g.pagesPerLayer - 1)
                + 2 * g.kPageBytes
            #expect(lastEnd <= (ord + 1) * regionBytes)
        }
    }

    @Test func evictedPage_fetchRoundTripsIdenticalBytes() throws {
        // Pool of 3 pages per layer; touch 4 pages to force one eviction.
        let (_, store, _) = try makeStore(maxContext: 512, poolPagesPerLayer: 3)
        let pageBytes = store.geometry.kPageBytes

        // Write distinct patterns into pages 0..2 of layer 3, sealing each.
        for page in 0..<3 {
            let k = try store.kSlot(layer: 3, position: page * Self.pageTokens)
            memset(k.buffer.contents() + k.offset, Int32(0xA0 + page), pageBytes)
            let v = try store.vSlot(layer: 3, position: page * Self.pageTokens)
            memset(v.buffer.contents() + v.offset, Int32(0xB0 + page), pageBytes)
            store.advance(by: Self.pageTokens)
        }
        store.flushSpills()
        #expect(store.residentPageCount(layer: 3) == 3)

        // Touching page 3 (the new unsealed page) forces eviction of the LRU
        // sealed page (page 0 — pages 1, 2 were sealed later).
        _ = try store.kSlot(layer: 3, position: 3 * Self.pageTokens)
        #expect(store.isResident(layer: 3, pageIndex: 0) == false)
        #expect(store.isResident(layer: 3, pageIndex: 1))

        // Fetch page 0 back: bytes must round-trip. Page 1 becomes the victim.
        let slot = try store.ensureResident(layer: 3, pageIndex: 0)
        let kPool = store.kPoolBuffer(layer: 3)
        let vPool = store.vPoolBuffer(layer: 3)
        let kBase = kPool.contents() + slot * pageBytes
        let vBase = vPool.contents() + slot * pageBytes
        #expect(kBase.load(as: UInt8.self) == 0xA0)
        #expect((kBase + pageBytes - 1).load(as: UInt8.self) == 0xA0)
        #expect(vBase.load(as: UInt8.self) == 0xB0)
    }

    @Test func pinnedPage_survivesEvictionPressure() throws {
        let (_, store, _) = try makeStore(maxContext: 512, poolPagesPerLayer: 3)
        for page in 0..<3 {
            _ = try store.kSlot(layer: 3, position: page * Self.pageTokens)
            store.advance(by: Self.pageTokens)
        }
        store.flushSpills()
        store.pin(layer: 3, pageIndex: 0)

        _ = try store.kSlot(layer: 3, position: 3 * Self.pageTokens)
        // Page 0 is pinned; page 1 must be the victim instead.
        #expect(store.isResident(layer: 3, pageIndex: 0))
        #expect(store.isResident(layer: 3, pageIndex: 1) == false)

        store.unpin(layer: 3, pageIndex: 0)
    }

    @Test func evictingDirtyPage_flushesItFirst() throws {
        let (_, store, _) = try makeStore(maxContext: 512, poolPagesPerLayer: 3)
        let pageBytes = store.geometry.kPageBytes
        for page in 0..<3 {
            let k = try store.kSlot(layer: 3, position: page * Self.pageTokens)
            memset(k.buffer.contents() + k.offset, Int32(0xE0 + page), pageBytes)
            store.advance(by: Self.pageTokens)
        }
        // No flushSpills(): page 0 may still be dirty when eviction hits it.
        _ = try store.kSlot(layer: 3, position: 3 * Self.pageTokens)
        let slot = try store.ensureResident(layer: 3, pageIndex: 0)
        let kBase = store.kPoolBuffer(layer: 3).contents() + slot * pageBytes
        #expect(kBase.load(as: UInt8.self) == 0xE0)
    }

    // MARK: page tables

    @Test func pageTable_mapsSelectionToPoolSlots() throws {
        let (_, store, _) = try makeStore(maxContext: 512, poolPagesPerLayer: 8)
        for page in 0..<3 {
            _ = try store.kSlot(layer: 3, position: page * Self.pageTokens)
            store.advance(by: Self.pageTokens)
        }
        _ = try store.kSlot(layer: 3, position: 3 * Self.pageTokens)   // unsealed tail

        let table = try store.pageTable(layer: 3, selectedPages: [0, 2, 3])
        #expect(table.count == 3)
        let pageBytes = store.geometry.kPageBytes
        let kPool = store.kPoolBuffer(layer: 3)
        for (i, page) in [0, 2, 3].enumerated() {
            let slot = try store.ensureResident(layer: 3, pageIndex: page)
            #expect(Int(table[i]) == slot)
            #expect((Int(table[i]) + 1) * pageBytes <= kPool.length)
        }
    }

    // MARK: reset

    @Test func reset_dropsResidencyAndPosition() throws {
        let (_, store, _) = try makeStore()
        for page in 0..<2 {
            _ = try store.kSlot(layer: 3, position: page * Self.pageTokens)
            store.advance(by: Self.pageTokens)
        }
        store.flushSpills()
        store.reset()
        #expect(store.position == 0)
        #expect(store.sealedPageCount == 0)
        #expect(store.residentPageCount(layer: 3) == 0)
    }

    // MARK: spill write failures

    @Test func writeFully_surfacesDescriptorErrors() {
        let bytes: [UInt8] = [1, 2, 3, 4]
        #expect(throws: KVPageStoreError.self) {
            try bytes.withUnsafeBytes { buf in
                try KVPageStore.writeFully(fd: -1, from: buf.baseAddress!,
                                           count: buf.count, offset: 0)
            }
        }
    }

    @Test func recordedSpillFailure_failsSpillReadsUntilReset() throws {
        let (ctx, store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try store.kSlot(layer: 3, position: 0)
        store.advance(by: Self.pageTokens)
        store.flushSpills()

        store.recordSpillError(.ioFailed(operation: "pwrite spill", errno: ENOSPC))
        #expect(store.spillFailure != nil)
        let staging = try #require(ctx.device.makeBuffer(
            length: 2 * store.geometry.kPageBytes, options: .storageModeShared))
        #expect(throws: KVPageStoreError.ioFailed(operation: "pwrite spill",
                                                  errno: ENOSPC)) {
            try store.readSpilledSpan(layer: 3, firstPage: 0, pageCount: 1,
                                      into: staging)
        }

        // Reset rewrites every page before it can be read again, so the
        // recorded failure clears with the rest of the state.
        store.reset()
        #expect(store.spillFailure == nil)
        _ = try store.kSlot(layer: 3, position: 0)
        store.advance(by: Self.pageTokens)
        store.flushSpills()
        try store.readSpilledSpan(layer: 3, firstPage: 0, pageCount: 1,
                                  into: staging)
    }

    // MARK: metadata layout

    @Test func metadataOffsets_areDistinctPerLayerAndPage() throws {
        let (_, store, _) = try makeStore(maxContext: 512)
        let g = store.geometry
        // 2 vectors (min/max) * kvHeads * headDim * FP16 per page.
        let perPage = 2 * 4 * 256 * 2
        #expect(g.metadataBytesPerPage == perPage)
        var seen = Set<Int>()
        for ord in 0..<g.fullLayerOrdinals.count {
            for page in 0..<g.pagesPerLayer {
                let off = g.metadataOffset(layerOrdinal: ord, pageIndex: page)
                #expect(off % perPage == 0)
                #expect(seen.insert(off).inserted)
                #expect(off + perPage <= store.metadataBuffer.length)
            }
        }
    }
}
