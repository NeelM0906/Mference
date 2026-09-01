import Testing
import Foundation
import Metal
@testable import Mference

/// Does the routed-expert streaming machinery accept Qwen3.8-Flash-Next's
/// pool shape — 512 experts per layer at a 2 768 896-byte stride?
///
/// Every shipped family has 128–256 experts of 5–20 MiB. Flash-Next inverts
/// that: twice the experts at a fifth the size. This suite pins the axes
/// where an "experts fit in a byte / a stride is large" assumption would
/// break, at the layout and streaming layer.
///
/// **Scope.** This is the plumbing, which passes. The *kernels* do not:
/// `MoE.encodeRouterGemma4` and `PrefillRouter.encode` both assert
/// `numExperts <= 256`, `MoE`'s router-logits buffer is allocated for 256
/// floats, and `moe.metal`'s `kRouterMaxPerLane = 8` is derived from
/// "num_experts <= 256" over 32 lanes. Raising those is router-kernel work
/// (512 experts, top-10 against a top-8 selection kernel) and belongs with the
/// forward math, not here. Nothing can reach them yet: the family is refused
/// at `ManifestReader.peekFamily`.
@Suite struct FlashNextExpertPoolGeometryTests {

    /// The production values, from the installed manifest at revision
    /// `de4b8e4d`.
    static let productionExperts = 512
    static let productionStride: UInt64 = 2_768_896

    @Test func productionStrideIsPageAlignedAndMatchesTheArch() {
        let pageSize = UInt64(getpagesize())
        #expect(Self.productionStride % pageSize == 0,
                "expert blobs must start on a page for the mmap/pread paths")
        #expect(Self.productionStride / pageSize == 169)

        let arch = ArchConfig.qwen38FlashNext_180B_A3_5B
        #expect(arch.numExperts == Self.productionExperts)
        #expect(arch.topKExperts == 10)
        // A layer's whole pool: 512 x 2.64 MiB = 1.32 GiB, and 48 of them.
        let layerBytes = UInt64(arch.numExperts) * Self.productionStride
        #expect(layerBytes == 1_417_674_752)
        #expect(layerBytes < UInt64(Int.max))
    }

    /// `StreamLayout`'s offset arithmetic must stay exact at the production
    /// numbers — the last expert of the last layer is ~68 GB into the pool, so
    /// a 32-bit intermediate anywhere would wrap.
    @Test func streamLayoutOffsetsAreExactAtProductionScale() {
        let layers = 48
        let layout = StreamLayout(
            path: "/dev/null",
            streamOffset: 0,
            streamSize: UInt64(layers * Self.productionExperts) * Self.productionStride,
            expertsPerLayer: Self.productionExperts,
            expertStride: Self.productionStride)

        for (layer, expert) in [(0, 0), (0, 511), (1, 0), (47, 511)] {
            let expected = UInt64(layer) * UInt64(Self.productionExperts)
                * Self.productionStride + UInt64(expert) * Self.productionStride
            #expect(layout.expertOffset(layer: layer, expert: expert) == expected)
        }
        let last = layout.expertOffset(layer: layers - 1, expert: Self.productionExperts - 1)
        #expect(last + Self.productionStride == layout.streamSize)
        #expect(last > UInt64(UInt32.max), "the pool is past the 32-bit range")
    }

    /// The slot cache's GPU-visible expert -> slot table is `Int16`. 512 fits
    /// with room to spare, but nothing had exercised an expert id above 255,
    /// so this pins it.
    @Test func slotMapAddressesFiveHundredAndTwelveExperts() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        // Real expert bytes at the production stride would be a 1.3 GB
        // fixture. The *count* is what the table indexes, so the stride here
        // is the minimum page-aligned one.
        let stride = UInt64(getpagesize())
        let experts = Self.productionExperts
        let (url, dir) = try Self.writePool(experts: experts, stride: stride)
        defer { try? FileManager.default.removeItem(at: dir) }

        let layout = StreamLayout(path: url.path,
                                  streamOffset: 0,
                                  streamSize: UInt64(experts) * stride,
                                  expertsPerLayer: experts,
                                  expertStride: stride)
        let streamer = try PreadExpertStreamer(layout: layout, device: device,
                                               slotCount: 16, cachePolicy: .lfu)
        #expect(streamer.slotMapBinding.table.length
                == experts * MemoryLayout<Int16>.stride)
        #expect(Self.productionExperts <= Int(Int16.max),
                "the slot map cannot address this many experts")

        // Read across the whole id range, including ids a UInt8 could not hold.
        for expert in [0, 255, 256, 511] {
            let read = try streamer.loadExpert(layer: 0, expert: expert)
            #expect(read.size == stride)
            let byte = read.buffer.contents().advanced(by: Int(read.offset))
                .assumingMemoryBound(to: UInt8.self)
            #expect(Int(byte[0]) | (Int(byte[1]) << 8) == expert,
                    "expert \(expert) read the wrong blob")
        }
    }

    /// The LFU cache's bookkeeping is sized from `expertsPerLayer`, so a plan
    /// over top-10 of 512 must place, hit and evict correctly.
    @Test func cachePlansTopTenOfFiveHundredAndTwelve() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let stride = UInt64(getpagesize())
        let experts = Self.productionExperts
        let (url, dir) = try Self.writePool(experts: experts, stride: stride)
        defer { try? FileManager.default.removeItem(at: dir) }

        let layout = StreamLayout(path: url.path,
                                  streamOffset: 0,
                                  streamSize: UInt64(experts) * stride,
                                  expertsPerLayer: experts,
                                  expertStride: stride)
        let streamer = try PreadExpertStreamer(layout: layout, device: device,
                                               slotCount: 16, cachePolicy: .lfu)

        let topTen = [500, 501, 7, 255, 256, 300, 480, 511, 42, 199]
        let first = streamer.planExpertsCached(experts: topTen)
        #expect(first.experts.count == 10)
        #expect(first.hits == 0)
        #expect(first.misses.count == 10)
        _ = try streamer.executeExpertCachePlan(first)
        #expect(Set(streamer.residentExpertsSnapshot()).isSuperset(of: topTen))

        let second = streamer.planExpertsCached(experts: topTen)
        #expect(second.hits == 10, "a repeated route must be served from slots")
        #expect(streamer.nonResidentExperts(topTen).isEmpty)
        #expect(streamer.nonResidentExperts([1, 2, 3]) == [1, 2, 3])
    }

    /// `PackedExpertsLayoutReader` must round-trip a 512-entry layer.
    @Test func packedLayoutReadsAFiveHundredAndTwelveExpertLayer() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flashnext-layout-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let experts = Self.productionExperts
        let stride = Self.productionStride
        let entries = (0..<experts).map { expert -> [String: Any] in
            ["expert": expert,
             "offset": UInt64(expert) * stride,
             "size": stride,
             "tensors": ["gate": ["offset": 0, "size": 1024,
                                  "dtype": "U32", "shape": [640, 2560], "bits": 4]]]
        }
        let root: [String: Any] = [
            "expertStride": stride,
            "numLayers": 1,
            "expertsPerLayer": experts,
            "layers": [["layer": 0, "file": "layer_00.bin", "experts": entries]],
        ]
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: exp.appendingPathComponent("layout.json"))

        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)
        #expect(layout.expertsPerLayer == experts)
        #expect(layout.expertStride == stride)
        #expect(layout.layers[0].experts.count == experts)
        #expect(layout.expert(layer: 0, expert: 511).offset
                == UInt64(511) * stride)
    }

    /// A pool of `experts` blobs whose first two bytes are the expert id.
    private static func writePool(experts: Int, stride: UInt64) throws
        -> (URL, URL) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flashnext-pool-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var payload = Data(count: experts * Int(stride))
        for expert in 0..<experts {
            let base = expert * Int(stride)
            payload[base] = UInt8(truncatingIfNeeded: expert)
            payload[base + 1] = UInt8(truncatingIfNeeded: expert >> 8)
        }
        let url = dir.appendingPathComponent("layer_00.bin")
        try payload.write(to: url)
        return (url, dir)
    }
}
