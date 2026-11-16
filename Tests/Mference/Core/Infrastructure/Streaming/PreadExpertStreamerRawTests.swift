import Foundation
import Metal
import Testing
@testable import Mference

/// Raw-expansion layouts (WS4 GGUF): the file stores each expert in fewer
/// bytes than a slot holds, the runner expands them on the GPU, and only
/// `executeExpertCachePlanRaw` may fill the slots.
@Suite struct PreadExpertStreamerRawTests {
    static let experts = 6, stored = 3_000, slot = 16_384

    static func makeFile() throws -> (URL, [UInt8]) {
        var bytes = [UInt8](repeating: 0, count: experts * stored)
        for i in bytes.indices { bytes[i] = UInt8(truncatingIfNeeded: i &* 31 &+ i / stored) }
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-raw-experts-\(UUID().uuidString).bin")
        try Data(bytes).write(to: url)
        return (url, bytes)
    }

    static func streamer(_ url: URL, device: MTLDevice) throws -> PreadExpertStreamer {
        try PreadExpertStreamer(
            layout: StreamLayout(path: url.path, streamOffset: 0,
                                 streamSize: UInt64(experts * stored), expertsPerLayer: experts,
                                 expertStride: UInt64(slot),
                                 expertOffsets: (0..<experts).map { UInt64($0 * stored) },
                                 rawStoredExpertStride: UInt64(stored)),
            device: device, slotCount: 4, readStats: nil)
    }

    @Test func rawPlanReadsMissesIntoStagingAndPublishesSlots() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let (url, bytes) = try Self.makeFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try Self.streamer(url, device: device)
        let staging = try #require(device.makeBuffer(length: 3 * Self.stored, options: .storageModeShared))

        // Experts 2 and 3 are adjacent on disk (one coalesced run), 5 is not.
        let plan = s.planExpertsCached(experts: [5, 2, 3])
        let result = try s.executeExpertCachePlanRaw(plan, staging: staging.contents(),
                                                     stagingBytes: staging.length)
        #expect(result.misses.count == 3 && result.buffers.count == 3)
        for (i, index) in plan.misses.enumerated() {
            let e = plan.experts[index]
            let got = [UInt8](UnsafeRawBufferPointer(start: staging.contents().advanced(by: result.misses[i].stagingOffset),
                                                     count: Self.stored))
            #expect(got == Array(bytes[(e * Self.stored)..<((e + 1) * Self.stored)]), "expert \(e)")
            #expect(result.misses[i].slotOffset == plan.assignedSlots[index] * Self.slot)
            #expect(result.buffers[index].size == UInt64(Self.slot))
        }
        // Published: a second plan for the same experts is all hits.
        let again = s.planExpertsCached(experts: [2, 3, 5])
        #expect(again.misses.isEmpty && again.hits == 3)
        #expect(try s.executeExpertCachePlanRaw(again, staging: staging.contents(),
                                                stagingBytes: staging.length).misses.isEmpty)
    }

    /// The plain read paths would put unexpanded bytes into a slot, so they
    /// refuse; a staging buffer too small for the misses is refused too.
    @Test func plainReadsAndShortStagingAreRefused() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let (url, _) = try Self.makeFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try Self.streamer(url, device: device)
        #expect(throws: StreamerError.self) { _ = try s.loadExpert(layer: 0, expert: 1, slot: 0) }
        #expect(throws: StreamerError.self) {
            _ = try s.executeExpertCachePlan(s.planExpertsCached(experts: [1]))
        }
        let small = try #require(device.makeBuffer(length: Self.stored, options: .storageModeShared))
        #expect(throws: StreamerError.self) {
            _ = try s.executeExpertCachePlanRaw(s.planExpertsCached(experts: [3, 4]),
                                                staging: small.contents(), stagingBytes: small.length)
        }
    }
}
