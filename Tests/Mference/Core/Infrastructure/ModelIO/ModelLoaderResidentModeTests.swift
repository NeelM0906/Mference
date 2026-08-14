import Foundation
import Metal
import Testing

@testable import Mference

/// Resident-mode `Model` behavior against the toy synthetic install:
/// all-hit plans, synchronous fetches that alias one mapped buffer per
/// layer, skipped advice, and byte parity with the `pread` backend.
@Suite struct ModelLoaderResidentModeTests {

    @Test func residentPlansAreAllHit() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: dir, device: device,
            expecting: .gemma4Toy(),
            streamingMode: .resident)

        let plan = try #require(try model.planRoutedExperts(layer: 0,
                                                            experts: [2, 0, 5]))
        #expect(plan.hits == 3)
        #expect(plan.misses.isEmpty)
        #expect(plan.assignedSlots.isEmpty)

        let advice = try model.adviseRoutedExperts(plan: plan)
        #expect(advice.skipped == 3)
        #expect(advice.calls == 0)
    }

    @Test func residentFetchMatchesPreadBytes() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let resident = try Model.load(
            directoryURL: dir, device: device,
            expecting: .gemma4Toy(),
            streamingMode: .resident)
        let pread = try Model.load(
            directoryURL: dir, device: device,
            expecting: .gemma4Toy(),
            streamingMode: .pread(slotCount: 8))

        for layer in 0..<2 {
            let experts = [0, 3, 5]
            let residentViews = try await resident.fetchRoutedExperts(
                layer: layer, experts: experts)
            let preadViews = try await pread.fetchRoutedExperts(
                layer: layer, experts: experts)
            #expect(residentViews.count == preadViews.count)
            for (r, p) in zip(residentViews, preadViews) {
                #expect(r.length == p.length)
                let rBytes = Data(bytes: r.buffer.contents()
                                      .advanced(by: Int(r.offset)),
                                  count: Int(r.length))
                let pBytes = Data(bytes: p.buffer.contents()
                                      .advanced(by: Int(p.offset)),
                                  count: Int(p.length))
                #expect(rBytes == pBytes)
            }
            // One buffer per expert: residency demands stay at the routed
            // working set instead of the whole layer file.
            #expect(residentViews[0].buffer !== residentViews[1].buffer)
        }
    }

    @Test func residentModeReportsNoSlotCache() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: dir, device: device,
            expecting: .gemma4Toy(),
            streamingMode: .resident)
        _ = try model.routedExpert(layer: 0, expert: 0)
        #expect(model.routedExpertCacheSlotCount(layer: 0) == nil)
        #expect(throws: (any Error).self) {
            _ = try model.routedExpertStreamer(layer: 0)
        }
    }
}
