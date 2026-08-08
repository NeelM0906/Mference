import Testing
import Foundation
import Darwin
import Metal
@testable import Mference

/// The GPU-visible expert->slot table must mirror `slotExpert` at every
/// mutation: publishes appear, evictions and speculative reservations
/// disappear, and entries always agree with the residency snapshot.
@Suite struct PreadExpertStreamerSlotMapTests {

    static let pageSize = Int(getpagesize())
    static let expertStride = 2 * pageSize
    static let numExperts = 6

    static func writeSyntheticLayer() throws -> URL {
        var bytes = [UInt8](repeating: 0, count: numExperts * expertStride)
        for e in 0..<numExperts {
            for i in 0..<expertStride { bytes[e * expertStride + i] = UInt8(0xD0 + e) }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("slotmap-\(UUID().uuidString).bin")
        try Data(bytes).write(to: url)
        return url
    }

    static func makeStreamer(slotCount: Int) throws -> (PreadExpertStreamer, URL) {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try writeSyntheticLayer()
        let layout = StreamLayout(path: url.path,
                                  streamOffset: 0,
                                  streamSize: UInt64(numExperts * expertStride),
                                  expertsPerLayer: numExperts,
                                  expertStride: UInt64(expertStride))
        return (try PreadExpertStreamer(layout: layout,
                                        device: device,
                                        slotCount: slotCount), url)
    }

    static func table(_ streamer: PreadExpertStreamer) -> [Int16] {
        let binding = streamer.slotMapBinding
        let ptr = binding.table.contents()
            .bindMemory(to: Int16.self, capacity: numExperts)
        return (0..<numExperts).map { ptr[$0] }
    }

    @Test("Table mirrors publishes and evictions")
    func tableMirrorsSlotState() throws {
        let (streamer, url) = try Self.makeStreamer(slotCount: 2)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(Self.table(streamer).allSatisfy { $0 == -1 })

        _ = try streamer.loadExpertsCached(experts: [0, 1])
        var t = Self.table(streamer)
        #expect(t[0] >= 0 && t[1] >= 0)

        // Evict by loading two new experts into the 2-slot cache.
        _ = try streamer.loadExpertsCached(experts: [2, 3])
        t = Self.table(streamer)
        #expect(t[0] == -1 && t[1] == -1)
        #expect(t[2] >= 0 && t[3] >= 0)

        // Table always agrees with the snapshot.
        let snapshot = streamer.residentExpertsSnapshot()
        for (slot, expert) in snapshot.enumerated() where expert >= 0 {
            #expect(t[expert] == Int16(slot))
        }
    }

    @Test("Speculative reservations hide entries until the read lands")
    func speculativeReservationHidesEntry() throws {
        let (streamer, url) = try Self.makeStreamer(slotCount: 3)
        defer { try? FileManager.default.removeItem(at: url) }

        _ = try streamer.loadExpertsCached(experts: [4])
        #expect(Self.table(streamer)[4] >= 0)

        let reservation = streamer.reserveSpeculativeSlots(experts: [5],
                                                           keepEvictable: 0)
        #expect(!reservation.isEmpty)
        // In flight: neither the reserved expert nor any stale owner visible.
        #expect(Self.table(streamer)[5] == -1)
        streamer.executeSpeculativeReservation(reservation)
        #expect(Self.table(streamer)[5] >= 0)
    }
}
