import Darwin
import Foundation
import Metal
import Testing

@testable import Mference

/// Speculative cross-layer prefetch primitives. The contract these pin down:
/// a guess may waste a slot and some bandwidth, but it must never corrupt a
/// slot the real plan is using, never make the real plan unplaceable, and
/// never shift the eviction statistics in favour of an unconfirmed expert.
extension PreadExpertStreamerTests {

    private func makeStreamer(_ url: URL, slots: Int = 4) throws -> PreadExpertStreamer {
        let device = try MetalContext().device
        return try PreadExpertStreamer(layout: Self.makeLayout(path: url.path),
                                       device: device,
                                       slotCount: slots)
    }

    /// The residency probe is read-only: it reports the non-resident subset and
    /// leaves the LFU counters alone, so probing cannot make a guessed expert
    /// look popular. Proof: after probing expert 3 many times, expert 3 is
    /// still the first slot evicted — a `planExpertsCached` bump would have
    /// protected it.
    @Test func nonResidentExpertsProbeDoesNotBumpUseCounts() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try makeStreamer(url, slots: 2)

        _ = try streamer.loadExpertsCached(experts: [0])
        #expect(streamer.nonResidentExperts([0, 1, 2]) == [1, 2])
        // Duplicates collapse, negatives are dropped, order is preserved.
        #expect(streamer.nonResidentExperts([2, 2, 1, 0]) == [2, 1])

        for _ in 0..<20 { _ = streamer.nonResidentExperts([3]) }
        // Fill both slots: 0 (used once) and 3 (probed, never used).
        let reservation = streamer.reserveSpeculativeSlots(experts: [3], keepEvictable: 0)
        streamer.executeSpeculativeReservation(reservation)
        // The real plan needs a new expert; the probed-but-unconfirmed 3 must
        // be the victim, not the confirmed 0.
        let plan = streamer.planExpertsCached(experts: [1])
        #expect(plan.misses == [0])
        #expect(streamer.residentExpertsSnapshot().contains(0))
        #expect(!streamer.residentExpertsSnapshot().contains(3))
    }

    /// A confirmed prediction turns into a free hit: the speculative read lands
    /// the right bytes in a slot, and the following real plan reports zero
    /// misses for that expert.
    @Test func confirmedSpeculativeLoadBecomesAHit() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try makeStreamer(url)

        let missing = streamer.nonResidentExperts([2, 3])
        #expect(missing == [2, 3])
        let reservation = streamer.reserveSpeculativeSlots(experts: missing, keepEvictable: 0)
        #expect(reservation.count == 2)
        let bytes = streamer.executeSpeculativeReservation(reservation)
        #expect(bytes == UInt64(2 * Self.expertStride))

        // Real plan asks for exactly what was predicted: all hits, no I/O.
        let plan = streamer.planExpertsCached(experts: [2, 3])
        #expect(plan.misses.isEmpty)
        #expect(plan.hits == 2)
        let buffers = try streamer.executeExpertCachePlan(plan)
        for (index, expert) in [2, 3].enumerated() {
            let got = Self.bytes(of: buffers[index].buffer, offset: buffers[index].offset, count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
        }
    }

    /// A misprediction is harmless: the wrong expert occupies a slot, the real
    /// plan evicts it, and the bytes the real plan hands back are still the
    /// bytes it asked for.
    @Test func mispredictedSpeculativeLoadIsHarmless() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try makeStreamer(url, slots: 2)

        let reservation = streamer.reserveSpeculativeSlots(experts: [3], keepEvictable: 0)
        streamer.executeSpeculativeReservation(reservation)
        #expect(streamer.residentExpertsSnapshot().contains(3))

        // Nobody asks for 3. The real plan proceeds normally.
        let plan = streamer.planExpertsCached(experts: [0, 1])
        let buffers = try streamer.executeExpertCachePlan(plan)
        for (index, expert) in [0, 1].enumerated() {
            let got = Self.bytes(of: buffers[index].buffer, offset: buffers[index].offset, count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
        }
        #expect(!streamer.residentExpertsSnapshot().contains(3))
    }

    /// Eviction protection: `keepEvictable` slots are always left for the real
    /// plan, so speculation can never starve it. With every slot reserved for
    /// the real plan, speculation declines to run at all.
    @Test func keepEvictableBoundsSpeculativeReservation() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try makeStreamer(url)

        #expect(streamer.reserveSpeculativeSlots(experts: [0, 1, 2, 3],
                                                 keepEvictable: 4).isEmpty)
        let bounded = streamer.reserveSpeculativeSlots(experts: [0, 1, 2, 3],
                                                       keepEvictable: 3)
        #expect(bounded.count == 1)
        streamer.executeSpeculativeReservation(bounded)

        // Whatever speculation took, a full top-k real plan still places.
        let plan = streamer.planExpertsCachedIfPossible(experts: [0, 1, 2, 3])
        #expect(plan != nil)
    }

    /// The join contract, from the streamer's side: while a speculative read is
    /// outstanding its slots are neither matchable nor evictable, so a real
    /// plan that forgot to join still cannot hand out a buffer being written.
    @Test func inFlightSlotsAreInvisibleToTheRealPlan() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try makeStreamer(url)

        // Reserve but deliberately do not execute: the slots stay in flight.
        let reservation = streamer.reserveSpeculativeSlots(experts: [2, 3], keepEvictable: 0)
        #expect(reservation.count == 2)
        let inFlight = Set(reservation.map(\.slot))

        // Two slots remain, so a two-expert plan fits and must avoid them.
        let plan = streamer.planExpertsCachedIfPossible(experts: [0, 1])
        #expect(plan != nil)
        #expect(plan!.assignedSlots.allSatisfy { !inFlight.contains($0) })
        // A third expert cannot be placed while the speculation holds two slots.
        #expect(streamer.planExpertsCachedIfPossible(experts: [0, 1, 2]) == nil)

        // Completing the read releases the reservation.
        streamer.executeSpeculativeReservation(reservation)
        #expect(streamer.planExpertsCachedIfPossible(experts: [0, 1, 2]) != nil)
    }

    /// A speculative read that fails leaves the slot empty rather than claiming
    /// to hold an expert whose bytes never arrived.
    @Test func failedSpeculativeReadLeavesSlotEmpty() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try makeStreamer(url)

        // Expert index past the layer's expert count: the offset check rejects
        // it, so the read throws inside the concurrent block.
        let reservation = streamer.reserveSpeculativeSlots(experts: [Self.numExperts + 5],
                                                           keepEvictable: 0)
        #expect(reservation.count == 1)
        let bytes = streamer.executeSpeculativeReservation(reservation)
        #expect(bytes == 0)
        #expect(streamer.residentExpertsSnapshot()[reservation[0].slot] == -1)
        // And the slot is usable again.
        #expect(streamer.planExpertsCachedIfPossible(experts: [0, 1, 2, 3]) != nil)
    }
}
