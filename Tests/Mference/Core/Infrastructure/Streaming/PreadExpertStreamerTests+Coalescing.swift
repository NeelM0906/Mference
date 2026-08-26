import Darwin
import Foundation
import Metal
import Testing

@testable import Mference

/// Coalesced miss reads: misses whose blobs are adjacent on disk are fetched
/// with one scattered `preadv` per contiguous run instead of one `pread` per
/// expert. Correctness must be identical for uniform layouts, permuted
/// `expertOffsets` tables, and the speculative fill path.
extension PreadExpertStreamerTests {

  @Test func coalescedReadRuns_groupsContiguousOffsets() {
    let stride = UInt64(Self.expertStride)
    // Offsets 0,1s,2s are one run; 4s,5s another; 7s alone. Input order is
    // scrambled to prove the grouping sorts by disk position first.
    let offsets: [UInt64] = [4 * stride, 0, 2 * stride, 7 * stride, stride, 5 * stride]
    let runs = PreadExpertStreamer.coalescedReadRuns(offsets: offsets, stride: stride)
    #expect(runs == [[1, 4, 2], [0, 5], [3]])
  }

  @Test func coalescedReadRuns_duplicateOffsetsNeverShareARun() {
    let stride = UInt64(Self.expertStride)
    let offsets: [UInt64] = [0, 0, stride]
    let runs = PreadExpertStreamer.coalescedReadRuns(offsets: offsets, stride: stride)
    // Overlapping reads must stay separate syscalls; only one duplicate can
    // extend into the following contiguous blob.
    #expect(runs.count == 2)
    #expect(runs.flatMap { $0 }.sorted() == [0, 1, 2])
    for run in runs {
      let sortedOffsets = run.map { offsets[$0] }
      #expect(sortedOffsets == sortedOffsets.sorted())
      for pair in zip(sortedOffsets, sortedOffsets.dropFirst()) {
        #expect(pair.1 == pair.0 + stride)
      }
    }
  }

  @Test func cachePlanWithAdjacentMisses_readsEverySlotCorrectly() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device,
      slotCount: Self.numExperts)

    // Cold cache: all four experts miss and are contiguous on disk, so this
    // exercises a single multi-entry scattered read.
    let results = try streamer.loadExpertsCached(experts: Array(0..<Self.numExperts))
    for (expert, r) in results.enumerated() {
      let got = Self.bytes(of: r.buffer, offset: r.offset, count: Self.expertStride)
      #expect(
        got.allSatisfy { $0 == Self.tagByte(expert) },
        "expert \(expert) slot not uniformly tagged after coalesced read")
    }
  }

  @Test func cachePlanWithPermutedOffsetTable_readsEverySlotCorrectly() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let stride = UInt64(Self.expertStride)
    // Expert id -> disk position: id 0 lives at blob 2, id 1 at blob 3,
    // id 2 at blob 0, id 3 at blob 1. Runs form on disk order (ids 2,3 then
    // 0,1), not id order.
    let positions = [2, 3, 0, 1]
    let layout = StreamLayout(
      path: url.path,
      streamOffset: Self.streamOffset,
      streamSize: Self.streamSize,
      expertsPerLayer: Self.numExperts,
      expertStride: stride,
      expertOffsets: positions.map { UInt64($0) * stride })
    let streamer = try PreadExpertStreamer(
      layout: layout, device: device, slotCount: Self.numExperts)

    let results = try streamer.loadExpertsCached(experts: Array(0..<Self.numExperts))
    for (expert, r) in results.enumerated() {
      let got = Self.bytes(of: r.buffer, offset: r.offset, count: Self.expertStride)
      #expect(
        got.allSatisfy { $0 == Self.tagByte(positions[expert]) },
        "expert \(expert) should hold the blob at disk position \(positions[expert])")
    }
  }

  @Test func speculativeFillWithAdjacentExperts_publishesTaggedSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device,
      slotCount: Self.numExperts)

    let reservation = streamer.reserveSpeculativeSlots(
      experts: [1, 2], keepEvictable: 0)
    #expect(reservation.count == 2)
    let bytes = streamer.executeSpeculativeReservation(reservation)
    #expect(bytes == UInt64(2 * Self.expertStride))

    let resident = streamer.residentExpertsSnapshot()
    for entry in reservation {
      #expect(resident[entry.slot] == entry.expert)
    }
    // A follow-up plan must treat both as hits and read nothing.
    let plan = streamer.planExpertsCached(experts: [1, 2])
    #expect(plan.misses.isEmpty)
    let results = try streamer.executeExpertCachePlan(plan)
    for (index, expert) in [1, 2].enumerated() {
      let got = Self.bytes(
        of: results[index].buffer, offset: results[index].offset,
        count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  @Test func coalescedPlanHittingEOF_throwsInsteadOfPublishing() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device,
      slotCount: Self.numExperts)

    // Truncate mid-run: experts 2 and 3 are one contiguous run, but the file
    // now ends inside expert 3's blob.
    let truncatedLen =
      off_t(Self.streamOffset) + off_t(3 * Self.expertStride + Self.expertStride / 2)
    #expect(truncate(url.path, truncatedLen) == 0)

    #expect(throws: StreamerError.self) {
      _ = try streamer.loadExpertsCached(experts: [2, 3])
    }
    // The failed run must not publish either slot as resident.
    let resident = streamer.residentExpertsSnapshot()
    #expect(!resident.contains(3))
  }

}
