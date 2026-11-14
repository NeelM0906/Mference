import Darwin
import Foundation
import Metal
import Testing

@testable import Mference

/// Phase-report expert I/O counters: per-(layer, expert) hit/miss tallies, the
/// all-hit layer-step rate they reproduce, read timing recorded by the
/// streamer, and the process storage counters from `proc_pid_rusage`.
@Suite struct ExpertIODiagnosticsTests {

  @Test func routingCounters_tallyHitsMissesAndAllHitSteps() {
    var c = ExpertRoutingCounters(numLayers: 2, numExperts: 8)
    c.recordLayerStep(layer: 0, experts: [1, 2, 3], missIndices: [])
    c.recordLayerStep(layer: 0, experts: [1, 4, 5], missIndices: [1, 2])
    c.recordLayerStep(layer: 1, experts: [7, 4], missIndices: [0])
    c.recordUnplannedLayerStep()

    #expect(c.layerSteps == 4)
    #expect(c.allHitLayerSteps == 1)
    #expect(c.unplannedLayerSteps == 1)
    #expect(c.hits(layer: 0, expert: 1) == 2)
    #expect(c.misses(layer: 0, expert: 4) == 1)
    #expect(c.hits(layer: 0, expert: 4) == 0)
    #expect(c.misses(layer: 1, expert: 7) == 1)
    #expect(c.hits(layer: 1, expert: 4) == 1)
    #expect(c.totalHits == 5)
    #expect(c.totalMisses == 3)
    #expect(c.layerTotals(0) == (hits: 4, misses: 2))
    #expect(c.distinctMissedExperts == 3)
  }

  @Test func routingCounters_ignoreOutOfRangeAndOrderTopMissed() {
    var c = ExpertRoutingCounters(numLayers: 2, numExperts: 4)
    c.recordLayerStep(layer: 5, experts: [0], missIndices: [0])
    c.recordLayerStep(layer: 0, experts: [9, -1], missIndices: [0, 1])
    #expect(c.layerSteps == 1)
    #expect(c.totalMisses == 0)

    c.recordLayerStep(layer: 1, experts: [2, 3], missIndices: [0, 1])
    c.recordLayerStep(layer: 1, experts: [2], missIndices: [0])
    c.recordLayerStep(layer: 0, experts: [3], missIndices: [0])
    let top = c.topMissed(2)
    #expect(top.map { [$0.layer, $0.expert, Int($0.misses)] } == [[1, 2, 2], [0, 3, 1]])

    c.reset()
    #expect(c.layerSteps == 0 && c.totalMisses == 0 && c.numExperts == 4)
  }

  @Test func readStats_bucketsByLatency() {
    let stats = ExpertReadStats()
    stats.record(experts: 1, bytes: 100, nanos: 50_000)        // <100 us
    stats.record(experts: 2, bytes: 200, nanos: 1_500_000)     // <2000 us
    stats.record(experts: 1, bytes: 100, nanos: 40_000_000)    // >=16000 us
    let s = stats.snapshot()
    #expect(s.reads == 3)
    #expect(s.experts == 4)
    #expect(s.bytes == 400)
    #expect(s.maxNanos == 40_000_000)
    #expect(s.buckets[0] == 1)
    #expect(s.buckets[4] == 1)
    #expect(s.buckets[ExpertReadStats.bucketUpperMicros.count] == 1)
    #expect(s.buckets.reduce(0, +) == 3)
    stats.reset()
    #expect(stats.snapshot() == ExpertReadStats.Snapshot())
  }

  @Test func streamer_recordsEveryReadRunWhenStatsGiven() throws {
    let url = try PreadExpertStreamerTests.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let stats = ExpertReadStats()
    let streamer = try PreadExpertStreamer(
      layout: PreadExpertStreamerTests.makeLayout(path: url.path), device: device,
      slotCount: 4, readStats: stats)
    let stride = UInt64(PreadExpertStreamerTests.expertStride)

    // Experts 0 and 1 are adjacent on disk: one coalesced run. Expert 3 is
    // its own run. A repeat of the same set is all hits and reads nothing.
    _ = try streamer.loadExpertsCached(experts: [0, 1, 3])
    var s = stats.snapshot()
    #expect(s.reads == 2)
    #expect(s.experts == 3)
    #expect(s.bytes == 3 * stride)
    _ = try streamer.loadExpertsCached(experts: [0, 1, 3])
    s = stats.snapshot()
    #expect(s.reads == 2)

    _ = try streamer.loadExpert(layer: 0, expert: 2)
    s = stats.snapshot()
    #expect(s.reads == 3)
    #expect(s.bytes == 4 * stride)
    #expect(s.totalNanos >= s.maxNanos)
  }

  @Test func streamer_withoutStatsStillReadsCorrectly() throws {
    let url = try PreadExpertStreamerTests.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: PreadExpertStreamerTests.makeLayout(path: url.path), device: device,
      slotCount: 2, readStats: nil)
    let r = try streamer.loadExpert(layer: 0, expert: 3)
    let got = PreadExpertStreamerTests.bytes(
      of: r.buffer, offset: r.offset, count: PreadExpertStreamerTests.expertStride)
    #expect(got.allSatisfy { $0 == PreadExpertStreamerTests.tagByte(3) })
  }

  @Test func processDiskIO_isAvailableAndMonotonic() throws {
    let first = try #require(ProcessDiskIO.current())
    let second = try #require(ProcessDiskIO.current())
    #expect(second.bytesRead >= first.bytesRead)
    #expect(second.bytesWritten >= first.bytesWritten)
  }

  @Test func phaseReport_printsCountersAndAllHitRate() {
    var c = ExpertRoutingCounters(numLayers: 2, numExperts: 4)
    c.recordLayerStep(layer: 0, experts: [0, 1], missIndices: [])
    c.recordLayerStep(layer: 1, experts: [2, 3], missIndices: [1])
    let reads = ExpertReadStats()
    reads.record(experts: 1, bytes: 1_048_576, nanos: 300_000)
    let text = ExpertIOPhaseReport.lines(
      routing: c, decodeTokens: 1,
      decodeReads: reads.snapshot(),
      decodeDiskStart: .init(bytesRead: 1_048_576, bytesWritten: 0),
      diskNow: .init(bytesRead: 3 * 1_048_576, bytesWritten: 0),
      prefillReads: nil, prefillDiskBytes: 4 * 1_048_576)
    #expect(text.contains("routed experts: 3 hit, 1 miss (75.0% hit) over 2 layer steps, 1.00 misses/token"))
    #expect(text.contains("all-hit layer steps (expert counters): 1/2 (50.0%)"))
    #expect(text.contains("most missed L1:E3 1m/0h"))
    #expect(text.contains("misses by layer: 0:0 1:1"))
    #expect(text.contains("expert reads (decode): 1 runs, 1 experts, 1.0 MB requested; mean 300 us, max 0.30 ms"))
    #expect(text.contains("<500 1"))
    #expect(text.contains("disk physically read (proc_pid_rusage): decode 2.0 MB, prefill 4.0 MB, process total 3.0 MB"))
  }

  @Test func phaseReport_isEmptyWithoutCounters() {
    #expect(ExpertIOPhaseReport.lines(
      routing: nil, decodeTokens: 0, decodeReads: nil, decodeDiskStart: nil,
      diskNow: nil, prefillReads: nil, prefillDiskBytes: nil).isEmpty)
  }
}
