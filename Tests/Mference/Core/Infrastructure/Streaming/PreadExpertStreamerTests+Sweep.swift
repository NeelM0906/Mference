import Darwin
import Foundation
import Metal
import Testing

@testable import Mference

/// The unbuffered prefill-sweep descriptor: opened only on request, never
/// leaked by a failed init, and invisible in the bytes a plan delivers.
extension PreadExpertStreamerTests {

    /// Open descriptors in this process that refer to `path`. Counting by path
    /// rather than in total keeps the check independent of other tests.
    static func openDescriptorCount(path: String) -> Int {
        let target = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        var count = 0
        for descriptor in 0..<Int32(getdtablesize()) {
            var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN))
            guard fcntl(descriptor, F_GETPATH, &buffer) != -1 else { continue }
            let open = URL(fileURLWithPath: String(cString: buffer))
                .resolvingSymlinksInPath().path
            if open == target { count += 1 }
        }
        return count
    }

    @Test(arguments: [false, true])
    func shortFileInitThrowsWithoutLeakingDescriptors(unbufferedSweep: Bool) throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        // One expert short of the stream the layout describes.
        let short = off_t(Self.streamOffset + Self.streamSize) - off_t(Self.expertStride)
        #expect(truncate(url.path, short) == 0)
        let device = try MetalContext().device

        #expect(Self.openDescriptorCount(path: url.path) == 0)
        #expect(throws: StreamerError.self) {
            _ = try PreadExpertStreamer(layout: Self.makeLayout(path: url.path),
                                        device: device,
                                        slotCount: 2,
                                        unbufferedSweep: unbufferedSweep)
        }
        #expect(Self.openDescriptorCount(path: url.path) == 0)
    }

    @Test(arguments: [false, true])
    func sweepDescriptorOpensOnlyOnRequest(unbufferedSweep: Bool) throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device

        var streamer: PreadExpertStreamer? = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path),
            device: device,
            slotCount: 2,
            unbufferedSweep: unbufferedSweep)
        #expect(streamer?.sweepsBypassPageCache == unbufferedSweep)
        #expect(Self.openDescriptorCount(path: url.path) == (unbufferedSweep ? 2 : 1))
        streamer = nil
        #expect(Self.openDescriptorCount(path: url.path) == 0)
    }

    /// `pread` returns the same bytes through either descriptor, so a sweep
    /// plan lands exactly what a buffered plan lands: the file's bytes.
    @Test func unbufferedSweepPlanDeliversFileBytes() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let layout = Self.makeLayout(path: url.path)
        let experts = Array((0..<Self.numExperts).reversed())

        let swept = try PreadExpertStreamer(layout: layout, device: device,
                                            slotCount: Self.numExperts,
                                            unbufferedSweep: true)
        #expect(swept.sweepsBypassPageCache)
        let buffered = try PreadExpertStreamer(layout: layout, device: device,
                                               slotCount: Self.numExperts)
        let sweptViews = try swept.executeExpertCachePlan(
            swept.planExpertsCached(experts: experts), sweep: true)
        let bufferedViews = try buffered.executeExpertCachePlan(
            buffered.planExpertsCached(experts: experts), sweep: false)

        for (index, expert) in experts.enumerated() {
            let a = Self.bytes(of: sweptViews[index].buffer,
                               offset: sweptViews[index].offset,
                               count: Self.expertStride)
            let b = Self.bytes(of: bufferedViews[index].buffer,
                               offset: bufferedViews[index].offset,
                               count: Self.expertStride)
            #expect(a.allSatisfy { $0 == Self.tagByte(expert) }, "expert \(expert)")
            #expect(a == b, "expert \(expert)")
        }
    }

    /// The prefill lookahead reads through the sweep descriptor too; its slots
    /// must hold the file's bytes and turn into hits.
    @Test func unbufferedSweepReservationDeliversFileBytes() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(layout: Self.makeLayout(path: url.path),
                                               device: device,
                                               slotCount: Self.numExperts,
                                               unbufferedSweep: true)

        let reservation = streamer.reserveSpeculativeSlots(experts: [1, 3], keepEvictable: 0)
        #expect(reservation.count == 2)
        let bytes = streamer.executeSpeculativeReservation(reservation, sweep: true)
        #expect(bytes == 2 * UInt64(Self.expertStride))

        let plan = streamer.planExpertsCached(experts: [1, 3])
        #expect(plan.misses.isEmpty)
        let views = streamer.expertCachePlanBuffers(plan)
        for (index, expert) in [1, 3].enumerated() {
            let got = Self.bytes(of: views[index].buffer, offset: views[index].offset,
                                 count: Self.expertStride)
            #expect(got.allSatisfy { $0 == Self.tagByte(expert) }, "expert \(expert)")
        }
    }
}
