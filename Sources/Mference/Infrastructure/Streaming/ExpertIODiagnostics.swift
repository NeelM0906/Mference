import Darwin
import Foundation

/// Phase-report counters for routed-expert I/O, populated only when the
/// phase report is enabled (`MFERENCE_PHASES=1`). Nothing here changes what is
/// read, when, or into which slot: the counters observe the existing reads
/// and plans, so output is identical with the report on or off.

/// Timing of every routed-expert read issued by a `PreadExpertStreamer`: one
/// sample per `pread`/`preadv` run (a run is one or more experts stored back
/// to back on disk), whether it was a demand miss, an eager fill, a
/// speculative prefetch or a prefill sweep.
public final class ExpertReadStats: @unchecked Sendable {
    /// The process-wide instance the streamers report to. Nil unless the phase
    /// report is enabled, so a disabled run takes no lock and reads no clock.
    public static let shared: ExpertReadStats? =
        ProcessInfo.processInfo.environment["MFERENCE_PHASES"] == "1"
            ? ExpertReadStats() : nil

    /// Upper bounds, in microseconds, of the latency histogram buckets; the
    /// last bucket is open-ended.
    public static let bucketUpperMicros: [UInt64] =
        [100, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]

    public struct Snapshot: Equatable, Sendable {
        public var reads: UInt64 = 0
        public var experts: UInt64 = 0
        public var bytes: UInt64 = 0
        public var totalNanos: UInt64 = 0
        public var maxNanos: UInt64 = 0
        /// `bucketUpperMicros.count + 1` counts; the last is `>=` the last bound.
        public var buckets: [UInt64] =
            [UInt64](repeating: 0, count: ExpertReadStats.bucketUpperMicros.count + 1)

        public var meanMicros: Double {
            reads == 0 ? 0 : Double(totalNanos) / Double(reads) / 1_000
        }
    }

    private let lock = NSLock()
    private var state = Snapshot()

    public init() {}

    /// Records one completed read run of `experts` experts and `bytes` bytes.
    public func record(experts: Int, bytes: UInt64, nanos: UInt64) {
        let micros = nanos / 1_000
        let bucket = Self.bucketUpperMicros.firstIndex { micros < $0 }
            ?? Self.bucketUpperMicros.count
        lock.lock()
        state.reads &+= 1
        state.experts &+= UInt64(experts)
        state.bytes &+= bytes
        state.totalNanos &+= nanos
        if nanos > state.maxNanos { state.maxNanos = nanos }
        state.buckets[bucket] &+= 1
        lock.unlock()
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return state
    }

    public func reset() {
        lock.lock()
        state = Snapshot()
        lock.unlock()
    }
}

/// Bytes this process has physically read from or written to storage, from
/// `proc_pid_rusage` (`ri_diskio_bytesread` / `ri_diskio_byteswritten`).
/// Reads served by the unified buffer cache do not count. The counter is
/// process-wide: besides routed-expert reads it includes page faults on the
/// mapped core weights, KV SSD-tier traffic, page-cache warming and any other
/// file reads, so it is not a per-expert-read measurement.
public enum ProcessDiskIO {
    public struct Counters: Equatable, Sendable {
        public var bytesRead: UInt64
        public var bytesWritten: UInt64
    }

    public static func current() -> Counters? {
        var info = rusage_info_v2()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V2, $0)
            }
        }
        guard status == 0 else { return nil }
        return Counters(bytesRead: info.ri_diskio_bytesread,
                        bytesWritten: info.ri_diskio_byteswritten)
    }
}

/// Per-(layer, expert) routed hit and miss counts over the decode window,
/// plus the layer-step tally the all-hit rate is computed from. A hit is a
/// routed expert already resident in its layer's slot cache when the step
/// was planned (or served by the GPU slot map); a miss had to be read.
public struct ExpertRoutingCounters: Equatable, Sendable {
    public let numLayers: Int
    public let numExperts: Int
    public private(set) var hits: [UInt32]
    public private(set) var misses: [UInt32]
    public private(set) var layerSteps: UInt64 = 0
    public private(set) var allHitLayerSteps: UInt64 = 0
    /// Layer steps whose experts were loaded without a CPU plan (top-k beyond
    /// the planner); they count as steps, never as all-hit, and carry no
    /// per-expert hit or miss.
    public private(set) var unplannedLayerSteps: UInt64 = 0

    public init(numLayers: Int, numExperts: Int) {
        self.numLayers = max(0, numLayers)
        self.numExperts = max(0, numExperts)
        let count = self.numLayers * self.numExperts
        hits = [UInt32](repeating: 0, count: count)
        misses = [UInt32](repeating: 0, count: count)
    }

    /// One routed layer step: `experts` are the routed IDs in router order and
    /// `missIndices` index into `experts` (the plan's `misses`).
    public mutating func recordLayerStep(layer: Int, experts: [Int], missIndices: [Int]) {
        guard layer >= 0, layer < numLayers else { return }
        layerSteps &+= 1
        if missIndices.isEmpty { allHitLayerSteps &+= 1 }
        // Routed top-k is small (8 here), so a linear scan of the miss list
        // is cheaper than building a lookup and allocates nothing.
        for (index, expert) in experts.enumerated() where expert >= 0 && expert < numExperts {
            let cell = layer * numExperts + expert
            if missIndices.contains(index) { misses[cell] &+= 1 } else { hits[cell] &+= 1 }
        }
    }

    public mutating func recordUnplannedLayerStep() {
        layerSteps &+= 1
        unplannedLayerSteps &+= 1
    }

    public func hits(layer: Int, expert: Int) -> UInt32 { hits[layer * numExperts + expert] }
    public func misses(layer: Int, expert: Int) -> UInt32 { misses[layer * numExperts + expert] }

    public var totalHits: UInt64 { hits.reduce(0) { $0 &+ UInt64($1) } }
    public var totalMisses: UInt64 { misses.reduce(0) { $0 &+ UInt64($1) } }

    public func layerTotals(_ layer: Int) -> (hits: UInt64, misses: UInt64) {
        let range = (layer * numExperts)..<((layer + 1) * numExperts)
        return (hits[range].reduce(0) { $0 &+ UInt64($1) },
                misses[range].reduce(0) { $0 &+ UInt64($1) })
    }

    /// Distinct (layer, expert) pairs that missed at least once.
    public var distinctMissedExperts: Int { misses.reduce(0) { $0 + ($1 > 0 ? 1 : 0) } }

    /// The `limit` (layer, expert) pairs with the most misses, most first;
    /// ties go to the lower layer, then the lower expert.
    public func topMissed(_ limit: Int) -> [(layer: Int, expert: Int, misses: UInt32, hits: UInt32)] {
        misses.indices
            .filter { misses[$0] > 0 }
            .sorted { misses[$0] != misses[$1] ? misses[$0] > misses[$1] : $0 < $1 }
            .prefix(max(0, limit))
            .map { (layer: $0 / numExperts, expert: $0 % numExperts,
                    misses: misses[$0], hits: hits[$0]) }
    }

    public mutating func reset() {
        self = ExpertRoutingCounters(numLayers: numLayers, numExperts: numExperts)
    }
}

/// Text lines the CLI phase report prints for the counters above. Kept here,
/// beside the counters, so the format is unit-testable without a model.
public enum ExpertIOPhaseReport {
    static func mb(_ bytes: UInt64) -> String {
        String(format: "%.1f", Double(bytes) / 1_048_576)
    }

    static func pct(_ part: UInt64, _ whole: UInt64) -> String {
        whole == 0 ? "0.0" : String(format: "%.1f", 100 * Double(part) / Double(whole))
    }

    static func readLines(_ label: String, _ s: ExpertReadStats.Snapshot) -> String {
        var line = "  expert reads (\(label)): \(s.reads) runs, \(s.experts) experts, "
        line += mb(s.bytes) + " MB requested; mean "
        line += String(format: "%.0f", s.meanMicros) + " us, max "
        line += String(format: "%.2f", Double(s.maxNanos) / 1e6) + " ms\n"
        if s.reads > 0 {
            var parts: [String] = []
            for (index, count) in s.buckets.enumerated() {
                let name = index < ExpertReadStats.bucketUpperMicros.count
                    ? "<\(ExpertReadStats.bucketUpperMicros[index])"
                    : ">=\(ExpertReadStats.bucketUpperMicros.last ?? 0)"
                parts.append("\(name) \(count)")
            }
            line += "    read latency (us): " + parts.joined(separator: ", ") + "\n"
        }
        return line
    }

    public static func lines(routing: ExpertRoutingCounters?,
                             decodeTokens: Int,
                             decodeReads: ExpertReadStats.Snapshot?,
                             decodeDiskStart: ProcessDiskIO.Counters?,
                             diskNow: ProcessDiskIO.Counters?,
                             prefillReads: ExpertReadStats.Snapshot?,
                             prefillDiskBytes: UInt64?,
                             topMissed: Int = 8) -> String {
        var out = ""
        if let r = routing, r.layerSteps > 0 {
            let hits = r.totalHits, misses = r.totalMisses
            out += "  routed experts: \(hits) hit, \(misses) miss ("
            out += pct(hits, hits &+ misses) + "% hit) over \(r.layerSteps) layer steps"
            if decodeTokens > 0 {
                out += String(format: ", %.2f misses per generated token",
                              Double(misses) / Double(decodeTokens))
            }
            out += "\n"
            out += "  all-hit layer steps (expert counters): \(r.allHitLayerSteps)"
            out += "/\(r.layerSteps) (" + pct(r.allHitLayerSteps, r.layerSteps) + "%)"
            if r.unplannedLayerSteps > 0 {
                out += ", \(r.unplannedLayerSteps) unplanned"
            }
            out += "\n"
            let top = r.topMissed(topMissed)
            out += "  expert misses: \(r.distinctMissedExperts) distinct layer:expert pairs"
            if !top.isEmpty {
                out += "; most missed "
                out += top.map { "L\($0.layer):E\($0.expert) \($0.misses)m/\($0.hits)h" }
                    .joined(separator: ", ")
            }
            out += "\n"
            out += "  misses by layer: "
            out += (0..<r.numLayers).map { "\($0):\(r.layerTotals($0).misses)" }
                .joined(separator: " ") + "\n"
        }
        if let decodeReads { out += readLines("decode", decodeReads) }
        if let prefillReads { out += readLines("prefill", prefillReads) }
        if let diskNow {
            out += "  disk physically read (proc_pid_rusage): "
            var parts: [String] = []
            if let start = decodeDiskStart, diskNow.bytesRead >= start.bytesRead {
                parts.append("decode " + mb(diskNow.bytesRead - start.bytesRead) + " MB")
            }
            if let prefillDiskBytes {
                parts.append("prefill " + mb(prefillDiskBytes) + " MB")
            }
            parts.append("process total " + mb(diskNow.bytesRead) + " MB")
            out += parts.joined(separator: ", ") + "\n"
        }
        return out
    }
}
