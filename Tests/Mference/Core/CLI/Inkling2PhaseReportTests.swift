import Testing
import Mference
@testable import MferenceCLICore

/// The Inkling 2 branch of `MFERENCE_PHASES=1`: totals and per-token figures
/// from the runner's counters, the readback mode the counts show, and where
/// the slot loads ran.
@Suite struct Inkling2PhaseReportTests {
    static func stats(staged: Int, deferred: Int) -> Inkling2DecodePhaseStats {
        var p = Inkling2DecodePhaseStats()
        p.decodeSteps = 4
        p.wallSeconds = 0.200
        p.blockedSeconds = 0.120
        p.gpuBusySeconds = 0.100
        p.gpuGapSeconds = 0.010
        p.stagedVisits = staged
        p.deferredVisits = deferred
        p.hits = 30
        p.misses = 10
        p.slotLoadSeconds = 0.012
        p.readbackSeconds = deferred == staged ? 0 : 0.040
        return p
    }

    @Test func splitReadbacksPutSlotLoadsOnTheRunnerThread() {
        let text = inkling2PhaseReport(Self.stats(staged: 48, deferred: 0), decodeSeconds: 0.25)
        #expect(text.hasPrefix("\n[phases over 4 decode steps, decode 250 ms, staged readbacks: split]\n"))
        #expect(text.contains("  runner wall:       200.0 ms (50.00 ms/token)\n"))
        #expect(text.contains("  cpu encode+commit: 80.0 ms (20.00 ms/token), slot loads included\n"))
        #expect(text.contains("  waits on the GPU:  120.0 ms (30.00 ms/token), of which staged readbacks 40.0 ms\n"))
        #expect(text.contains("  gpu busy:          100.0 ms (25.00 ms/token), gaps 10.0 ms\n"))
        #expect(text.contains("  slot loads:        12.0 ms (3.00 ms/token), on the runner thread, with the store's waits\n"))
        #expect(text.hasSuffix("  staged visits:     48, hits 30, misses 10 (2.5 per token, hit rate 75.0%)\n"))
    }

    @Test func deferredReadbacksAreLabelledOverlapped() {
        let text = inkling2PhaseReport(Self.stats(staged: 48, deferred: 48), decodeSeconds: 0.25)
        #expect(text.contains("staged readbacks: deferred]"))
        #expect(text.contains("  cpu encode+commit: 80.0 ms (20.00 ms/token)\n"))
        #expect(text.contains("of which staged readbacks 0.0 ms"))
        #expect(text.contains("on the decode worker, overlapped with the GPU, with the store's waits\n"))
        let mixed = inkling2PhaseReport(Self.stats(staged: 48, deferred: 12), decodeSeconds: 0.25)
        #expect(mixed.contains("staged readbacks: mixed (12 of 48 deferred)]"))
        #expect(mixed.contains("split visits on the runner thread, deferred ones on the decode worker, with the store's waits\n"))
    }

    /// With every layer resident there are no staged visits: no slot lines.
    @Test func withoutStagedLayersThereAreNoSlotLines() {
        var p = Self.stats(staged: 0, deferred: 0)
        p.hits = 0
        p.misses = 0
        p.slotLoadSeconds = 0
        p.readbackSeconds = 0
        let text = inkling2PhaseReport(p, decodeSeconds: 0.25)
        #expect(text.contains("staged readbacks: none (no staged layers)]"))
        #expect(!text.contains("slot loads") && !text.contains("staged visits"))
        #expect(text.hasSuffix("  gpu busy:          100.0 ms (25.00 ms/token), gaps 10.0 ms\n"))
    }
}
