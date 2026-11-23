import Foundation
import Mference

/// The `MFERENCE_PHASES=1` lines for an Inkling 2 run, in the style of the
/// MiniCPM5 branch: totals and per-token figures from the runner's
/// `Inkling2DecodePhaseStats` and the readback mode the counts show.
/// `decodeSeconds` is the run's decode time from its footer.
func inkling2PhaseReport(_ p: Inkling2DecodePhaseStats, decodeSeconds: Double) -> String {
    let steps = Double(max(1, p.decodeSteps))
    let ms = { (seconds: Double) in String(format: "%.1f", seconds * 1000) }
    let per = { (seconds: Double) in String(format: "%.2f", seconds * 1000 / steps) }
    let mode: String
    if p.stagedVisits == 0 {
        mode = "none (no staged layers)"
    } else if p.deferredVisits == 0 {
        mode = "split"
    } else if p.deferredVisits == p.stagedVisits {
        mode = "deferred"
    } else {
        mode = "mixed (\(p.deferredVisits) of \(p.stagedVisits) deferred)"
    }
    var lines = "\n[phases over \(p.decodeSteps) decode steps, decode "
    lines += String(format: "%.0f", decodeSeconds * 1000) + " ms, staged readbacks: " + mode + "]\n"
    lines += "  runner wall:       " + ms(p.wallSeconds) + " ms (" + per(p.wallSeconds) + " ms/token)\n"
    let cpu = max(0, p.wallSeconds - p.blockedSeconds)
    lines += "  cpu encode+commit: " + ms(cpu) + " ms (" + per(cpu) + " ms/token)"
    lines += p.stagedVisits > p.deferredVisits ? ", slot loads included\n" : "\n"
    lines += "  waits on the GPU:  " + ms(p.blockedSeconds) + " ms (" + per(p.blockedSeconds) + " ms/token)"
    lines += ", of which staged readbacks " + ms(p.readbackSeconds) + " ms\n"
    lines += "  gpu busy:          " + ms(p.gpuBusySeconds) + " ms (" + per(p.gpuBusySeconds) + " ms/token)"
    lines += ", gaps " + ms(p.gpuGapSeconds) + " ms"
    lines += p.deferredVisits > 0 ? ", busy can include waits for deferred slot loads\n" : "\n"
    if p.stagedVisits > 0 {
        let place = p.deferredVisits == 0 ? "on the runner thread"
            : p.deferredVisits == p.stagedVisits ? "on the decode worker, overlapped with the GPU"
            : "split visits on the runner thread, deferred ones on the decode worker"
        lines += "  slot loads:        " + ms(p.slotLoadSeconds) + " ms (" + per(p.slotLoadSeconds) + " ms/token), "
            + place + ", with the store's waits\n"
        let rate = 100.0 * Double(p.hits) / Double(max(1, p.hits + p.misses))
        lines += "  staged visits:     \(p.stagedVisits), hits \(p.hits), misses \(p.misses) ("
            + String(format: "%.1f", Double(p.misses) / steps) + " per token, hit rate "
            + String(format: "%.1f", rate) + "%)\n"
    }
    return lines
}
