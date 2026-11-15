import Foundation
import Metal

/// Stage timing for the shared chunked prefill path (Gemma 4, Gemma QAT,
/// Qwen 3.6), the counterpart of Inkling's routed-loop split under the same
/// `MFERENCE_PREFILL_BREAKDOWN=1` diagnostic toggle. Off by default; when off,
/// nothing here is touched.
///
/// While the toggle is on, the runner commits and waits at each stage boundary
/// inside the attention command buffer so every stage gets its own GPU
/// timestamps. The kernels, their inputs and their order on the serial queue
/// are unchanged, so output is byte-identical; only the timing of the
/// diagnostic run itself is slightly pessimistic (one extra submit and wait
/// per boundary). Read the GPU columns as shares, not as the production
/// schedule: routed tiles, the shared branch and the next tile's SSD read
/// overlap in the real run.
public enum PrefillStage: Int, CaseIterable, Sendable {
    case gdnInputProjections
    case gdnCore
    case gdnOutputProjection
    case attentionInputProjections
    case attentionCore
    case attentionOutputProjection
    case postAttentionAndRouter
    case sharedExpert
    case routedTiles
    case moeTail
    case head

    var label: String {
        switch self {
        case .gdnInputProjections: return "gdn in-proj (+L0 embed)"
        case .gdnCore: return "gdn core (conv+delta+gnorm)"
        case .gdnOutputProjection: return "gdn out-proj"
        case .attentionInputProjections: return "attn in-proj (+L0 embed)"
        case .attentionCore: return "attn core (kv copy+attention)"
        case .attentionOutputProjection: return "attn out-proj (+gate)"
        case .postAttentionAndRouter: return "post-attn norm+router"
        case .sharedExpert: return "shared expert *"
        case .routedTiles: return "routed tiles (GPU) *"
        case .moeTail: return "moe reduce+residual"
        case .head: return "final norm+head"
        }
    }
}

public final class PrefillStageBreakdown: @unchecked Sendable {
    public static let shared = PrefillStageBreakdown()

    private let lock = NSLock()
    private var gpuSeconds = [Double](repeating: 0, count: PrefillStage.allCases.count)
    private var gpuCounts = [Int](repeating: 0, count: PrefillStage.allCases.count)
    private var routedFetchWaitNanos: UInt64 = 0
    private var routedFetchCount = 0
    private var routedPlanNanos: UInt64 = 0
    private var layerWallNanos: UInt64 = 0
    private var chunkWallNanos: UInt64 = 0
    private var chunks = 0
    private var tokens = 0

    /// The diagnostic clock. Out of line, like every recording method, so the
    /// disabled path in the runner is a branch on one hoisted Bool and no
    /// recording code is inlined into the prefill loops.
    @inline(never)
    static func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

    /// Adds a completed command buffer's GPU span to `stage`.
    @inline(never)
    func addGPU(_ stage: PrefillStage, _ commandBuffer: MTLCommandBuffer) {
        let span = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        guard span.isFinite, span > 0 else { return }
        lock.lock()
        gpuSeconds[stage.rawValue] += span
        gpuCounts[stage.rawValue] += 1
        lock.unlock()
    }

    @inline(never)
    func addRoutedFetchWait(nanos: UInt64) {
        lock.lock()
        routedFetchWaitNanos &+= nanos
        routedFetchCount += 1
        lock.unlock()
    }

    @inline(never)
    func addRoutedPlan(nanos: UInt64) {
        lock.lock(); routedPlanNanos &+= nanos; lock.unlock()
    }

    @inline(never)
    func addLayerWall(nanos: UInt64) {
        lock.lock(); layerWallNanos &+= nanos; lock.unlock()
    }

    @inline(never)
    func addChunk(wallNanos: UInt64, tokens chunkTokens: Int) {
        lock.lock()
        chunkWallNanos &+= wallNanos
        chunks += 1
        tokens += chunkTokens
        lock.unlock()
    }

    var recordedStages: Set<PrefillStage> {
        lock.lock(); defer { lock.unlock() }
        return Set(PrefillStage.allCases.filter { gpuCounts[$0.rawValue] > 0 })
    }

    var chunkCount: Int {
        lock.lock(); defer { lock.unlock() }
        return chunks
    }

    var tokenCount: Int {
        lock.lock(); defer { lock.unlock() }
        return tokens
    }

    public var isEmpty: Bool {
        lock.lock(); defer { lock.unlock() }
        return chunks == 0
    }

    /// The report printed after a CLI run with the toggle on.
    public func report() -> String {
        lock.lock(); defer { lock.unlock() }
        let fmt = { (seconds: Double) in String(format: "%9.1f ms", seconds * 1000) }
        let wall = Double(chunkWallNanos) / 1e9
        var gpuTotal = 0.0
        var lines = "[prefill stages] chunks=\(chunks) tokens=\(tokens) "
        lines += "chunk wall=" + String(format: "%.2f", wall) + " s"
        lines += " (diagnostic run: each stage boundary adds a commit+wait)\n"
        for stage in PrefillStage.allCases where gpuCounts[stage.rawValue] > 0 {
            let seconds = gpuSeconds[stage.rawValue]
            gpuTotal += seconds
            let share = wall > 0 ? 100 * seconds / wall : 0
            lines += "  " + stage.label.padding(toLength: 30, withPad: " ", startingAt: 0)
            lines += fmt(seconds) + String(format: "  %5.1f%% of wall", share)
            lines += "  n=\(gpuCounts[stage.rawValue])\n"
        }
        lines += "  " + "GPU sum".padding(toLength: 30, withPad: " ", startingAt: 0) + fmt(gpuTotal)
        lines += "  (* rows overlap each other, so sums can exceed wall)\n"
        lines += "  " + "routed fetch wait (CPU)".padding(toLength: 30, withPad: " ", startingAt: 0)
        lines += fmt(Double(routedFetchWaitNanos) / 1e9) + "  n=\(routedFetchCount)\n"
        lines += "  " + "router readback→tiles (CPU)".padding(toLength: 30, withPad: " ", startingAt: 0)
        lines += fmt(Double(routedPlanNanos) / 1e9) + "\n"
        lines += "  " + "layer loop wall".padding(toLength: 30, withPad: " ", startingAt: 0)
        lines += fmt(Double(layerWallNanos) / 1e9) + "\n"
        return lines
    }
}
