import Foundation
import Metal

/// Keeps the GPU busy with a one-simdgroup spin kernel on a second queue while
/// the runtime is working. Decode submits short command buffers separated by
/// CPU round trips; after ~200 us of idle the GPU runs the next buffer about 4x
/// slower (M4 Pro, 24 GiB, 27 Oct: a 16-dispatch GEMV chain took 52 us
/// back-to-back, ~215 us with 200-1000 us gaps, ~50 us with this filler
/// running, and the filler cost back-to-back work nothing). Runners call
/// `touch()` once per decode token; prefill keeps the GPU busy on its own.
/// The filler stops 500 ms after the last touch and its thread blocks until
/// the next one.
///
/// The filler is off on Apple7-generation GPUs, the M1 series among Macs (see
/// `policy(supportsFamily:)`), where it slows decode instead of helping.
public final class GPUKeepWarm: @unchecked Sendable {
    /// Whether this process runs the filler, and how long each buffer spins.
    public enum Policy: Sendable, Equatable {
        case spin(iterations: UInt32)
        case disabled
    }

    /// ~1.2 ms per filler buffer on an M4 Pro; two stay in flight.
    static let spinIterations: UInt32 = 100_000

    /// Picks the filler policy from the device's Metal GPU families.
    ///
    /// Off for devices that support `.apple7` but not `.apple8`: the M1
    /// series among Macs (Apple's Metal Feature Set Tables, "Metal GPUs":
    /// M1-series is Apple7, M2-series Apple8, M3- and M4-series Apple9,
    /// M5-series Apple10). The check names the GPU generation, not a chip, so
    /// a virtual GPU that reports Apple7 without Apple8 is treated alike. On an
    /// M1 Max (32 GB, macOS 27.2, Qwen 3.6, 14 Nov 2026) each 100k-iteration
    /// buffer took ~2.9 ms of GPU time, decode buffers queued behind it, and
    /// decode ran at 2.98-3.01 tok/s with the filler against 34.4-35.2 tok/s
    /// without it, byte-identical output. A 10k-iteration filler (~0.3 ms)
    /// still gave only 6.0 tok/s, so a shorter spin does not fix it there.
    ///
    /// Every other device keeps the filler, including the M4 Pro (Apple9) it
    /// was measured on. M2 and M3 are unmeasured and keep today's behaviour.
    static func policy(supportsFamily: (MTLGPUFamily) -> Bool) -> Policy {
        if supportsFamily(.apple7) && !supportsFamily(.apple8) { return .disabled }
        return .spin(iterations: spinIterations)
    }

    public let policy: Policy
    private let spinner: Spinner?

    /// Whether a filler thread and queue exist; false exactly under `.disabled`.
    var isSpinning: Bool { spinner != nil }

    private static let sharedLock = NSLock()
    nonisolated(unsafe) private static var sharedInstance: GPUKeepWarm?

    /// One filler per process: every context drives the same GPU, and the
    /// filler thread keeps its instance alive for the process lifetime.
    static func shared(device: MTLDevice,
                       pipeline: () throws -> MTLComputePipelineState) throws -> GPUKeepWarm {
        sharedLock.lock()
        defer { sharedLock.unlock() }
        if let sharedInstance { return sharedInstance }
        let policy = Self.policy(supportsFamily: { device.supportsFamily($0) })
        let instance = try GPUKeepWarm(device: device, policy: policy, pipeline: pipeline)
        sharedInstance = instance
        return instance
    }

    private init(device: MTLDevice, policy: Policy,
                 pipeline: () throws -> MTLComputePipelineState) throws {
        self.policy = policy
        switch policy {
        case .disabled:
            spinner = nil
        case .spin(let iterations):
            spinner = try Spinner(device: device, pipeline: try pipeline(), iterations: iterations)
        }
    }

    /// Marks the runtime busy; cheap enough to call once per token. A no-op
    /// when the policy is `.disabled`.
    public func touch() {
        spinner?.touch()
    }
}

/// The filler thread and its command queue; exists only under `.spin`.
private final class Spinner: @unchecked Sendable {
    private static let idleTimeoutNanos: UInt64 = 500_000_000

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let scratch: MTLBuffer
    private let iterations: UInt32
    private let lock = NSLock()
    private var lastTouch: UInt64 = 0
    private var sleeping = true
    private let wake = DispatchSemaphore(value: 0)
    private let inFlight = DispatchSemaphore(value: 2)

    init(device: MTLDevice, pipeline: MTLComputePipelineState, iterations: UInt32) throws {
        guard let queue = device.makeCommandQueue() else { throw MetalError.noQueue }
        guard let scratch = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride,
                                              options: .storageModePrivate)
        else { throw MetalError.noDevice }
        self.queue = queue
        self.pipeline = pipeline
        self.scratch = scratch
        self.iterations = iterations
        let thread = Thread { [self] in self.run() }
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func touch() {
        lock.lock()
        lastTouch = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let wasSleeping = sleeping
        sleeping = false
        lock.unlock()
        if wasSleeping { wake.signal() }
    }

    private func run() {
        var iterations = self.iterations
        while true {
            lock.lock()
            let idle = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- lastTouch > Self.idleTimeoutNanos
            if idle { sleeping = true }
            lock.unlock()
            if idle {
                wake.wait()
                continue
            }
            inFlight.wait()
            guard let cb = queue.makeCommandBuffer(),
                  let encoder = cb.makeComputeCommandEncoder() else {
                inFlight.signal()
                continue
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(scratch, offset: 0, index: 0)
            encoder.setBytes(&iterations, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
            cb.addCompletedHandler { [inFlight] _ in inFlight.signal() }
            cb.commit()
        }
    }
}
