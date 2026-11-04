import Foundation
import Metal

/// Keeps the GPU busy with a one-simdgroup spin kernel on a second queue while
/// the runtime is working. Decode submits short command buffers separated by
/// CPU round trips; after ~200 us of idle the GPU runs the next buffer about 4x
/// slower (M4 Pro, 24 GiB, 27 Oct: a 16-dispatch GEMV chain took 52 us
/// back-to-back, ~215 us with 200-1000 us gaps, ~50 us with this filler
/// running, and the filler cost back-to-back work nothing). Runners call
/// `touch()` once per decode token; prefill keeps the GPU busy on its own.
/// The filler stops `idleTimeoutNanos` after the last touch and its thread
/// blocks until the next one.
public final class GPUKeepWarm: @unchecked Sendable {
    private static let idleTimeoutNanos: UInt64 = 500_000_000
    /// ~1.2 ms per filler buffer on an M4 Pro; two stay in flight.
    private static let spinIterations: UInt32 = 100_000

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let scratch: MTLBuffer
    private let lock = NSLock()
    private var lastTouch: UInt64 = 0
    private var sleeping = true
    private let wake = DispatchSemaphore(value: 0)
    private let inFlight = DispatchSemaphore(value: 2)

    private static let sharedLock = NSLock()
    nonisolated(unsafe) private static var sharedInstance: GPUKeepWarm?

    /// One filler per process: every context drives the same GPU, and the
    /// filler thread keeps its instance alive for the process lifetime.
    static func shared(device: MTLDevice,
                       pipeline: () throws -> MTLComputePipelineState) throws -> GPUKeepWarm {
        sharedLock.lock()
        defer { sharedLock.unlock() }
        if let sharedInstance { return sharedInstance }
        let instance = try GPUKeepWarm(device: device, pipeline: try pipeline())
        sharedInstance = instance
        return instance
    }

    private init(device: MTLDevice, pipeline: MTLComputePipelineState) throws {
        guard let queue = device.makeCommandQueue() else { throw MetalError.noQueue }
        guard let scratch = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride,
                                              options: .storageModePrivate)
        else { throw MetalError.noDevice }
        self.queue = queue
        self.pipeline = pipeline
        self.scratch = scratch
        let thread = Thread { [self] in self.run() }
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    /// Marks the runtime busy; cheap enough to call once per token.
    public func touch() {
        lock.lock()
        lastTouch = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let wasSleeping = sleeping
        sleeping = false
        lock.unlock()
        if wasSleeping { wake.signal() }
    }

    private func run() {
        var iterations = Self.spinIterations
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
