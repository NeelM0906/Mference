import Foundation

/// A serial executor that runs its jobs on one dedicated thread at a fixed
/// quality of service. `LlamaModel` uses it as its actor executor, so every
/// llama.cpp call a model makes (prefill, decode, MTP steps, and the Metal
/// command encoding they do on the calling thread) runs at that QoS instead of
/// the priority a request's task inherited, which can leave it on efficiency
/// cores.
///
/// `MFERENCE_LLAMA_THREAD_QOS` (`userInteractive`, the default,
/// `userInitiated`, `default` or `utility`) is a diagnostic override for
/// measuring the effect; it is not a supported control.
final class LlamaThreadExecutor: SerialExecutor, @unchecked Sendable {
    // Guarded by `condition`.
    private var jobs: [UnownedJob] = []
    private var stopped = false
    private let condition = NSCondition()

    /// Starts the thread. It keeps the executor alive until `shutdown`.
    init(name: String, qualityOfService: QualityOfService = LlamaThreadExecutor.configuredQoS) {
        let thread = Thread { self.drain() }
        thread.name = name
        thread.qualityOfService = qualityOfService
        thread.stackSize = 8 << 20
        thread.start()
    }

    /// The QoS `MFERENCE_LLAMA_THREAD_QOS` names, else `.userInteractive`.
    static var configuredQoS: QualityOfService {
        switch ProcessInfo.processInfo.environment["MFERENCE_LLAMA_THREAD_QOS"] {
        case "userInitiated": .userInitiated
        case "default": .default
        case "utility": .utility
        default: .userInteractive
        }
    }

    func enqueue(_ job: consuming ExecutorJob) {
        let job = UnownedJob(job)
        condition.lock()
        jobs.append(job)
        condition.signal()
        condition.unlock()
    }

    func asUnownedSerialExecutor() -> UnownedSerialExecutor {
        UnownedSerialExecutor(ordinary: self)
    }

    /// Lets the thread exit once its queue is empty, releasing the executor.
    func shutdown() {
        condition.lock()
        stopped = true
        condition.signal()
        condition.unlock()
    }

    private func drain() {
        while true {
            condition.lock()
            while jobs.isEmpty && !stopped { condition.wait() }
            let job = jobs.isEmpty ? nil : jobs.removeFirst()
            condition.unlock()
            guard let job else { return }
            job.runSynchronously(on: asUnownedSerialExecutor())
        }
    }
}
