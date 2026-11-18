import Foundation
import Testing
@testable import MferenceEngine

/// An actor on a `LlamaThreadExecutor`, as `LlamaModel` is.
private actor ExecutorProbe {
    private let executor: LlamaThreadExecutor
    private var order: [Int] = []

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        executor.asUnownedSerialExecutor()
    }

    init(qualityOfService: QualityOfService) {
        executor = LlamaThreadExecutor(name: "mference.llama.test", qualityOfService: qualityOfService)
    }

    deinit { executor.shutdown() }

    func threadQoS() -> qos_class_t { qos_class_self() }

    func threadName() -> String {
        var name = [CChar](repeating: 0, count: 64)
        pthread_getname_np(pthread_self(), &name, name.count)
        return String(cString: name)
    }

    func record(_ value: Int) { order.append(value) }

    func recorded() -> [Int] { order }
}

@Suite("Llama thread executor")
struct LlamaThreadExecutorTests {
    @Test func runsActorWorkOnItsOwnThreadAtTheRequestedQoS() async {
        let probe = ExecutorProbe(qualityOfService: .userInteractive)
        #expect(await probe.threadName() == "mference.llama.test")
        #expect(await probe.threadQoS() == QOS_CLASS_USER_INTERACTIVE)

        let utility = ExecutorProbe(qualityOfService: .utility)
        #expect(await utility.threadQoS() == QOS_CLASS_UTILITY)
    }

    @Test func runsJobsOneAtATimeInSubmissionOrder() async {
        let probe = ExecutorProbe(qualityOfService: .userInteractive)
        for value in 0..<200 { await probe.record(value) }
        #expect(await probe.recorded() == Array(0..<200))

        // Concurrent callers are serialized: every value lands exactly once.
        await withTaskGroup(of: Void.self) { group in
            for value in 200..<400 { group.addTask { await probe.record(value) } }
        }
        #expect(await probe.recorded().sorted() == Array(0..<400))
    }

    @Test func defaultQoSIsUserInteractive() {
        // The environment override is unset in the test run.
        if ProcessInfo.processInfo.environment["MFERENCE_LLAMA_THREAD_QOS"] == nil {
            #expect(LlamaThreadExecutor.configuredQoS == .userInteractive)
        }
    }
}
