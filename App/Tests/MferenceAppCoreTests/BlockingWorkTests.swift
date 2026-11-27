import Foundation
@testable import MferenceAppCore
import Synchronization
import Testing

@Suite struct BlockingWorkTests {
    /// More blocking jobs than cores all run at once. On the cooperative pool
    /// (one thread per core) the extra jobs could never start, so the first
    /// ones would wait out the deadline; that was the CI hang.
    @Test func moreBlockedJobsThanCoresAllStart() async {
        let jobs = ProcessInfo.processInfo.activeProcessorCount + 2
        let started = Atomic<Int>(0)
        let deadline = Date().addingTimeInterval(20)
        let allStarted = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<jobs {
                group.addTask {
                    await BlockingWork.run {
                        started.add(1, ordering: .sequentiallyConsistent)
                        // Blocks this thread, the way Vision and FindMedia.blocking do.
                        while started.load(ordering: .sequentiallyConsistent) < jobs, Date() < deadline {
                            Thread.sleep(forTimeInterval: 0.005)
                        }
                        return started.load(ordering: .sequentiallyConsistent) == jobs
                    }
                }
            }
            return await group.allSatisfy { $0 }
        }
        #expect(allStarted, "only \(started.load(ordering: .sequentiallyConsistent)) of \(jobs) jobs started")
    }

    /// A read that never finishes is reported as skipped once the extractor's
    /// timeout passes, instead of holding its caller until the read returns.
    @Test func aStuckReadIsSkippedAtTheTimeout() async {
        let release = DispatchSemaphore(value: 0)
        let stuck = Task.detached {
            await BlockingWork.run { () -> FindExtraction in
                _ = release.wait(timeout: .now() + 30)
                return .text([])
            }
        }
        let clock = ContinuousClock()
        let start = clock.now
        let result = await InProcessFindExtractor.result(of: stuck, within: .milliseconds(200))
        let waited = clock.now - start
        release.signal()
        guard case .skipped = result else {
            Issue.record("a stuck read was not skipped: \(result)")
            return
        }
        #expect(waited < .seconds(5), "waited \(waited) for a 200 ms timeout")
    }

    @Test func aQuickReadReturnsItsOwnResult() async {
        let quick = Task.detached { await BlockingWork.run { FindExtraction.text([]) } }
        #expect(await InProcessFindExtractor.result(of: quick, within: .seconds(30)) == .text([]))
    }

    @Test func aCancelledCallerStopsWaiting() async {
        let release = DispatchSemaphore(value: 0)
        let stuck = Task.detached {
            await BlockingWork.run { () -> FindExtraction in
                _ = release.wait(timeout: .now() + 30)
                return .text([])
            }
        }
        let clock = ContinuousClock()
        let start = clock.now
        let caller = Task { await InProcessFindExtractor.result(of: stuck, within: .seconds(30)) }
        caller.cancel()
        let result = await caller.value
        let waited = clock.now - start
        release.signal()
        #expect(result != .text([]), "a cancelled caller does not wait for the read")
        #expect(waited < .seconds(5), "waited \(waited) after cancelling")
    }

    @Test func runsAtTheCallersPriority() {
        #expect(BlockingWork.qos(for: .userInitiated) == .userInitiated)
        #expect(BlockingWork.qos(for: .medium) == .default)
        #expect(BlockingWork.qos(for: .utility) == .utility)
        #expect(BlockingWork.qos(for: .background) == .background)
    }
}
