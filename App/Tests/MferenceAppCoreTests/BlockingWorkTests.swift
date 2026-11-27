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

    @Test func runsAtTheCallersPriority() {
        #expect(BlockingWork.qos(for: .userInitiated) == .userInitiated)
        #expect(BlockingWork.qos(for: .medium) == .default)
        #expect(BlockingWork.qos(for: .utility) == .utility)
        #expect(BlockingWork.qos(for: .background) == .background)
    }
}
