import Foundation

extension Native {
    /// The scheduler's dedicated generation thread (`userInteractive`, never
    /// the Swift cooperative pool): runs submitted jobs in order, each inside
    /// its own autorelease pool, since a long-lived thread never drains its
    /// implicit one.
    final class SchedulerGenerationThread: @unchecked Sendable {
        private final class State: @unchecked Sendable {
            let condition = NSCondition()
            var work: [@Sendable () -> Void] = []
            var stopped = false
        }

        private let state = State()

        init(name: String) {
            let state = self.state
            let thread = Thread {
                while true {
                    state.condition.lock()
                    while state.work.isEmpty && !state.stopped { state.condition.wait() }
                    if state.work.isEmpty && state.stopped {
                        state.condition.unlock()
                        return
                    }
                    let next = state.work.removeFirst()
                    state.condition.unlock()
                    autoreleasepool { next() }
                }
            }
            thread.name = name
            thread.qualityOfService = .userInteractive
            thread.start()
        }

        func submit(_ work: @escaping @Sendable () -> Void) {
            state.condition.lock()
            state.work.append(work)
            state.condition.signal()
            state.condition.unlock()
        }

        deinit {
            state.condition.lock()
            state.stopped = true
            state.condition.signal()
            state.condition.unlock()
        }
    }
}
