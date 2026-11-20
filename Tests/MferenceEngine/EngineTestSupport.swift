import Foundation
import Synchronization
import Testing
import MferenceEngine

/// Thread-safe append-only list for callbacks.
final class Collector<Element: Sendable>: Sendable {
    private let items = Mutex<[Element]>([])

    func append(_ item: Element) { items.withLock { $0.append(item) } }
    var all: [Element] { items.withLock { $0 } }
}

/// An idle-timer clock the test advances by hand.
final class ManualSleeper: Sendable {
    private let latches = Mutex<[FakeEngineLatch]>([])

    var requested: Int { latches.withLock { $0.count } }

    func sleep(_ duration: Duration) async throws {
        let latch = FakeEngineLatch()
        latches.withLock { $0.append(latch) }
        try await latch.wait()
    }

    /// Ends every sleep requested so far; cancelled ones already ended.
    func fireAll() {
        for latch in latches.withLock({ $0 }) { latch.open() }
    }
}

struct InjectedLoadError: Error, Equatable {}

/// Polls `condition` until it holds, recording an issue after `timeout`.
func eventually(_ what: String,
                timeout: Duration = .seconds(10),
                _ condition: @Sendable () async -> Bool) async throws {
    let deadline = ContinuousClock.now + timeout
    while !(await condition()) {
        if ContinuousClock.now >= deadline {
            Issue.record("timed out waiting for: \(what)")
            return
        }
        try await Task.sleep(for: .milliseconds(1))
    }
}

extension Array where Element == FakeEngineJournalEntry {
    func index(of entry: FakeEngineJournalEntry) -> Int? { firstIndex(of: entry) }

    var generationsStarted: [String] {
        compactMap { if case .generationStarted(let id) = $0 { id } else { nil } }
    }

    func count(of entry: FakeEngineJournalEntry) -> Int { filter { $0 == entry }.count }
}
