import Foundation
import MferenceEngine
import Synchronization

/// The idle-unload delay the engine host waits for, changeable while the app
/// runs (Settings > Models, decision L13).
///
/// `EngineHost` fixes its `idleUnloadAfter` when it is built and keeps the
/// race-free part (the epoch, the gate, "nothing is running") to itself; it
/// calls its sleeper each time the model goes idle. This control is that
/// sleeper: it ignores the host's duration and waits for the current
/// setting instead. A change while a wait is running restarts the wait with
/// the new delay, counted from the change. `nil` keeps the model until a
/// swap, an explicit unload or quit.
public final class IdleUnloadControl: Sendable {
    public typealias Sleep = @Sendable (Duration) async throws -> Void

    private struct State {
        var delay: Duration?
        var generation: UInt64 = 0
        var nextWaiterID: UInt64 = 0
        var waiters: [UInt64: CheckedContinuation<Void, Never>] = [:]
    }

    private let state: Mutex<State>
    private let sleep: Sleep

    /// - Parameters:
    ///   - delay: nil never unloads.
    ///   - sleep: the clock; tests inject a controllable one. It must throw
    ///     `CancellationError` when its task is cancelled.
    public init(delay: Duration? = EngineService.defaultIdleUnload,
                sleep: @escaping Sleep = { try await Task.sleep(for: $0) }) {
        state = Mutex(State(delay: delay))
        self.sleep = sleep
    }

    public var delay: Duration? { state.withLock { $0.delay } }

    /// Applies at once, also to a wait already running.
    public func setDelay(_ delay: Duration?) {
        let waiting = state.withLock { s -> [CheckedContinuation<Void, Never>] in
            guard s.delay != delay else { return [] }
            s.delay = delay
            s.generation &+= 1
            defer { s.waiters.removeAll() }
            return Array(s.waiters.values)
        }
        for continuation in waiting { continuation.resume() }
    }

    /// The sleeper to hand to `EngineHost(sleep:)`. Build the host with any
    /// non-nil `idleUnloadAfter` so it calls it.
    public var sleeper: EngineHost.Sleeper {
        { [self] _ in try await self.waitForIdleDelay() }
    }

    /// Returns once the current delay has passed with no change in between;
    /// throws `CancellationError` when cancelled (the host then keeps the
    /// model).
    func waitForIdleDelay() async throws {
        while true {
            try Task.checkCancellation()
            let (delay, generation) = state.withLock { ($0.delay, $0.generation) }
            let elapsed = try await withThrowingTaskGroup(of: Bool.self) { group in
                if let delay {
                    group.addTask { [sleep] in
                        try await sleep(delay)
                        return true
                    }
                }
                group.addTask { [self] in
                    await self.waitForChange(after: generation)
                    return false
                }
                defer { group.cancelAll() }
                return try await group.next() ?? false
            }
            if elapsed { return }
        }
    }

    /// Returns when the setting changes after `generation`, or when cancelled.
    private func waitForChange(after generation: UInt64) async {
        let id = state.withLock { s -> UInt64 in
            defer { s.nextWaiterID &+= 1 }
            return s.nextWaiterID
        }
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                let resumeNow = state.withLock { s -> Bool in
                    if s.generation != generation || Task.isCancelled { return true }
                    s.waiters[id] = continuation
                    return false
                }
                if resumeNow { continuation.resume() }
            }
        } onCancel: {
            let continuation = state.withLock { $0.waiters.removeValue(forKey: id) }
            continuation?.resume()
        }
    }
}
