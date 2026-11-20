import Foundation
import Synchronization

/// The speech status, readable without waiting and observable by many
/// (`SpeechServicing.status`, `statusUpdates()`). Changes reach observers
/// in order.
final class SpeechStatusBox: Sendable {
    private struct State {
        var status: SpeechStatus
        var observers: [UUID: AsyncStream<SpeechStatus>.Continuation] = [:]
    }
    private let state: Mutex<State>

    init(_ status: SpeechStatus) { state = Mutex(State(status: status)) }

    var status: SpeechStatus { state.withLock { $0.status } }

    func updates() -> AsyncStream<SpeechStatus> {
        let (stream, continuation) = AsyncStream.makeStream(of: SpeechStatus.self, bufferingPolicy: .bufferingNewest(16))
        let id = UUID()
        state.withLock { s in
            s.observers[id] = continuation
            continuation.yield(s.status)
        }
        continuation.onTermination = { [weak self] _ in
            self?.state.withLock { _ = $0.observers.removeValue(forKey: id) }
        }
        return stream
    }

    /// Changes the status and tells observers inside the lock, so two
    /// changes never reach an observer swapped.
    func mutate(_ change: (inout SpeechStatus) -> Void) {
        state.withLock { s in
            let before = s.status
            change(&s.status)
            guard s.status != before else { return }
            for observer in s.observers.values { observer.yield(s.status) }
        }
    }

    func setPackage(_ package: SpeechPackageID, _ value: SpeechModelState) {
        mutate { status in
            switch package {
            case .speech: status.speech = value
            case .speakerLabels: status.speakerLabels = value
            }
        }
    }

    func updateJob(_ id: FileJob.ID, _ change: (inout FileJobStatus) -> Void) {
        mutate { status in
            if let index = status.fileQueue.firstIndex(where: { $0.id == id }) { change(&status.fileQueue[index]) }
        }
    }
}

/// The one recogniser, used in order: phrases from dictation, both meeting
/// tracks and files queue here, so Fluid Speech Pro runs one phrase at a
/// time (spec 6.3: "the one transcriber in time order").
actor TranscriptionLane {
    private var tail: Task<Void, Never>?

    func run<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) async throws -> T {
        let previous = tail
        let task = Task { () async throws -> T in
            await previous?.value
            return try await work()
        }
        tail = Task { _ = try? await task.value }
        return try await task.value
    }
}

/// System sleep and wake (spec 6.3: capture stops on sleep and resumes in
/// new files on wake).
public enum PowerEvent: Sendable, Hashable {
    case willSleep
    case didWake
}

public protocol PowerEventSource: Sendable {
    /// A new stream of sleep and wake events; ends when the listener goes away.
    func events() -> AsyncStream<PowerEvent>
}

/// Sleep and wake on demand, for tests.
public final class FakePowerEvents: PowerEventSource {
    private let continuations = Mutex<[UUID: AsyncStream<PowerEvent>.Continuation]>([:])

    public init() {}

    public func events() -> AsyncStream<PowerEvent> {
        let (stream, continuation) = AsyncStream.makeStream(of: PowerEvent.self)
        let id = UUID()
        continuations.withLock { $0[id] = continuation }
        continuation.onTermination = { [weak self] _ in
            self?.continuations.withLock { _ = $0.removeValue(forKey: id) }
        }
        return stream
    }

    public func send(_ event: PowerEvent) {
        for continuation in continuations.withLock({ Array($0.values) }) { continuation.yield(event) }
    }

    /// Listeners waiting for events.
    public var listenerCount: Int { continuations.withLock { $0.count } }
}

/// Measured speed for the file queue's time left (spec 6.3: ETA from
/// measured speed, not a guess).
struct SpeedMeter {
    private(set) var audioDone: TimeInterval = 0
    private(set) var busy: TimeInterval = 0

    mutating func record(audio: TimeInterval, took: TimeInterval) {
        audioDone += audio
        busy += max(0, took)
    }

    /// Seconds left for `remainingAudio`, once a few seconds were measured.
    func remaining(for remainingAudio: TimeInterval) -> TimeInterval? {
        guard audioDone >= 5, busy > 0 else { return nil }
        return max(0, remainingAudio) * busy / audioDone
    }
}

/// The most common language among phrases, for a transcript's label.
func dominantLanguage(_ codes: [String?]) -> String? {
    var counts: [String: Int] = [:]
    for code in codes.compactMap({ $0 }) { counts[code, default: 0] += 1 }
    return counts.max { $0.value < $1.value || ($0.value == $1.value && $0.key > $1.key) }?.key
}
