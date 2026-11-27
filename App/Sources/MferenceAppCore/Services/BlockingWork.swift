import Foundation

/// Runs synchronous, blocking work (Vision, the AVFoundation bridge in
/// `FindMedia`, file reads, catalog discovery) on a Dispatch thread, never on
/// a thread of Swift concurrency's cooperative pool.
///
/// The pool has one thread per core. Vision's text requests wait for work of
/// their own that runs on that pool, and `FindMedia.blocking` waits for a
/// detached task, so a pool thread blocked in either one holds a thread those
/// jobs need; once every pool thread is held, nothing finishes. The 3-core CI
/// runner reached that state with three parallel reads, and its app tests hung.
public enum BlockingWork {
    /// Runs `work` on a global Dispatch queue at the caller's priority and
    /// resumes with its result. The calling task is suspended, not blocked.
    public static func run<T: Sendable>(_ work: @escaping @Sendable () -> T) async -> T {
        let qos = qos(for: Task.currentPriority)
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: qos).async { continuation.resume(returning: work()) }
        }
    }

    static func qos(for priority: TaskPriority) -> DispatchQoS.QoSClass {
        if priority >= .userInitiated { return .userInitiated }
        if priority >= .medium { return .default }
        if priority >= .utility { return .utility }
        return .background
    }
}
