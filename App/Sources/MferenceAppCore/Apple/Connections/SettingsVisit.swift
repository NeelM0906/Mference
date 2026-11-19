import Foundation

/// Whether a Full Disk Access request took the person to System Settings.
/// The broker's answer cannot tell: Full Disk Access has no prompt, and
/// "Not now" under the reason returns the same state as a visit. Mference
/// leaving the foreground while the request runs, or within a few seconds
/// after it, counts as the visit (System Settings came to the front).
public struct SettingsVisit: Sendable {
    /// How long after the request a switch away still counts.
    static let window: Duration = .seconds(5)

    private var started = false
    private var ended: ContinuousClock.Instant?
    public private(set) var happened = false

    public init() {}

    mutating func begin() {
        started = true
        ended = nil
    }

    mutating func end(at now: ContinuousClock.Instant = .now) {
        if started { ended = now }
    }

    /// The app resigned active.
    mutating func appLeft(at now: ContinuousClock.Instant = .now) {
        guard started else { return }
        started = false
        if let ended, now - ended > Self.window { return }
        happened = true
    }
}
