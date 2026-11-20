import Foundation

/// How `LivePermissionBroker` reads and asks for call-audio access
/// (`AppPermission.systemAudio`, the Core Audio process-tap permission).
///
/// macOS offers no public call that reads this permission, and the system
/// asks only when a tap first starts. So the Speech service, the one place
/// that creates taps (`MferenceAppCore/Speech`), supplies the live value: it
/// starts and stops a tap to ask, and remembers the outcome. Tests and
/// previews use a fixed value; nothing here touches TCC.
public struct SystemAudioAccess: Sendable {
    /// The current state; never prompts.
    public var state: @Sendable () async -> PermissionState
    /// Triggers the system prompt; throws what the system reports.
    public var request: @Sendable () async throws -> Void

    public init(state: @escaping @Sendable () async -> PermissionState,
                request: @escaping @Sendable () async throws -> Void) {
        self.state = state
        self.request = request
    }

    public enum Failure: Error, Sendable, Equatable {
        /// No Speech service is wired in to create a tap.
        case unavailable
    }

    /// Before the Speech service is wired in: never asked, and asking
    /// fails, so the broker answers `.needsSystemSettings` and no meeting
    /// records call audio by accident.
    public static let unavailable = SystemAudioAccess(state: { .notDetermined },
                                                      request: { throw Failure.unavailable })

    /// A fixed answer, for tests and previews.
    public static func fixed(_ state: PermissionState) -> SystemAudioAccess {
        SystemAudioAccess(state: { state }, request: {})
    }
}
