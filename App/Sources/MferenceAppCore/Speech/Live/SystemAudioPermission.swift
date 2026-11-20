import Foundation
import Synchronization

/// The live call-audio permission (`AppPermission.systemAudio`), for
/// `LivePermissionBroker` (Notes spec 5.2). macOS has no public call that
/// reads it, and asks only when a tap first starts, so the outcome is kept
/// in the app's settings: without it the reason and a test tap would run
/// again at every launch. A tap that fails later (the person revoked it in
/// System Settings) records a denial, which reads as `.needsSystemSettings`.
public final class SystemAudioPermission: @unchecked Sendable {
    public static let key = "speech.systemAudioAccess"

    private let store: any SettingsStore
    private let lock = NSLock()
    private let probe: @Sendable () async throws -> Void

    /// - Parameter probe: starts and stops a test tap, throwing when the
    ///   system refuses it; the live one asks the system the first time.
    public init(store: any SettingsStore = UserDefaults.standard,
                probe: @escaping @Sendable () async throws -> Void = SystemAudioPermission.testTap) {
        self.store = store
        self.probe = probe
    }

    public var state: PermissionState {
        switch lock.withLock({ store.object(forKey: Self.key) as? String }) {
        case "granted": .granted
        case "denied": .needsSystemSettings
        default: .notDetermined
        }
    }

    /// Remembers the outcome of a tap.
    public func record(granted: Bool) {
        lock.withLock { store.set(granted ? "granted" : "denied", forKey: Self.key) }
    }

    /// Runs the test tap (the system prompt the first time) and remembers what happened.
    public func request() async throws {
        do {
            try await probe()
            record(granted: true)
        } catch {
            record(granted: false)
            throw error
        }
    }

    /// What `LivePermissionBroker` reads and asks through.
    public var access: SystemAudioAccess {
        SystemAudioAccess(state: { [self] in state }, request: { [self] in try await request() })
    }

    /// Taps every app but this one for a moment.
    public static let testTap: @Sendable () async throws -> Void = {
        let source = ProcessTapSource(target: .allApps)
        let stream = try await source.start()
        try await Task.sleep(for: .milliseconds(300))
        await source.stop()
        for try await _ in stream {}
    }
}

extension SpeechService {
    /// The app's speech service: the FluidAudio stages loaded from the
    /// verified install, the real microphone and process tap, recordings
    /// under Application Support, sleep and wake from `NSWorkspace`.
    /// Builds nothing heavy: models load on first use.
    public static func live(isOffline: @escaping @Sendable () async -> Bool,
                            systemAudio: SystemAudioPermission) -> SpeechService {
        SpeechService(Dependencies(
            installer: SpeechModelInstaller(policy: .init(isOffline: isOffline)),
            loader: FluidModelLoader(),
            sources: LiveAudioSourceFactory(),
            recordings: RecordingStore(),
            power: WorkspacePowerEvents(),
            onCallAudioFailure: { error in
                if error == .permissionDenied(.systemAudio) { systemAudio.record(granted: false) }
            }))
    }
}
