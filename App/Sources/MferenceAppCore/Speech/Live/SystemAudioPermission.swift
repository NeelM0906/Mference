import Foundation

/// The live call-audio permission (`AppPermission.systemAudio`) for
/// `LivePermissionBroker` (Notes spec 5.2, L6).
///
/// macOS has no public call that reads it, asks only when a tap first
/// starts, and a denied tap does not fail: it delivers digital silence. So
/// the answer is judged from a tap's audio (`TapAccessVerdict`) and kept in
/// the app's settings, so the reason and a test tap don't run again at
/// every launch:
///
/// - `granted`: a tap heard other apps.
/// - `denied`: a tap stayed digitally silent while the tapped apps were
///   playing. Reads as `.needsSystemSettings`; opening that pane
///   (`systemSettingsOpened()`) lets the next meeting try again.
/// - `asked`: the prompt ran but nothing was playing to tell. Reads as
///   granted; the next meeting's tap settles it.
///
/// Core Audio errors are failures, never a recorded denial.
public final class SystemAudioPermission: @unchecked Sendable {
    public static let key = "speech.systemAudioAccess"

    private let store: any SettingsStore
    private let lock = NSLock()
    private let probe: @Sendable () async throws -> TapAccessVerdict.Result

    /// - Parameter probe: runs a short test tap (the system prompt the first
    ///   time) and returns what its audio showed.
    init(store: any SettingsStore = UserDefaults.standard,
         probe: @escaping @Sendable () async throws -> TapAccessVerdict.Result = SystemAudioPermission.testTap) {
        self.store = store
        self.probe = probe
    }

    public var state: PermissionState {
        switch stored {
        case "granted", "asked": .granted
        case "denied": .needsSystemSettings
        default: .notDetermined
        }
    }

    var stored: String? { lock.withLock { store.object(forKey: Self.key) as? String } }

    private func store(_ value: String) { lock.withLock { store.set(value, forKey: Self.key) } }

    /// What a tap's audio showed (live meetings report here).
    func record(_ verdict: TapAccessVerdict.Result) {
        switch verdict {
        case .granted: store("granted")
        case .denied: store("denied")
        case .undecided: break
        }
    }

    /// The person opened Screen & System Audio Recording in System Settings:
    /// a recorded denial no longer stands; the next tap checks again.
    public func systemSettingsOpened() {
        if stored == "denied" { store("asked") }
    }

    /// Runs the test tap and keeps what it showed. Throws on a Core Audio
    /// failure (nothing is recorded then).
    public func request() async throws {
        switch try await probe() {
        case .granted: store("granted")
        case .denied: store("denied")
        case .undecided: store("asked")
        }
    }

    /// What `LivePermissionBroker` reads and asks through.
    public var access: SystemAudioAccess {
        SystemAudioAccess(state: { [self] in state }, request: { [self] in try await request() })
    }

    /// Taps every app but this one for up to 3 s, or until its audio shows
    /// an answer. A denial needs longer than that to show, so a short test
    /// tap answers granted (something was heard) or undecided.
    static let testTap: @Sendable () async throws -> TapAccessVerdict.Result = {
        let verdict = VerdictBox()
        let source = ProcessTapSource(target: .allApps, onVerdict: { verdict.set($0) })
        let stream = try await source.start()
        let deadline = ContinuousClock.now + .seconds(3)
        while verdict.value == .undecided, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(100))
        }
        await source.stop()
        do { for try await _ in stream {} } catch {}
        return verdict.value
    }

    final class VerdictBox: @unchecked Sendable {
        private let lock = NSLock()
        private var result = TapAccessVerdict.Result.undecided
        var value: TapAccessVerdict.Result { lock.withLock { result } }
        func set(_ value: TapAccessVerdict.Result) { lock.withLock { result = value } }
    }
}

extension SpeechService {
    /// The app's speech service: the FluidAudio stages loaded from the
    /// verified install, the real microphone and process tap, recordings
    /// under Application Support, sleep and wake from `NSWorkspace`.
    /// Builds nothing heavy: models load on first use. Call audio is tapped
    /// only while `systemAudio` reads granted, and what a tap's audio shows
    /// is recorded there.
    public static func live(isOffline: @escaping @Sendable () async -> Bool,
                            systemAudio: SystemAudioPermission) -> SpeechService {
        SpeechService(Dependencies(
            installer: SpeechModelInstaller(policy: .init(isOffline: isOffline)),
            loader: FluidModelLoader(),
            sources: LiveAudioSourceFactory(onCallAccess: { systemAudio.record($0) }),
            recordings: RecordingStore(),
            power: WorkspacePowerEvents(),
            callAudioAllowed: { systemAudio.state == .granted }))
    }
}
