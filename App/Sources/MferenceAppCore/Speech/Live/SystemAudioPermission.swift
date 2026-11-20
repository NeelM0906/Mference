import Foundation

/// The live call-audio permission (`AppPermission.systemAudio`) for
/// `LivePermissionBroker` (Notes spec 5.2, L6).
///
/// macOS has no public call that reads it and asks only when a tap first
/// starts; a denied tap does not fail, it delivers digital silence. Silence
/// is never read as a denial: an in-person meeting, or a call app holding
/// idle output, is silent too. So the stored value only moves forward, and
/// only on a definitive signal:
///
/// - nothing stored: never asked. Reads `.notDetermined`.
/// - `asked`: the test tap ran (the system prompt, the first time). Reads
///   `.granted`, so the reason and the test tap don't run again and each
///   meeting taps; `hasHeardCallAudio` stays false until a tap hears audio.
/// - `granted`: a tap heard other apps at least once.
///
/// Core Audio errors are failures and record nothing. A `denied` left by an
/// earlier build reads as `asked`.
public final class SystemAudioPermission: @unchecked Sendable {
    public static let key = "speech.systemAudioAccess"

    private let store: any SettingsStore
    private let lock = NSLock()
    private let probe: @Sendable () async throws -> Bool

    /// - Parameter probe: runs a short test tap (the system prompt the first
    ///   time) and returns whether it heard other apps.
    init(store: any SettingsStore = UserDefaults.standard,
         probe: @escaping @Sendable () async throws -> Bool = SystemAudioPermission.testTap) {
        self.store = store
        self.probe = probe
    }

    public var state: PermissionState { stored == nil ? .notDetermined : .granted }

    /// Whether a tap has heard other apps at least once: until then call
    /// audio has been asked for but is not known to work.
    public var hasHeardCallAudio: Bool { stored == "granted" }

    var stored: String? { lock.withLock { store.object(forKey: Self.key) as? String } }

    private func store(_ value: String) { lock.withLock { store.set(value, forKey: Self.key) } }

    /// A meeting's tap heard other apps.
    func heardCallAudio() {
        if stored != "granted" { store("granted") }
    }

    /// Runs the test tap and keeps what it showed. Throws on a Core Audio
    /// failure (nothing is recorded then).
    public func request() async throws {
        if try await probe() {
            store("granted")
        } else if stored != "granted" {
            store("asked")
        }
    }

    /// What `LivePermissionBroker` reads and asks through.
    public var access: SystemAudioAccess {
        SystemAudioAccess(state: { [self] in state }, request: { [self] in try await request() })
    }

    /// Taps every app but this one for up to 3 s, or until it hears one.
    static let testTap: @Sendable () async throws -> Bool = {
        let heard = HeardBox()
        let source = ProcessTapSource(target: .allApps, onHeard: { heard.set() })
        let stream = try await source.start()
        let deadline = ContinuousClock.now + .seconds(3)
        while !heard.value, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(100))
        }
        await source.stop()
        do { for try await _ in stream {} } catch {}
        return heard.value
    }

    final class HeardBox: @unchecked Sendable {
        private let lock = NSLock()
        private var heard = false
        var value: Bool { lock.withLock { heard } }
        func set() { lock.withLock { heard = true } }
    }
}

extension SpeechService {
    /// The app's speech service: the FluidAudio stages loaded from the
    /// verified install, the real microphone and process tap, recordings
    /// under Application Support, sleep and wake from `NSWorkspace`.
    /// Builds nothing heavy: models load on first use. Each meeting taps
    /// call audio while `systemAudio` reads not determined or granted (never
    /// on a stored denial), and a tap that hears other apps is recorded there.
    public static func live(isOffline: @escaping @Sendable () async -> Bool,
                            systemAudio: SystemAudioPermission) -> SpeechService {
        SpeechService(Dependencies(
            installer: SpeechModelInstaller(policy: .init(isOffline: isOffline)),
            loader: FluidModelLoader(),
            sources: LiveAudioSourceFactory(onCallAudioHeard: { systemAudio.heardCallAudio() }),
            recordings: RecordingStore(),
            power: WorkspacePowerEvents(),
            callAudioAllowed: { [.notDetermined, .granted].contains(systemAudio.state) }))
    }
}
