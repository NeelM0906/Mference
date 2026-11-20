import Foundation
import MferenceAppCore
import Observation

/// The record popover (board 3, Notes spec 4.3): the microphone, the call
/// audio source, Speaker labels, Keep the recording and Summary when you
/// stop, remembered as the Settings › Speech defaults; Start asks for the
/// microphone, then call audio (each only the first time), then starts the
/// meeting through `start` (N4's `MeetingSession`).
@MainActor
@Observable
public final class RecordMeetingModel {
    /// What Start hands to the meeting session (N4's `MeetingStartOptions`).
    public struct Request: Sendable, Hashable {
        public var microphoneID: String?
        /// The source the person chose, also when its permission was denied:
        /// the speech service then records the microphone only and says so
        /// (`MeetingEvent.callAudioUnavailable`), and the note says so.
        public var callAudio: CallAudioSource
        public var speakerLabels: Bool
        public var keepAudio: Bool
        public var summaryWhenYouStop: Bool
        public var language: SpeechLanguage
    }

    public let preferences: SpeechPreferences
    public private(set) var devices: [AudioInputDevice] = []
    public private(set) var apps: [RunningAudioApp] = []
    public private(set) var isStarting = false
    /// Why Start did not start (microphone denied, busy, low disk), as one line.
    public private(set) var errorMessage: String?
    /// What the meeting now recording started with (the recording bar shows
    /// these, not the defaults, which the person may change meanwhile).
    public private(set) var lastStarted: Request?

    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let permissions: any PermissionBroker
    @ObservationIgnored private let start: @MainActor (Request) async throws -> Void
    @ObservationIgnored private let speakerLabelsReady: @MainActor () -> Bool

    /// - Parameter start: starts the meeting (N4's `MeetingSession.start`)
    ///   and opens its note; throws a `SpeechError` when it cannot.
    /// - Parameter speakerLabelsReady: whether the speaker-labels download is
    ///   installed, read from an observed status so the switch updates.
    public init(speech: any SpeechServicing, permissions: any PermissionBroker, preferences: SpeechPreferences,
                speakerLabelsReady: (@MainActor () -> Bool)? = nil,
                start: @escaping @MainActor (Request) async throws -> Void) {
        self.speakerLabelsReady = speakerLabelsReady ?? { speech.status.speakerLabels.isReady }
        self.speech = speech
        self.permissions = permissions
        self.preferences = preferences
        self.start = start
    }

    /// Reads devices and running apps when the popover opens.
    public func load() async {
        devices = await speech.inputDevices()
        apps = await speech.runningAudioApps()
        errorMessage = nil
    }

    public var microphoneName: String {
        SpeechSettingsModel.microphoneName(id: preferences.microphoneID, devices: devices)
    }

    public var callAudioName: String { SpeechSettingsModel.callAudioName(preferences.callAudio) }

    /// Speaker labels needs its download; without it the switch shows off
    /// and the meeting labels "You" and "Others".
    public var speakerLabelsAvailable: Bool { speakerLabelsReady() }

    /// Start recording. Returns true when the meeting started (the popover closes).
    @discardableResult
    public func startRecording() async -> Bool {
        guard !isStarting else { return false }
        isStarting = true
        defer { isStarting = false }
        errorMessage = nil
        guard await permissions.request(.microphone) == .granted else {
            errorMessage = SpeechError.permissionDenied(.microphone).message
            return false
        }
        if preferences.callAudio != .off {
            // Asked once, at the first meeting (L6). A denial is not an
            // error: the meeting records the microphone only.
            _ = await permissions.request(.systemAudio)
        }
        let request = Request(microphoneID: preferences.microphoneID, callAudio: preferences.callAudio,
                              speakerLabels: preferences.speakerLabels && speakerLabelsAvailable,
                              keepAudio: preferences.keepAudio == .keep,
                              summaryWhenYouStop: preferences.writesMeetingSummary,
                              language: preferences.language)
        do {
            try await start(request)
            lastStarted = request
            return true
        } catch {
            errorMessage = (error as? SpeechError)?.message ?? "The recording didn't start. Try again."
            return false
        }
    }
}
