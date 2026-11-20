import Foundation
import MferenceAppCore
import Observation

/// The voice screens' long-lived models (team N3): the setup card, Settings
/// › Speech, the remembered voice choices and dictation. The app builds one at launch
/// from `AppServices` and injects it with `.environment(voice)`; snapshots
/// and tests build one over `FakeSpeechService`. A screen without it shows
/// its honest "not available" state.
@MainActor
@Observable
public final class VoiceModels {
    public let preferences: SpeechPreferences
    public let setup: SpeechSetupModel
    public let settings: SpeechSettingsModel
    /// `AppServices.dictation`: the one consumer of dictation events.
    public let dictation: DictationController
    /// The note editor that has focus, set by the editor (team N1): ⇧⌘D
    /// dictates into it. Weak, so a closed editor never receives text.
    @ObservationIgnored public weak var focusedDictationTarget: (any DictationInsertionTarget)?
    /// The record popover (board 3).
    @ObservationIgnored public private(set) var recordMeetingModel: RecordMeetingModel!
    public var showsRecordPopover = false

    // Hooks the app wires to the other teams' models.

    /// Starts a meeting and returns its note (N4's `MeetingSession.start`).
    @ObservationIgnored public var startMeeting: (@MainActor (RecordMeetingModel.Request) async throws -> UUID)?
    /// Pause, resume and stop for the menu bar and quit (N4's `MeetingSession`).
    @ObservationIgnored public var recordingControls: (any LiveRecordingControls)?
    /// Shows a note in Notes (N1's navigation): the timer chip, Open note.
    @ObservationIgnored public var openNote: (@MainActor (UUID) -> Void)?

    public init(speech: any SpeechServicing, permissions: any PermissionBroker, preferences: SpeechPreferences,
                dictation: DictationController? = nil,
                isOffline: @escaping @MainActor () -> Bool) {
        self.preferences = preferences
        self.dictation = dictation ?? DictationController(speech: speech, permissions: permissions,
                                                          preferences: preferences)
        setup = SpeechSetupModel(speech: speech, isOffline: isOffline)
        settings = SpeechSettingsModel(setup: setup, preferences: preferences, speech: speech,
                                       permissions: permissions)
        recordMeetingModel = RecordMeetingModel(speech: speech, permissions: permissions, preferences: preferences) {
            [weak self] request in
            guard let self, let startMeeting = self.startMeeting else {
                throw MeetingsUnavailable()
            }
            let noteID = try await startMeeting(request)
            self.openNote?(noteID)
        }
    }

    /// The app's: over `services.speech` and `services.dictation`, Offline
    /// mode from `services.settings`.
    public static func live(services: AppServices) -> VoiceModels {
        let settings = services.settings
        let voice = VoiceModels(speech: services.speech, permissions: services.permissions,
                                preferences: services.speechPreferences, dictation: services.dictation,
                                isOffline: { settings.offlineMode })
        voice.setup.start()
        return voice
    }

    // MARK: Dictation (spec 4.2)

    /// The Dictate button and ⇧⌘D: stops dictating into `target`, or starts
    /// there (through the setup card on first use).
    public func toggleDictation(in target: any DictationInsertionTarget) {
        if dictation.isDictating(into: target.noteID) {
            Task { await dictation.stop() }
            return
        }
        setup.require(.dictate) { [dictation, weak target] in
            guard let target else { return }
            Task { await dictation.start(in: target) }
        }
    }

    /// ⇧⌘D from the menu: the focused editor, or stop when there is none.
    public func toggleDictation() {
        if let target = focusedDictationTarget {
            toggleDictation(in: target)
        } else if dictation.isActive {
            Task { await dictation.stop() }
        }
    }

    // MARK: Meetings (spec 4.3, D7)

    /// The meeting that holds the microphone, for the timer chip, the menu
    /// bar and the quit confirmation.
    public var liveMeeting: LiveSessionStatus? {
        setup.status.liveSession.flatMap { $0.kind == .meeting ? $0 : nil }
    }

    /// Record meeting (⇧⌘R): the popover, through the setup card on first use.
    public func recordMeeting() {
        guard liveMeeting == nil else {
            openLiveMeeting()
            return
        }
        setup.require(.recordMeeting) { [weak self] in self?.showsRecordPopover = true }
    }

    /// The timer chip and the menu bar's Open: the meeting note.
    public func openLiveMeeting() {
        if let noteID = liveMeeting?.noteID { openNote?(noteID) }
    }

    /// Quit and the menu bar's Stop: stops the meeting and waits until its
    /// transcript is saved; then any dictation (its last phrase lands).
    public func stopLiveSessions() async {
        if liveMeeting != nil { await recordingControls?.stopRecording() }
        await dictation.stop()
    }

    /// Switching notes or leaving Notes: the last phrase lands in the note
    /// it belongs to, then dictation stops (N1 calls this).
    public func stopDictation() async {
        await dictation.stop()
    }
}

/// Start was pressed in a build where meetings are not wired yet.
struct MeetingsUnavailable: Error {}
