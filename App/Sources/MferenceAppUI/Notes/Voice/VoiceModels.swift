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
    @ObservationIgnored private let speech: any SpeechServicing
    /// The note editor that has focus, set by the editor (team N1): ⇧⌘D
    /// dictates into it. Weak, so a closed editor never receives text.
    public var focusedDictationTarget: (any DictationInsertionTarget)? {
        get { focusedTarget }
        set {
            focusedTarget = newValue
            focusedDictationNoteID = newValue?.noteID
        }
    }
    @ObservationIgnored private weak var focusedTarget: (any DictationInsertionTarget)?
    /// Observed (the menu's Dictate re-evaluates when focus moves): the note
    /// of the editor that has focus.
    public private(set) var focusedDictationNoteID: UUID?
    /// The record popover (board 3).
    @ObservationIgnored public private(set) var recordMeetingModel: RecordMeetingModel!
    public var showsRecordPopover = false

    // Hooks the app wires to the other teams' models.

    /// Starts a meeting and returns its note (N4's `MeetingSession.start`).
    @ObservationIgnored public var startMeeting: (@MainActor (RecordMeetingModel.Request) async throws -> UUID)?
    /// Whether N4's `MeetingSession` is active (recording, paused or
    /// finishing), and whether it is finishing; set by `connect(meetings:)`.
    @ObservationIgnored public var meetingIsActive: (@MainActor () -> Bool)?
    @ObservationIgnored public var meetingIsFinishing: (@MainActor () -> Bool)?
    /// Pause, resume and stop for the menu bar and quit (N4's `MeetingSession`).
    @ObservationIgnored public var recordingControls: (any LiveRecordingControls)?
    /// The file queue the import sheet shows (N4's `FileTranscriptionQueue`
    /// through an adapter); nil until wired.
    @ObservationIgnored public var fileQueue: (any FileQueueModel)?
    /// The import sheet (board 6).
    public var showsImportSheet = false
    /// Shows a note in Notes (N1's navigation): the timer chip, Open note.
    @ObservationIgnored public var openNote: (@MainActor (UUID) -> Void)?

    public init(speech: any SpeechServicing, permissions: any PermissionBroker, preferences: SpeechPreferences,
                dictation: DictationController? = nil,
                isOffline: @escaping @MainActor () -> Bool) {
        self.preferences = preferences
        self.speech = speech
        self.dictation = dictation ?? DictationController(speech: speech, permissions: permissions,
                                                          preferences: preferences)
        setup = SpeechSetupModel(speech: speech, isOffline: isOffline)
        settings = SpeechSettingsModel(setup: setup, preferences: preferences, speech: speech,
                                       permissions: permissions)
        let setup = setup
        recordMeetingModel = RecordMeetingModel(speech: speech, permissions: permissions, preferences: preferences,
                                                speakerLabelsReady: { setup.status.speakerLabels.isReady }) {
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

    /// ⇧⌘D is enabled with a focused note editor, or to stop dictation.
    public var canToggleDictation: Bool {
        (focusedDictationNoteID != nil && focusedTarget != nil) || dictation.isActive
    }

    /// ⇧⌘D from the menu, keyed on the dictation itself: while dictating
    /// (into any note, even one not shown) it stops; otherwise it starts in
    /// the focused editor. The menu title says the same ("Stop Dictating").
    public func toggleDictation() {
        if dictation.isActive {
            Task { await dictation.stop() }
        } else if let target = focusedDictationTarget {
            toggleDictation(in: target)
        }
    }

    /// Esc in the note editor (N1 forwards `cancelOperation`): stops
    /// dictation, landing the last phrase. Returns whether it handled Esc.
    @discardableResult
    public func cancelDictation() -> Bool {
        guard dictation.isActive else { return false }
        Task { await dictation.stop() }
        return true
    }

    // MARK: Meetings (spec 4.3, D7)

    /// The meeting that holds the microphone, for the timer chip, the menu
    /// bar and the quit confirmation.
    public var liveMeeting: LiveSessionStatus? {
        setup.status.liveSession.flatMap { $0.kind == .meeting ? $0 : nil }
    }

    /// ⇧⌘R works while no meeting records (from any section: it opens Notes).
    public var canRecordMeeting: Bool { !isMeetingActive }

    /// Record meeting (⇧⌘R): the popover, through the setup card on first use.
    public func recordMeeting() {
        guard liveMeeting == nil else {
            openLiveMeeting()
            return
        }
        setup.require(.recordMeeting) { [weak self] in self?.showsRecordPopover = true }
    }

    /// ⇧⌘M works only while a meeting records.
    public var canMarkMoment: Bool { liveMeeting != nil && recordingControls != nil }

    /// Mark moment (⇧⌘M), from any section.
    public func markMoment() {
        guard canMarkMoment, let controls = recordingControls else { return }
        Task { await controls.markMomentInRecording() }
    }

    /// The timer chip and the menu bar's Open: the meeting note.
    public func openLiveMeeting() {
        if let noteID = liveMeeting?.noteID { openNote?(noteID) }
    }

    /// A meeting records, is paused, or is still finishing (last phrases,
    /// speaker labels): quit asks first and waits for it.
    public var isMeetingActive: Bool { liveMeeting != nil || meetingIsActive?() == true }
    public var isMeetingFinishing: Bool { meetingIsFinishing?() == true }

    /// Quit (spec 6.1), inside the app's 10 s deadline:
    /// 1. dictation stops and its last phrase lands;
    /// 2. a meeting (recording, paused or finishing) quits through N4's
    ///    `stopForQuit`: it marks the session as quitting, shuts the speech
    ///    service down (capture stops, files close; the last phrases, speaker
    ///    pass and conversion are deferred to the next launch), saves the quick
    ///    end and leaves the summary pending for the next launch;
    /// 3. the speech service shuts down (already done when a meeting ran).
    /// A meeting is never stopped the normal way here, and the session never
    /// sees an ordinary end it would summarize during quit.
    public func shutDownForQuit() async {
        await dictation.stop()
        if isMeetingActive, let recordingControls { await recordingControls.finishForQuit() }
        await speech.shutdown()
    }

    // MARK: Files (spec 4.6)

    /// Transcribe file (⇧⌘O): the sheet, through the setup card on first use.
    public func transcribeFiles() {
        setup.require(.transcribeFiles) { [weak self] in self?.showsImportSheet = true }
    }

    /// Audio or video dropped on the Notes window or list (N1 calls this):
    /// the audio and video files among them are queued with the remembered
    /// options, and the sheet shows the queue. Returns false (the drop is
    /// refused) when none qualifies or no queue is wired.
    @discardableResult
    public func transcribe(_ urls: [URL]) -> Bool {
        let urls = TranscribeFilesSheet.acceptedFiles(urls)
        guard !urls.isEmpty, fileQueue != nil else { return false }
        // Without the model the drop is refused (the setup card shows), so
        // no file is taken and then lost if the person declines setup.
        guard setup.modelState.isReady else {
            transcribeFiles()
            return false
        }
        setup.require(.transcribeFiles) { [weak self] in
            guard let self else { return }
            showsImportSheet = true
            guard let queue = fileQueue else { return }
            let preferences = self.preferences
            Task {
                await queue.add(urls, language: preferences.language, speakerLabels: preferences.fileSpeakerLabels,
                                summary: preferences.fileSummary)
            }
        }
        return true
    }

    /// The shell's hook: leaving Notes while dictating stops dictation first,
    /// so its last phrase lands in its note before the editor goes away.
    public static func finishBeforeLeaving(_ section: AppSection,
                                           voice: VoiceModels) -> (@MainActor () async -> Void)? {
        guard section == .notes, voice.dictation.isActive else { return nil }
        return { await voice.stopDictation() }
    }

    /// Switching notes or leaving Notes: the last phrase lands in the note
    /// it belongs to, then dictation stops (N1 calls this).
    public func stopDictation() async {
        await dictation.stop()
    }
}

/// Start was pressed in a build where meetings are not wired yet.
struct MeetingsUnavailable: Error {}
