import Foundation
import Observation

/// The one consumer of `DictationSession.events` (stream ownership rule),
/// held by `AppServices` for the app's lifetime: it starts dictation into a
/// note, puts each phrase where its caret was, and on stop (Done, Esc, ⇧⌘D,
/// switching notes, leaving Notes) finishes the last phrase before it lets
/// go of the note.
@MainActor
@Observable
public final class DictationController {
    public enum Phase: Sendable, Hashable {
        case idle
        /// Asking for the microphone or starting the session.
        case starting
        case listening
        /// Finishing the last phrase.
        case stopping
    }

    public private(set) var phase: Phase = .idle
    /// The note being dictated into.
    public private(set) var activeNoteID: UUID?
    /// Input level, 0...1 (the bar's meter).
    public private(set) var level: Float = 0
    /// The language last detected (the chip's "English · auto").
    public private(set) var detectedLanguage: String?
    /// The last thing that went wrong, for the bar or an alert; cleared on start.
    public private(set) var error: SpeechError?
    /// Phrases started and not finished yet.
    public private(set) var pendingPhrases: Set<UUID> = []

    public var isActive: Bool { phase != .idle }

    public let preferences: SpeechPreferences
    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let permissions: any PermissionBroker
    @ObservationIgnored private weak var target: (any DictationInsertionTarget)?
    @ObservationIgnored private var session: DictationSession?
    @ObservationIgnored private var consumer: Task<Void, Never>?
    /// Bumped by each start and by a stop during a start, so a start that
    /// was overtaken does nothing more.
    @ObservationIgnored private var generation: UInt64 = 0

    public init(speech: any SpeechServicing, permissions: any PermissionBroker, preferences: SpeechPreferences) {
        self.speech = speech
        self.permissions = permissions
        self.preferences = preferences
    }

    /// Whether dictation is going into `noteID` (the toolbar's "Dictating").
    public func isDictating(into noteID: UUID) -> Bool { isActive && activeNoteID == noteID }

    /// The Dictate button and ⇧⌘D: stop when dictating into this note;
    /// otherwise stop any other dictation and start here.
    public func toggle(_ target: any DictationInsertionTarget) async {
        if isDictating(into: target.noteID) {
            await stop()
        } else {
            await start(in: target)
        }
    }

    /// Starts dictation into `target` at its caret. The caller made sure the
    /// model is installed (the setup card). Asks for the microphone the
    /// first time (reason, then the system prompt).
    public func start(in target: any DictationInsertionTarget) async {
        if isActive { await stop() }
        generation &+= 1
        let attempt = generation
        error = nil
        phase = .starting
        activeNoteID = target.noteID
        self.target = target
        guard await permissions.request(.microphone) == .granted else {
            if attempt == generation { finish(with: .permissionDenied(.microphone)) }
            return
        }
        // Stopped (or started elsewhere) while the permission was asked.
        guard attempt == generation, phase == .starting else { return }
        let session: DictationSession
        do {
            session = try await speech.startDictation(preferences.dictationOptions)
        } catch {
            if attempt == generation {
                finish(with: (error as? SpeechError) ?? .failed(String(describing: error)))
            }
            return
        }
        // Stopped while the session was starting: end it unused.
        guard attempt == generation, phase == .starting else {
            await session.stop()
            for await _ in session.events {}
            return
        }
        self.session = session
        phase = .listening
        consumer = Task { [weak self] in
            for await event in session.events {
                self?.handle(event)
            }
            self?.sessionEnded()
        }
    }

    /// Done, Esc, ⇧⌘D, switching notes or leaving Notes: finishes the phrase
    /// in progress into the note, then ends. Returns once it is all in.
    public func stop() async {
        switch phase {
        case .idle:
            return
        case .starting:
            // The start sees the new generation and ends its session unused.
            generation &+= 1
            finish(with: nil)
            return
        case .listening, .stopping:
            phase = .stopping
            await session?.stop()
            await consumer?.value
        }
    }

    /// The language chip: pin one of the 25, or Automatic. Remembered as the default.
    public func setLanguage(_ language: SpeechLanguage) async {
        preferences.language = language
        await session?.setLanguage(language)
    }

    /// The Clean up fillers chip mirrors the Settings switch; it applies
    /// from the next dictation.
    public func setCleanUpFillers(_ on: Bool) {
        preferences.cleanUpFillers = on
    }

    /// Clears the last error (the bar's message was dismissed).
    public func clearError() { error = nil }

    // MARK: Events

    private func handle(_ event: DictationEvent) {
        switch event {
        case .phraseStarted(let id):
            pendingPhrases.insert(id)
            target?.beginPhrase(id)
        case .phraseFinished(let id, let text, let languageCode):
            pendingPhrases.remove(id)
            if let languageCode { detectedLanguage = languageCode }
            // The editor joins the text to what is around the marker
            // (PhraseJoiner: spacing, capitals).
            target?.finishPhrase(id, text: text)
        case .phraseDropped(let id):
            pendingPhrases.remove(id)
            target?.dropPhrase(id)
        case .level(let value):
            level = min(max(value, 0), 1)
        case .failed(let failure):
            error = failure
            target?.dropAllPhrases()
            pendingPhrases = []
        case .ended:
            break
        }
    }

    private func sessionEnded() {
        // A phrase that never finished (the stream ended early) loses its marker.
        target?.dropAllPhrases()
        finish(with: error)
    }

    private func finish(with failure: SpeechError?) {
        error = failure
        phase = .idle
        level = 0
        pendingPhrases = []
        activeNoteID = nil
        target = nil
        session = nil
        consumer = nil
    }
}
