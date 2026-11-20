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
    /// Clean up fillers as this dictation started; the chip can change the
    /// default meanwhile, which applies from the next dictation.
    public private(set) var sessionCleansUpFillers = true

    /// The fillers chip was changed during this dictation: it applies next time.
    public var cleanUpChangesNextTime: Bool {
        isActive && preferences.cleanUpFillers != sessionCleansUpFillers
    }

    public var isActive: Bool { phase != .idle }

    /// Events handled since launch, so tests can wait for one specific event
    /// instead of yielding and hoping (deterministic under load).
    @ObservationIgnored public private(set) var handledEventCount = 0

    public let preferences: SpeechPreferences
    @ObservationIgnored private let speech: any SpeechServicing
    @ObservationIgnored private let permissions: any PermissionBroker
    /// Held strongly until `stop()` returns, so the last phrase lands even
    /// when the editor's view has gone (N1's controller routes each phrase
    /// to the note it began in). The shell stops dictation before Notes goes
    /// away (`ShellModel.finishBeforeLeaving`).
    @ObservationIgnored private var target: (any DictationInsertionTarget)?
    @ObservationIgnored private var session: DictationSession?
    @ObservationIgnored private var consumer: Task<Void, Never>?
    /// Phrases that were not begun because the target named another note.
    @ObservationIgnored private var strayPhrases: Set<UUID> = []
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
        let stoppedPrevious = isActive
        if isActive { await stop() }
        generation &+= 1
        let attempt = generation
        error = nil
        phase = .starting
        activeNoteID = target.noteID
        self.target = target
        // The editor binds this dictation to the note it shows now, so every
        // phrase lands there whatever is shown later (N1's #237).
        target.dictationWillStart()
        guard await permissions.request(.microphone) == .granted else {
            if attempt == generation { finish(with: .permissionDenied(.microphone)) }
            return
        }
        // Stopped (or started elsewhere) while the permission was asked.
        guard attempt == generation, phase == .starting else { return }
        let options = preferences.dictationOptions
        let session: DictationSession
        do {
            session = try await Self.startSession(speech, options, retryIfBusy: stoppedPrevious)
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
        sessionCleansUpFillers = options.cleanUpFillers
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

    /// Starts a session; right after stopping one (Dictate, Esc, Dictate)
    /// the service may still hold the microphone for a moment, so a
    /// dictation-busy start is tried once more.
    private static func startSession(_ speech: any SpeechServicing, _ options: DictationOptions,
                                     retryIfBusy: Bool) async throws -> DictationSession {
        do {
            return try await speech.startDictation(options)
        } catch SpeechError.busy(.dictation) where retryIfBusy {
            try? await Task.sleep(for: .milliseconds(250))
            return try await speech.startDictation(options)
        }
    }

    /// Sets the language the dictation is labelled with. Fluid Speech Pro
    /// detects the language itself; a pinned language only labels (spike S1).
    public func setLanguage(_ language: SpeechLanguage) async {
        preferences.language = language
        await session?.setLanguage(language)
    }

    /// The Clean up fillers chip mirrors the Settings switch; it applies
    /// from the next dictation (`cleanUpChangesNextTime` says so meanwhile).
    public func setCleanUpFillers(_ on: Bool) {
        preferences.cleanUpFillers = on
    }

    /// Clears the last error (the bar's message was dismissed).
    public func clearError() { error = nil }

    // MARK: Events

    private func handle(_ event: DictationEvent) {
        defer { handledEventCount += 1 }
        switch event {
        case .phraseStarted(let id):
            // Second guard (N1's editor binds a dictation to its note and
            // reports that note while dictating): a phrase never begins in a
            // note other than the one dictation started in. If the target now
            // names another note, the phrase is not begun and dictation stops.
            guard let target, target.noteID == activeNoteID else {
                strayPhrases.insert(id)
                Task { [weak self] in await self?.stop() }
                return
            }
            pendingPhrases.insert(id)
            target.beginPhrase(id)
        case .phraseFinished(let id, let text, let languageCode):
            // A finish goes through even if the shown note changed: the
            // editor routes it by phrase id to the note it began in.
            if strayPhrases.remove(id) != nil { return }
            pendingPhrases.remove(id)
            if let languageCode { detectedLanguage = languageCode }
            // The editor joins the text to what is around the marker
            // (PhraseJoiner: spacing, capitals).
            target?.finishPhrase(id, text: text)
        case .phraseDropped(let id):
            if strayPhrases.remove(id) != nil { return }
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
        strayPhrases = []
        activeNoteID = nil
        target = nil
        session = nil
        consumer = nil
    }
}
