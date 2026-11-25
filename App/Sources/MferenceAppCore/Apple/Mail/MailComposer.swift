import Foundation
import Observation

/// What the compose card's Quick tools do.
public enum ComposeAction: Sendable, Hashable {
    /// The body from the prompt, building on any notes already in the body.
    case draft
    case rewrite(DraftTone)
    case fixGrammar
    case suggestSubject
}

/// Compose (C): one new message at a time, docked at the bottom of Mail.
/// It lives as long as the app (owned by `MailInboxModel`), so a draft keeps
/// its place while the person works in another section. Nothing is sent
/// until `send()`, which only the Send button calls.
@MainActor @Observable
public final class MailComposer {
    public enum AssistState: Sendable, Hashable {
        case idle
        case writing(ComposeAction)
        /// `message` is user-facing.
        case failed(String)
    }

    public enum SendState: Sendable, Hashable {
        case idle
        case sending
        /// `message` is user-facing.
        case failed(String)
    }

    public private(set) var isOpen = false
    public private(set) var isMinimized = false
    public var to = ""
    public var subject = ""
    public var body = ""
    /// What the person asked Quick to write; kept across actions and Undo.
    public var prompt = ""
    public private(set) var assistState: AssistState = .idle
    public private(set) var sendState: SendState = .idle
    /// The text before the last Quick change, for Undo.
    public private(set) var undo: (body: String, subject: String)?

    @ObservationIgnored private var task: Task<Void, Never>?
    /// Bumped by every action, Stop and close: a superseded task that
    /// finishes late changes nothing.
    @ObservationIgnored private var generation = 0
    /// Changes whenever a draft is dropped, so a send that finishes later
    /// can only ever close the draft it sent.
    @ObservationIgnored private var draftID = UUID()

    private let writer: MailComposeWriter?
    private let sender: any MailSending

    public init(writer: MailComposeWriter?, sender: any MailSending) {
        self.writer = writer
        self.sender = sender
    }

    // MARK: The card

    /// Opens an empty draft, or brings back the one already open.
    public func open() {
        isOpen = true
        isMinimized = false
    }

    public func minimize() {
        guard isOpen else { return }
        isMinimized = true
    }

    public func restore() {
        guard isOpen else { return }
        isMinimized = false
    }

    /// Close and Discard are off while the message sends.
    public var canDiscard: Bool { sendState != .sending }

    /// Closes the card and drops the draft. Not while it sends: the send's
    /// own outcome decides whether the card closes.
    public func discard() {
        guard canDiscard else { return }
        draftID = UUID()
        cancelAssist()
        isOpen = false
        isMinimized = false
        to = ""
        subject = ""
        body = ""
        prompt = ""
        undo = nil
        assistState = .idle
        sendState = .idle
    }

    /// The draft has something worth confirming before it is dropped.
    public var hasContent: Bool {
        ![to, subject, body, prompt].allSatisfy { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    // MARK: Quick

    public var isWriting: Bool {
        if case .writing = assistState { return true }
        return false
    }

    /// The fields are read-only while Quick writes or the message sends.
    public var isLocked: Bool { isWriting || sendState == .sending }

    public var canDraft: Bool {
        !isLocked && writer != nil && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Rewrites, grammar and the subject need a body to work on.
    public var canRewrite: Bool {
        !isLocked && writer != nil && !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canUndo: Bool { !isLocked && undo != nil }

    /// "Quick 4B", shown beside the tools.
    public var modelLabel: String? { writer?.model.shortName }

    /// Starts a Quick action; `waitForAssist()` awaits it (tests).
    public func run(_ action: ComposeAction) {
        let allowed = action == .draft ? canDraft : canRewrite
        guard allowed, let writer else {
            if writer == nil { assistState = .failed("Quick is needed to write drafts. Install it from the model library.") }
            return
        }
        generation += 1
        let mine = generation
        assistState = .writing(action)
        let prompt = prompt, notes = body, to = to, subject = subject
        task = Task { [weak self] in
            let result: Result<String, any Error>
            do {
                switch action {
                case .draft:
                    result = .success(try await writer.draft(prompt: prompt, notes: notes, to: to, subject: subject))
                case .rewrite(let tone): result = .success(try await writer.rewrite(notes, tone: tone))
                case .fixGrammar: result = .success(try await writer.fixGrammar(notes))
                case .suggestSubject: result = .success(try await writer.subject(for: notes))
                }
            } catch {
                result = .failure(error)
            }
            self?.finish(action, result, generation: mine)
        }
    }

    private func finish(_ action: ComposeAction, _ result: Result<String, any Error>, generation mine: Int) {
        guard mine == generation else { return }
        task = nil
        switch result {
        case .success(let text):
            undo = (body, subject)
            if action == .suggestSubject { subject = text } else { body = text }
            assistState = .idle
        case .failure(let error):
            switch error {
            case is CancellationError: assistState = .idle
            case let error as EngineServiceError: assistState = .failed(error.userMessage)
            case let error as AssistantModelError: assistState = .failed(error.userMessage)
            case let error as MailComposeError: assistState = .failed(error.userMessage)
            default: assistState = .failed("Quick could not write this.")
            }
        }
    }

    /// Stop: the running action ends and its text is never used.
    public func cancelAssist() {
        generation += 1
        // Kept (cancelled) so `waitForAssist()` can still await its end.
        task?.cancel()
        if isWriting { assistState = .idle }
    }

    /// Puts back the body and subject from before the last Quick change.
    public func undoAssist() {
        guard canUndo, let undo else { return }
        body = undo.body
        subject = undo.subject
        self.undo = nil
    }

    public func dismissAssistError() {
        if case .failed = assistState { assistState = .idle }
    }

    public func waitForAssist() async {
        await task?.value
    }

    // MARK: Send

    public var canSend: Bool { isOpen && !isLocked }

    /// Only the Send button calls this. Returns nil when sent, else the
    /// reason it was not; a sent draft closes the card.
    @discardableResult
    public func send() async -> String? {
        guard canSend else { return nil }
        let sending = draftID
        sendState = .sending
        let error = await Self.send(to: to, subject: subject, body: body, via: sender)
        guard sending == draftID else { return error }
        if let error {
            sendState = .failed(error)
        } else {
            sendState = .idle
            discard()
        }
        return error
    }

    /// A new message through Mail. Returns nil when sent, else the reason it
    /// was not. One plain address for now: no lists, Cc or Bcc.
    static func send(to: String, subject: String, body: String, via sender: any MailSending) async -> String? {
        let recipient = to.trimmingCharacters(in: .whitespacesAndNewlines)
        guard AppleMailSender.isPlausibleAddress(recipient) else { return MailSendError.invalidRecipient.userMessage }
        let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "Write the message first." }
        do {
            try await sender.send(MailDraft(to: recipient, subject: subject.trimmingCharacters(in: .whitespaces),
                                            body: text))
            return nil
        } catch let error as MailSendError {
            return error.userMessage
        } catch {
            return "Mail could not send the message."
        }
    }
}
