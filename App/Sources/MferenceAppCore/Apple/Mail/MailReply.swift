import AppKit
import Foundation

/// A reply the person reviews; sent only when they click Send.
public struct MailDraft: Sendable, Hashable {
    /// The Reply-To address when the mail has one, else the sender's.
    public var to: String
    /// The account that received the mail, so the reply goes out from it;
    /// nil lets Mail pick its default account.
    public var from: String?
    public var subject: String
    public var body: String
    /// The Message-ID answered: Mail's reply command threads with it.
    public var inReplyTo: String?

    public init(to: String, from: String? = nil, subject: String, body: String, inReplyTo: String? = nil) {
        self.to = to
        self.from = from
        self.subject = subject
        self.body = body
        self.inReplyTo = inReplyTo
    }

    /// The reply to `message`: Reply-To honoured, sent from the address the
    /// mail was sent to when it is one of the person's accounts.
    public static func reply(to message: MailMessageSummary, headers: [String: String], accountAddresses: [String],
                             body: String) -> MailDraft {
        let replyTo = headers["reply-to"].flatMap { addresses(in: MailMessageParser.decodeHeader($0)).first }
        let received = (headers["to"].map { addresses(in: $0) } ?? []) + (headers["cc"].map { addresses(in: $0) } ?? [])
        let accounts = Set(accountAddresses.map { $0.lowercased() })
        return MailDraft(to: replyTo ?? message.senderAddress,
                         from: received.first { accounts.contains($0.lowercased()) },
                         subject: replySubject(message.subject), body: body,
                         inReplyTo: message.messageID.hasPrefix("mail-row:") ? nil : message.messageID)
    }

    /// Addresses in a header ("Priya <p@example.com>, b@example.com").
    static func addresses(in header: String) -> [String] {
        header.split(separator: ",").compactMap { part in
            let text = part.trimmingCharacters(in: .whitespaces)
            if let open = text.lastIndex(of: "<"), let close = text.lastIndex(of: ">"), open < close {
                let inner = String(text[text.index(after: open)..<close]).trimmingCharacters(in: .whitespaces)
                return inner.contains("@") ? inner : nil
            }
            return text.contains("@") && !text.contains(" ") ? text : nil
        }
    }

    /// "Re: Wednesday design review" without stacking prefixes.
    public static func replySubject(_ subject: String) -> String {
        let trimmed = subject.trimmingCharacters(in: .whitespaces)
        return trimmed.lowercased().hasPrefix("re:") ? trimmed : "Re: \(trimmed)"
    }
}

/// How a draft should change ("Shorter", "Warmer" in mockup 03).
public enum DraftTone: String, Sendable, Hashable, CaseIterable {
    case standard, shorter, warmer, formal

    var instruction: String {
        switch self {
        case .standard: "Keep it brief and friendly, in the first person, as the recipient of the email."
        case .shorter: "Make it as short as possible: one or two sentences."
        case .warmer: "Make it warmer and more personal, still brief."
        case .formal: "Make it more formal and professional, still brief."
        }
    }
}

/// Writes reply drafts with Quick through the engine ("in your voice ·
/// Quick 4B"). The draft is text only; nothing is sent here.
public struct MailReplyDrafter: Sendable {
    private let engine: any EngineServicing
    public let model: AppModel

    public init(engine: any EngineServicing, model: AppModel = .quick) {
        self.engine = engine
        self.model = model
    }

    static let instructions = AssistantPrompts.instructions("""
        You draft email replies for the person who received the email. Reply with the body of the \
        reply only: no subject line, no placeholders in brackets, no signature block beyond a first \
        name if the email used one. Never promise things the person did not say. Do not invent facts.
        """)

    public func draft(replyTo message: MailMessageSummary, body: String, tone: DraftTone = .standard,
                      previous: String? = nil) async throws -> String {
        // The task first, then the email (and any earlier draft) as fenced,
        // untrusted data.
        let sender = AssistantPrompts.oneLine(message.senderName.isEmpty ? message.senderAddress : message.senderName,
                                              limit: 120)
        var prompt = "Write the reply to this email. \(tone.instruction)\n\n"
            + AssistantPrompts.untrusted("""
                From: \(sender)
                Subject: \(AssistantPrompts.oneLine(message.subject))

                \(AssistantPrompts.clip(body, to: 4_000))
                """)
        if let previous, !previous.isEmpty {
            prompt += "\n\nRewrite this earlier draft accordingly:\n" + AssistantPrompts.untrusted(previous)
        }
        let turns = [ChatTurn(role: .system, content: Self.instructions), ChatTurn(role: .user, content: prompt)]
        let options = GenerationOptions(thinking: false, maximumCompletionTokens: 400)
        var text = ""
        for try await update in engine.generate(model: model, turns: turns, options: options) {
            switch update {
            case .content(let chunk): text += chunk
            case .completed(let summary): if !summary.content.isEmpty { text = summary.content }
            case .prepared, .reasoning, .toolCall: break
            }
        }
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw AssistantModelError.unreadableAnswer }
        return cleaned
    }
}

/// Writes and rewrites a new message with Quick through the engine, for the
/// compose card. The text is only shown for review; nothing is sent here.
public struct MailComposeWriter: Sendable {
    private let engine: any EngineServicing
    public let model: AppModel

    public init(engine: any EngineServicing, model: AppModel = .quick) {
        self.engine = engine
        self.model = model
    }

    static let instructions = AssistantPrompts.instructions("""
        You help a person write a new email that they will review and send themselves. Reply with \
        the email text only: no subject line unless asked, no placeholders in brackets, no notes \
        about what you did. Write in the first person, as the sender. Never promise things the \
        person did not say. Do not invent facts, names, dates or numbers.
        """)

    /// The body of a new email from what the person asked for, building on
    /// their notes when the body already has text.
    public func draft(prompt: String, notes: String, to: String, subject: String) async throws -> String {
        var text = "Write the body of an email. What the person wants: \(AssistantPrompts.oneLine(prompt, limit: 500))"
        let recipient = AssistantPrompts.oneLine(to, limit: 120)
        if !recipient.isEmpty { text += "\nIt goes to: \(recipient)" }
        let topic = AssistantPrompts.oneLine(subject)
        if !topic.isEmpty { text += "\nSubject: \(topic)" }
        if !notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            text += "\n\nTurn the person's notes into the email:\n" + AssistantPrompts.untrusted(
                AssistantPrompts.clip(notes, to: 4_000))
        }
        return try await generate(text, maximumTokens: 600)
    }

    /// The same email in another tone.
    public func rewrite(_ body: String, tone: DraftTone) async throws -> String {
        let text = "Rewrite this email. \(tone.instruction) Keep its meaning and every fact.\n\n"
            + AssistantPrompts.untrusted(AssistantPrompts.clip(body, to: 4_000))
        return try await generate(text, maximumTokens: 600)
    }

    /// The same email with only spelling and grammar fixed.
    public func fixGrammar(_ body: String) async throws -> String {
        let text = "Fix only the spelling, grammar and punctuation of this email. Change nothing else: "
            + "keep its words, tone, order and line breaks wherever they are already correct.\n\n"
            + AssistantPrompts.untrusted(AssistantPrompts.clip(body, to: 4_000))
        return try await generate(text, maximumTokens: 600)
    }

    /// A short subject line for the email.
    public func subject(for body: String) async throws -> String {
        let text = "Write a subject line for this email: at most eight words, no quotes, nothing else.\n\n"
            + AssistantPrompts.untrusted(AssistantPrompts.clip(body, to: 2_000))
        let answer = try await generate(text, maximumTokens: 40)
        let line = answer.components(separatedBy: .newlines).first ?? answer
        var cleaned = AssistantPrompts.oneLine(line, limit: 120).trimmingCharacters(in: .whitespaces)
        if cleaned.lowercased().hasPrefix("subject:") { cleaned = String(cleaned.dropFirst("subject:".count)) }
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”‘’ "))
        guard !cleaned.isEmpty else { throw AssistantModelError.unreadableAnswer }
        return cleaned
    }

    private func generate(_ prompt: String, maximumTokens: Int) async throws -> String {
        let turns = [ChatTurn(role: .system, content: Self.instructions), ChatTurn(role: .user, content: prompt)]
        let options = GenerationOptions(thinking: false, maximumCompletionTokens: maximumTokens)
        var text = ""
        for try await update in engine.generate(model: model, turns: turns, options: options) {
            switch update {
            case .content(let chunk): text += chunk
            case .completed(let summary): if !summary.content.isEmpty { text = summary.content }
            case .prepared, .reasoning, .toolCall: break
            }
        }
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw AssistantModelError.unreadableAnswer }
        return cleaned
    }
}

/// Sends a reviewed draft. Only ever called from the Send button.
public protocol MailSending: Sendable {
    func send(_ draft: MailDraft) async throws
}

public enum MailSendError: Error, Sendable, Hashable {
    case notAllowed
    case invalidRecipient
    case scriptFailed(String)

    public var userMessage: String {
        switch self {
        case .notAllowed: "Mference may not control Mail. Allow it in System Settings > Privacy & Security > Automation."
        case .invalidRecipient: "The recipient's address is not valid."
        case .scriptFailed(let detail): "Mail could not send the message: \(detail)"
        }
    }
}

/// Sends through Apple Mail's scripting (Automation permission, asked
/// through the broker the first time the person clicks Send).
public struct AppleMailSender: MailSending {
    private let broker: any PermissionBroker

    public init(broker: any PermissionBroker) {
        self.broker = broker
    }

    public func send(_ draft: MailDraft) async throws {
        guard Self.isPlausibleAddress(draft.to) else { throw MailSendError.invalidRecipient }
        // Only a granted Automation permission runs the script: a declined
        // reason (still not determined) must not reach Mail's own prompt.
        guard await broker.request(.mailAutomation) == .granted else { throw MailSendError.notAllowed }
        let source = Self.script(for: draft)
        try await MainActor.run {
            var error: NSDictionary?
            guard let script = NSAppleScript(source: source) else { throw MailSendError.scriptFailed("bad script") }
            _ = script.executeAndReturnError(&error)
            if let error {
                let number = error[NSAppleScript.errorNumber] as? Int
                if number == -1743 { throw MailSendError.notAllowed }
                throw MailSendError.scriptFailed(error[NSAppleScript.errorMessage] as? String ?? "unknown error")
            }
        }
    }

    static func isPlausibleAddress(_ address: String) -> Bool {
        let parts = address.split(separator: "@")
        return parts.count == 2 && !parts[0].isEmpty && parts[1].contains(".")
            && !address.contains(where: { $0.isWhitespace || $0 == "\"" || $0 == "\\" || $0 == "<" || $0 == ">" })
    }

    /// AppleScript string literal: backslashes and quotes escaped, control
    /// characters other than tab and line breaks dropped (a NUL would cut the
    /// script short).
    static func literal(_ text: String) -> String {
        let cleaned = String(String.UnicodeScalarView(text.unicodeScalars.filter {
            $0 == "\n" || $0 == "\r" || $0 == "\t" || !($0.properties.generalCategory == .control)
        }))
        return "\"" + cleaned.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }

    /// Replies through Mail's own reply command when the original is in the
    /// inbox (threading, Reply-To and quoting as Mail does them), else as a
    /// new message; from the receiving account when known.
    static func script(for draft: MailDraft) -> String {
        let messageID = draft.inReplyTo.map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "<> ")) } ?? ""
        let sender = draft.from.map { "\n    set sender of newMessage to \(literal($0))" } ?? ""
        return """
        tell application "Mail"
            set originals to {}
            if \(literal(messageID)) is not "" then set originals to (messages of inbox whose message id is \(literal(messageID)))
            if (count of originals) > 0 then
                set newMessage to reply (item 1 of originals) without opening window
                set content of newMessage to \(literal(draft.body))
            else
                set newMessage to make new outgoing message with properties {subject:\(literal(draft.subject)), \
        content:\(literal(draft.body)), visible:false}
                tell newMessage to make new to recipient at end of to recipients with properties \
        {address:\(literal(draft.to))}
            end if\(sender)
            send newMessage
        end tell
        """
    }
}

/// Records sends instead of sending: tests and previews.
public actor RecordingMailSender: MailSending {
    public private(set) var sent: [MailDraft] = []

    public init() {}

    public func send(_ draft: MailDraft) async throws {
        sent.append(draft)
    }
}
