import AppKit
import Foundation

/// A reply the person reviews; sent only when they click Send.
public struct MailDraft: Sendable, Hashable {
    public var to: String
    public var subject: String
    public var body: String
    /// The message answered, for the source link.
    public var inReplyTo: String?

    public init(to: String, subject: String, body: String, inReplyTo: String? = nil) {
        self.to = to
        self.subject = subject
        self.body = body
        self.inReplyTo = inReplyTo
    }

    /// "Re: Wednesday design review" without stacking prefixes.
    public static func replySubject(_ subject: String) -> String {
        let trimmed = subject.trimmingCharacters(in: .whitespaces)
        return trimmed.lowercased().hasPrefix("re:") ? trimmed : "Re: \(trimmed)"
    }
}

/// How a draft should change ("Shorter", "Warmer" in mockup 03).
public enum DraftTone: String, Sendable, Hashable, CaseIterable {
    case standard, shorter, warmer

    var instruction: String {
        switch self {
        case .standard: "Keep it brief and friendly, in the first person, as the recipient of the email."
        case .shorter: "Make it as short as possible: one or two sentences."
        case .warmer: "Make it warmer and more personal, still brief."
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

    static let instructions = """
        You draft email replies for the person who received the email. Reply with the body of the \
        reply only: no subject line, no placeholders in brackets, no signature block beyond a first \
        name if the email used one. Never promise things the person did not say. Do not invent facts.
        """

    public func draft(replyTo message: MailMessageSummary, body: String, tone: DraftTone = .standard,
                      previous: String? = nil) async throws -> String {
        var prompt = """
            Email from \(message.senderName.isEmpty ? message.senderAddress : message.senderName):
            Subject: \(message.subject)

            \(AssistantPrompts.clip(body, to: 4_000))

            \(tone.instruction)
            """
        if let previous, !previous.isEmpty {
            prompt += "\n\nRewrite this draft accordingly:\n\(previous)"
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
        case .scriptFailed(let detail): "Mail could not send the reply: \(detail)"
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
        let state = await broker.request(.mailAutomation)
        guard state == .granted || state == .notDetermined else { throw MailSendError.notAllowed }
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

    /// AppleScript string literal: backslashes and quotes escaped.
    static func literal(_ text: String) -> String {
        "\"" + text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }

    static func script(for draft: MailDraft) -> String {
        """
        tell application "Mail"
            set newMessage to make new outgoing message with properties {subject:\(literal(draft.subject)), \
        content:\(literal(draft.body)), visible:false}
            tell newMessage to make new to recipient at end of to recipients with properties \
        {address:\(literal(draft.to))}
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
