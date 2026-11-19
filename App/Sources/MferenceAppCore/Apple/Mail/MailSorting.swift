import Foundation

/// What the sorter sees of one message: never the whole body.
public struct MailSortItem: Sendable, Hashable {
    /// Mail's Message-ID header (the `MailBucketAssignment` key).
    public var messageID: String
    public var sender: String
    public var subject: String
    /// The first part of the plain-text body.
    public var snippet: String

    public init(messageID: String, sender: String, subject: String, snippet: String) {
        self.messageID = messageID
        self.sender = sender
        self.subject = subject
        self.snippet = snippet
    }
}

/// A bucket the sorter may choose. The key is stable; the name is the
/// person's (renamable, L8), so the model sees the key and a description.
public struct MailSortBucket: Sendable, Hashable {
    public var key: String
    public var description: String

    public init(key: String, description: String) {
        self.key = key
        self.description = description
    }
}

/// A meeting found in a mail, before it becomes a `ProposedEvent`.
public struct MailEventSuggestion: Sendable, Hashable {
    public var title: String
    public var start: Date
    /// When the mail gives no end, the card proposes 45 minutes.
    public var end: Date

    public init(title: String, start: Date, end: Date) {
        self.title = title
        self.start = start
        self.end = end
    }

    /// Nil when the answer names no meeting or its times do not parse.
    init?(json object: [String: Any], timeZone: TimeZone) {
        let string = AssistantAnswerParser.string
        guard let title = string(object["title"]), let startText = string(object["start"]),
              let start = Self.parse(startText, timeZone: timeZone) else { return nil }
        let end = string(object["end"]).flatMap { Self.parse($0, timeZone: timeZone) }
        self.title = title
        self.start = start
        self.end = end.flatMap { $0 > start ? $0 : nil } ?? start.addingTimeInterval(45 * 60)
    }

    static func parse(_ text: String, timeZone: TimeZone) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        for format in ["yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd HH:mm"] {
            formatter.dateFormat = format
            if let date = formatter.date(from: text) { return date }
        }
        return nil
    }
}
