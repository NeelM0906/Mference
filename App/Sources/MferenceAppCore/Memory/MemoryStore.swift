import Foundation
import SwiftData

/// Where a remembered fact came from (Settings > Memory, "where learned").
public enum MemorySource: Sendable, Hashable {
    /// Typed in Settings by the person.
    case manual
    /// Learned in a chat. The store looks the chat up itself: a private or
    /// missing chat never feeds memory, whatever the caller believes.
    case chat(id: UUID)
    /// Learned from mail; `label` describes it ("14 sent emails").
    case mail(id: UUID?, label: String)
    /// Learned from a confirmed scan.
    case scan(id: UUID, title: String)

    var kind: String {
        switch self {
        case .manual: "manual"
        case .chat: "chat"
        case .mail: "mail"
        case .scan: "scan"
        }
    }

    var sourceID: UUID? {
        switch self {
        case .manual: nil
        case .chat(let id): id
        case .mail(let id, _): id
        case .scan(let id, _): id
        }
    }

    var label: String? {
        switch self {
        case .manual, .chat: nil
        case .mail(_, let label): label
        case .scan(_, let title): title
        }
    }
}

/// What Mference remembers about the person, kept on this Mac (spec 4.4,
/// A4): add, edit, delete, switch off, and where each item was learned.
///
/// Private chats never feed memory: `learn(_:from:)` reads the chat's own
/// `isPrivate` and refuses private or unknown chats, and the chat store calls
/// `forget(learnedFrom:)` when a chat is made private or deleted, removing
/// what it taught. A fact the person switched off stays off: only adding it
/// by hand switches it on again.
@MainActor
public final class MemoryStore {
    public let context: ModelContext
    /// Held so an in-memory container lives as long as the store.
    private let container: ModelContainer?

    public init(context: ModelContext) {
        self.context = context
        self.container = nil
    }

    /// Uses the container's main context and keeps the container alive.
    public init(container: ModelContainer) {
        self.context = container.mainContext
        self.container = container
    }

    /// Every item, newest change first.
    public func items() throws -> [MemoryItem] {
        try context.fetch(FetchDescriptor<MemoryItem>(sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]))
    }

    /// The enabled facts, for the model's context.
    public func enabledFacts() throws -> [String] {
        try items().filter(\.isEnabled).map(\.text)
    }

    public func count() throws -> Int {
        try context.fetchCount(FetchDescriptor<MemoryItem>())
    }

    /// Adds a fact the person typed. Returns the existing item when the same
    /// fact (ignoring case and surrounding spaces) is already remembered.
    @discardableResult
    public func add(_ text: String, now: Date = .now) throws -> MemoryItem? {
        try insert(text, source: .manual, now: now)
    }

    /// Remembers a fact the app learned. Returns nil, and stores nothing,
    /// for a private or unknown chat, an empty fact, or a fact the person
    /// switched off. A fact already remembered keeps its original source.
    @discardableResult
    public func learn(_ text: String, from source: MemorySource, now: Date = .now) throws -> MemoryItem? {
        var label = source.label
        if case .chat(let id) = source {
            let match = FetchDescriptor<Conversation>(predicate: #Predicate { $0.id == id })
            guard let conversation = try context.fetch(match).first, !conversation.isPrivate else { return nil }
            label = conversation.title
        }
        return try insert(text, source: source, label: label, now: now)
    }

    private func insert(_ text: String, source: MemorySource, label: String? = nil,
                        now: Date) throws -> MemoryItem? {
        let trimmed = Self.normalized(text)
        guard !trimmed.isEmpty else { return nil }
        let key = trimmed.lowercased()
        if let existing = try items().first(where: { $0.text.lowercased() == key }) {
            // Only the person turns a switched-off fact back on.
            guard source == .manual else { return existing.isEnabled ? existing : nil }
            existing.isEnabled = true
            existing.updatedAt = now
            try context.save()
            return existing
        }
        let item = MemoryItem(text: trimmed, sourceKind: source.kind, sourceID: source.sourceID,
                              sourceLabel: label ?? source.label, createdAt: now)
        context.insert(item)
        try context.save()
        return item
    }

    /// Changes the fact's wording; an empty edit deletes it.
    public func edit(_ item: MemoryItem, text: String, now: Date = .now) throws {
        let trimmed = Self.normalized(text)
        guard !trimmed.isEmpty else { return try delete(item) }
        item.text = trimmed
        item.updatedAt = now
        try context.save()
    }

    public func setEnabled(_ item: MemoryItem, _ enabled: Bool, now: Date = .now) throws {
        item.isEnabled = enabled
        item.updatedAt = now
        try context.save()
    }

    public func delete(_ item: MemoryItem) throws {
        context.delete(item)
        try context.save()
    }

    /// Removes everything learned from one chat, mail or scan; returns how
    /// many items went. The chat store calls it when a chat is made private
    /// or deleted.
    @discardableResult
    public func forget(learnedFrom id: UUID) throws -> Int {
        let matches = try context.fetch(FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.sourceID == id }))
        for item in matches { context.delete(item) }
        try context.save()
        return matches.count
    }

    public func deleteAll() throws {
        try context.delete(model: MemoryItem.self)
        try context.save()
    }

    /// "From chat, Sep 10", "From Mail, Sep 14", "Added by you, Sep 2".
    public static func whereLearned(_ item: MemoryItem, locale: Locale = .current,
                                    timeZone: TimeZone = .current) -> String {
        let style = Date.FormatStyle(locale: locale, calendar: Calendar(identifier: .gregorian), timeZone: timeZone)
            .month(.abbreviated).day()
        let date = item.createdAt.formatted(style)
        let origin = switch item.sourceKind {
        case "chat": "From chat"
        case "mail": "From Mail"
        case "scan": "From a scan"
        default: "Added by you"
        }
        return "\(origin), \(date)"
    }

    static func normalized(_ text: String) -> String {
        text.split(whereSeparator: \.isNewline).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
