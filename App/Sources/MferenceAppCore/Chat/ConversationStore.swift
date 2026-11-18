import Foundation
import SwiftData

/// The chat's view of the SwiftData store. Every message is inserted and
/// saved the moment it is final, so a crash in the engine loses at most the
/// answer being streamed (spec 13).
@MainActor
public final class ConversationStore {
    public let context: ModelContext
    /// A context does not keep its container alive, and SwiftData traps when
    /// a context outlives it; the store holds it for as long as it is used.
    public let container: ModelContainer

    public init(context: ModelContext) {
        self.context = context
        self.container = context.container
    }

    public convenience init(container: ModelContainer) {
        self.init(context: container.mainContext)
    }

    // MARK: Conversations

    /// Every conversation, most recently updated first.
    public func conversations() throws -> [Conversation] {
        try context.fetch(FetchDescriptor<Conversation>(
            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]))
    }

    public func conversation(id: UUID) throws -> Conversation? {
        var descriptor = FetchDescriptor<Conversation>(predicate: #Predicate { $0.id == id })
        descriptor.fetchLimit = 1
        return try context.fetch(descriptor).first
    }

    @discardableResult
    public func createConversation(model: AppModel, webEnabled: Bool = false, isPrivate: Bool = false,
                                   now: Date = .now) throws -> Conversation {
        let conversation = Conversation(createdAt: now, modelRawValue: model.rawValue,
                                        webEnabled: webEnabled, isPrivate: isPrivate)
        context.insert(conversation)
        try context.save()
        return conversation
    }

    public func update(_ conversation: Conversation, _ change: (Conversation) -> Void) throws {
        change(conversation)
        try context.save()
    }

    public func delete(_ conversation: Conversation) throws {
        context.delete(conversation)
        try context.save()
    }

    // MARK: Messages

    /// Appends a final message and saves at once.
    @discardableResult
    public func append(to conversation: Conversation, role: ChatTurn.Role, content: String,
                       reasoning: String? = nil, model: AppModel? = nil, now: Date = .now,
                       configure: (StoredMessage) -> Void = { _ in }) throws -> StoredMessage {
        guard conversation.modelContext != nil, !conversation.isDeleted else {
            throw ConversationStoreError.conversationDeleted
        }
        let next = (conversation.messages.map(\.index).max() ?? -1) + 1
        let message = StoredMessage(index: next, role: role, content: content,
                                    reasoning: reasoning?.isEmpty == true ? nil : reasoning,
                                    modelRawValue: model?.rawValue, createdAt: now)
        configure(message)
        context.insert(message)
        message.conversation = conversation
        conversation.updatedAt = now
        if role == .user, conversation.title == Conversation.defaultTitle {
            conversation.title = Self.title(from: content)
        }
        try context.save()
        return message
    }

    /// Deletes every message at or after `index` (regenerate).
    public func deleteMessages(in conversation: Conversation, from index: Int) throws {
        for message in conversation.messages where message.index >= index {
            context.delete(message)
        }
        try context.save()
    }

    /// The history sent to the model, in order, reasoning included (engine
    /// contract). Tool calls and tool results of earlier answers are left
    /// out: their page text would grow the history without bound, and their
    /// source numbers restart with each answer. Earlier answers also lose
    /// their "[n]" markers for the same reason. The answer being written keeps
    /// all of its tool turns.
    public func turns(for conversation: Conversation) -> [ChatTurn] {
        let ordered = conversation.orderedMessages
        let lastUser = ordered.lastIndex { $0.role == .user } ?? -1
        return ordered.enumerated().compactMap { offset, message in
            guard offset < lastUser else { return message.turn }
            switch message.role {
            case .tool:
                return nil
            case .assistant where !message.toolCalls.isEmpty:
                return nil
            case .assistant:
                var turn = message.turn
                let numbers = Set(message.citations.map(\.number))
                if !numbers.isEmpty, let content = turn.content {
                    turn.content = CitationMarkers.segments(content, valid: numbers).compactMap {
                        if case .text(let text) = $0 { return text }
                        return nil
                    }.joined()
                }
                return turn
            default:
                return message.turn
            }
        }
    }

    /// Enabled memory items, for the sidebar card.
    public func rememberedCount() -> Int {
        (try? context.fetchCount(FetchDescriptor<MemoryItem>(predicate: #Predicate { $0.isEnabled == true }))) ?? 0
    }

    // MARK: Titles

    /// A provisional title from the first message: its first line, at most
    /// about six words. (Apple's model may retitle later, L12.)
    nonisolated public static func title(from text: String) -> String {
        let line = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        let words = line.split(whereSeparator: \.isWhitespace)
        guard !words.isEmpty else { return Conversation.defaultTitle }
        var title = words.prefix(6).joined(separator: " ")
        if title.count > 48 { title = String(title.prefix(47)) + "…" }
        else if words.count > 6 { title += "…" }
        return title
    }
}

public enum ConversationStoreError: Error, Sendable, Hashable {
    /// The chat was deleted while something was still writing to it.
    case conversationDeleted
}

extension Conversation {
    /// The schema's default title; replaced by the first message.
    public static let defaultTitle = "New chat"
}

/// Sidebar grouping (mockup 02: Today, then earlier chats; Private last).
public struct ConversationGroups: Sendable, Hashable {
    public struct Row: Sendable, Hashable, Identifiable {
        public var id: UUID
        public var title: String
        public var updatedAt: Date
        public var isPrivate: Bool

        public init(id: UUID, title: String, updatedAt: Date, isPrivate: Bool) {
            self.id = id
            self.title = title
            self.updatedAt = updatedAt
            self.isPrivate = isPrivate
        }
    }

    public struct Group: Sendable, Hashable, Identifiable {
        public var title: String
        public var rows: [Row]
        public var isPrivate: Bool
        public var id: String { title }
    }

    public var groups: [Group]

    public init(rows: [Row], now: Date = .now, calendar: Calendar = .current) {
        let shared = rows.filter { !$0.isPrivate }.sorted { $0.updatedAt > $1.updatedAt }
        let startOfToday = calendar.startOfDay(for: now)
        let startOfWeek = calendar.date(byAdding: .day, value: -7, to: startOfToday) ?? startOfToday
        let today = shared.filter { $0.updatedAt >= startOfToday }
        let week = shared.filter { $0.updatedAt < startOfToday && $0.updatedAt >= startOfWeek }
        let earlier = shared.filter { $0.updatedAt < startOfWeek }
        let privateRows = rows.filter(\.isPrivate).sorted { $0.updatedAt > $1.updatedAt }
        groups = [
            Group(title: "Today", rows: today, isPrivate: false),
            Group(title: "Previous 7 days", rows: week, isPrivate: false),
            Group(title: "Earlier", rows: earlier, isPrivate: false),
            Group(title: "Private", rows: privateRows, isPrivate: true),
        ].filter { !$0.rows.isEmpty }
    }
}
