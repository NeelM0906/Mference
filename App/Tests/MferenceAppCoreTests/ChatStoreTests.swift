import Foundation
import MferenceAppCore
import SwiftData
import Testing

@MainActor
@Suite struct ChatStoreTests {
    @Test func appendSavesAtOnceAndKeepsOrder() throws {
        let container = try AppModelContainer.inMemory()
        let store = ConversationStore(container: container)
        let conversation = try store.createConversation(model: .deep, webEnabled: true)
        try store.append(to: conversation, role: .user, content: "Which desk has the best warranty?")
        try store.append(to: conversation, role: .assistant, content: "The Stride E2.", reasoning: "Compare.",
                         model: .deep) { $0.citations = [ChatCitation(number: 1, url: URL(string: "https://a.example/x")!, title: "A")] }

        // A second context sees both messages: they were saved, not just inserted.
        let other = ModelContext(container)
        let saved = try other.fetch(FetchDescriptor<StoredMessage>(sortBy: [SortDescriptor(\.index)]))
        #expect(saved.map(\.content) == ["Which desk has the best warranty?", "The Stride E2."])
        #expect(saved.map(\.index) == [0, 1])
        #expect(saved[1].reasoning == "Compare.")
        #expect(saved[1].citations.first?.label == "a.example/x")
        #expect(saved[1].model == .deep)
        #expect(conversation.title == "Which desk has the best warranty?")
        #expect(conversation.model == .deep)
    }

    @Test func turnsReplayReasoningAndToolCalls() throws {
        let store = ConversationStore(container: try AppModelContainer.inMemory())
        let conversation = try store.createConversation(model: .quick)
        let call = ToolCall(id: "c1", name: "web_search", arguments: .object(["query": .string("desk")]),
                            argumentsJSON: #"{"query":"desk"}"#)
        try store.append(to: conversation, role: .user, content: "hi")
        try store.append(to: conversation, role: .assistant, content: "", reasoning: "search first") { $0.toolCalls = [call] }
        try store.append(to: conversation, role: .tool, content: "results") {
            $0.toolCallID = "c1"
            $0.toolName = "web_search"
        }
        let turns = store.turns(for: conversation)
        #expect(turns.map(\.role) == [.user, .assistant, .tool])
        #expect(turns[1].reasoning == "search first")
        #expect(turns[1].toolCalls == [call])
        #expect(turns[2].toolCallID == "c1")
        #expect(turns[2].toolName == "web_search")
    }

    @Test func earlierAnswersLeaveTheirToolTurnsAndMarkersOut() throws {
        let store = ConversationStore(container: try AppModelContainer.inMemory())
        let conversation = try store.createConversation(model: .quick)
        let call = ToolCall(id: "c1", name: "web_search", arguments: .object([:]), argumentsJSON: "{}")
        try store.append(to: conversation, role: .user, content: "First?")
        try store.append(to: conversation, role: .assistant, content: "", reasoning: "search") { $0.toolCalls = [call] }
        try store.append(to: conversation, role: .tool, content: "long page text") {
            $0.toolCallID = "c1"
            $0.toolName = "web_search"
        }
        try store.append(to: conversation, role: .assistant, content: "It is on sale [1].", reasoning: "found it") {
            $0.citations = [ChatCitation(number: 1, url: URL(string: "https://a.example/")!, title: "A")]
        }
        try store.append(to: conversation, role: .user, content: "Second?")
        try store.append(to: conversation, role: .assistant, content: "") { $0.toolCalls = [call] }
        try store.append(to: conversation, role: .tool, content: "new results") {
            $0.toolCallID = "c1"
            $0.toolName = "web_search"
        }

        let turns = store.turns(for: conversation)
        #expect(turns.map(\.role) == [.user, .assistant, .user, .assistant, .tool])
        #expect(turns[1].content == "It is on sale .")
        #expect(turns[1].reasoning == "found it")
        #expect(turns[4].content == "new results")
    }

    @Test func writesToADeletedChatAreRefused() throws {
        let store = ConversationStore(container: try AppModelContainer.inMemory())
        let conversation = try store.createConversation(model: .quick)
        try store.delete(conversation)
        #expect(throws: ConversationStoreError.conversationDeleted) {
            try store.append(to: conversation, role: .assistant, content: "late")
        }
    }

    @Test func deleteFromIndexDropsTheTail() throws {
        let store = ConversationStore(container: try AppModelContainer.inMemory())
        let conversation = try store.createConversation(model: .quick)
        for text in ["a", "b", "c", "d"] { try store.append(to: conversation, role: .user, content: text) }
        try store.deleteMessages(in: conversation, from: 2)
        #expect(conversation.orderedMessages.map(\.content) == ["a", "b"])
        try store.append(to: conversation, role: .user, content: "e")
        #expect(conversation.orderedMessages.map(\.index) == [0, 1, 2])
    }

    @Test func titleComesFromTheFirstLine() {
        #expect(ConversationStore.title(from: "Lisbon trip, 4 days") == "Lisbon trip, 4 days")
        #expect(ConversationStore.title(from: "one two three four five six seven") == "one two three four five six…")
        #expect(ConversationStore.title(from: "First line\nsecond") == "First line")
        #expect(ConversationStore.title(from: "   ") == Conversation.defaultTitle)
    }

    @Test func groupsSplitTodayEarlierAndPrivate() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let rows = [
            ConversationGroups.Row(id: UUID(), title: "today", updatedAt: now.addingTimeInterval(-60), isPrivate: false),
            ConversationGroups.Row(id: UUID(), title: "week", updatedAt: now.addingTimeInterval(-3 * 86_400), isPrivate: false),
            ConversationGroups.Row(id: UUID(), title: "old", updatedAt: now.addingTimeInterval(-30 * 86_400), isPrivate: false),
            ConversationGroups.Row(id: UUID(), title: "secret", updatedAt: now, isPrivate: true),
        ]
        let groups = ConversationGroups(rows: rows, now: now, calendar: calendar).groups
        #expect(groups.map(\.title) == ["Today", "Previous 7 days", "Earlier", "Private"])
        #expect(groups.map { $0.rows.map(\.title) } == [["today"], ["week"], ["old"], ["secret"]])
        #expect(ConversationGroups(rows: [], now: now).groups.isEmpty)
    }

    @Test func rememberedCountCountsEnabledItems() throws {
        let container = try AppModelContainer.inMemory()
        let store = ConversationStore(container: container)
        container.mainContext.insert(MemoryItem(text: "Prefers metric units"))
        let off = MemoryItem(text: "Old address")
        off.isEnabled = false
        container.mainContext.insert(off)
        try container.mainContext.save()
        #expect(store.rememberedCount() == 1)
    }
}

@Suite struct ChatMarkdownTests {
    @Test func parsesParagraphsTablesAndLists() {
        let text = """
        Three desks have warranties worth comparing.

        | Desk | Price today | Warranty |
        | --- | --- | --- |
        | Stride E2 | $589 | 15 yr frame [1] |
        | Oakline Duo | $499 |

        - first
        - second

        1. one
        2. two

        ```swift
        let x = 1
        ```
        ## Heading
        > quoted
        """
        let blocks = MarkdownBlock.parse(text)
        #expect(blocks == [
            .paragraph("Three desks have warranties worth comparing."),
            .table(header: ["Desk", "Price today", "Warranty"],
                   rows: [["Stride E2", "$589", "15 yr frame [1]"], ["Oakline Duo", "$499", ""]]),
            .bulletList(["first", "second"]),
            .orderedList(start: 1, items: ["one", "two"]),
            .code(language: "swift", text: "let x = 1"),
            .heading(level: 2, text: "Heading"),
            .quote("quoted"),
        ])
    }

    @Test func unterminatedFenceRunsToTheEnd() {
        #expect(MarkdownBlock.parse("```\npartial") == [.code(language: nil, text: "partial")])
    }

    @Test func citationMarkersOnlyForKnownSources() {
        let segments = CitationMarkers.segments("Sale through Sunday [2]. Also [1][3], [9] and [link](x) [1, 2].",
                                                valid: [1, 2, 3])
        #expect(segments == [
            .text("Sale through Sunday "), .citation(2), .text(". Also "), .citation(1), .citation(3),
            .text(", [9] and [link](x) "), .citation(1), .citation(2), .text("."),
        ])
    }

    @Test func citationLabelsAreShort() {
        #expect(ChatCitation(number: 1, url: URL(string: "https://www.oakline.example/sale?x=1")!, title: "").label
                == "oakline.example/sale")
        #expect(ChatCitation(number: 2, url: URL(string: "https://deskreviews.example/")!, title: "").label
                == "deskreviews.example")
    }

    @Test func stepsSummaryMatchesTheMockup() {
        let steps: [ChatStep] = [.search("a", results: 5), .search("b", results: 3), .search("c", results: 4)]
            + (1...7).map { .readPage(URL(string: "https://p\($0).example")!, title: nil) }
        #expect(ChatStepsSummary.title(steps: steps, hasReasoning: true) == "Searched the web · 3 searches · 7 pages read")
        #expect(ChatStepsSummary.title(steps: [], hasReasoning: true) == "Reasoning")
        #expect(ChatStepsSummary.title(steps: [], hasReasoning: false) == nil)
    }
}
