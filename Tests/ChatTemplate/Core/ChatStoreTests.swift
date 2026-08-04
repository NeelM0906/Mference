import Foundation
import Testing

@testable import ChatTemplateCore

/// Scripted backend: yields fixed tokens, optionally waiting between them,
/// optionally failing at the end.
private struct ScriptedBackend: ChatBackend {
    var tokens: [String]
    var tokenDelay: Duration = .zero
    var finalError: Error?

    func stream(
        messages: [ChatMessage],
        model: String
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                for token in tokens {
                    if tokenDelay > .zero {
                        try? await Task.sleep(for: tokenDelay)
                    }
                    if Task.isCancelled { break }
                    continuation.yield(token)
                }
                if let finalError, !Task.isCancelled {
                    continuation.finish(throwing: finalError)
                } else {
                    continuation.finish()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@MainActor
@Suite struct ChatStoreTests {
    @Test func sendCreatesChatAndStreamsResponse() async {
        let store = ChatStore(chats: [], throttle: .zero)
        store.send("Hello there", using: ScriptedBackend(tokens: ["Hi ", "friend"]), model: "m")

        #expect(store.chats.count == 1)
        #expect(store.isGenerating)
        #expect(store.selectedChat?.messages.first?.content == "Hello there")

        await store.generationTask?.value
        #expect(!store.isGenerating)
        #expect(store.streamingText.isEmpty)
        #expect(store.selectedChat?.messages.last?.content == "Hi friend")
        #expect(store.selectedChat?.messages.count == 2)
        #expect(store.errorMessage == nil)
    }

    @Test func sendAppendsToExistingChatAndSendsHistory() async {
        let existing = Chat(title: "T", messages: [
            ChatMessage(role: .user, content: "q1"),
            ChatMessage(role: .assistant, content: "a1"),
        ])
        let store = ChatStore(chats: [existing], throttle: .zero)
        store.selectedChatID = existing.id
        store.send("q2", using: ScriptedBackend(tokens: ["a2"]), model: "m")
        await store.generationTask?.value

        let contents = store.selectedChat?.messages.map(\.content)
        #expect(contents == ["q1", "a1", "q2", "a2"])
        #expect(store.chats.count == 1)
    }

    @Test func ignoresEmptyInputAndSendWhileGenerating() async {
        let store = ChatStore(chats: [], throttle: .zero)
        store.send("   \n", using: ScriptedBackend(tokens: ["x"]), model: "m")
        #expect(store.chats.isEmpty)

        store.send("first", using: ScriptedBackend(
            tokens: ["slow"], tokenDelay: .milliseconds(50)), model: "m")
        store.send("second", using: ScriptedBackend(tokens: ["y"]), model: "m")
        #expect(store.chats.count == 1)
        await store.generationTask?.value
        #expect(store.selectedChat?.messages.count == 2)
    }

    @Test func stopCommitsPartialText() async {
        let store = ChatStore(chats: [], throttle: .zero)
        store.send("go", using: ScriptedBackend(
            tokens: ["one ", "two ", "three"], tokenDelay: .milliseconds(40)), model: "m")

        try? await Task.sleep(for: .milliseconds(60))
        store.stopGenerating()
        await store.generationTask?.value

        #expect(!store.isGenerating)
        let content = store.selectedChat?.messages.last?.content ?? ""
        #expect(content.hasPrefix("one"))
        #expect(content != "one two three")
    }

    @Test func backendErrorSurfacesAndKeepsPartial() async {
        let backend = ScriptedBackend(
            tokens: ["partial"],
            finalError: OpenAICompatError(status: 400, message: "context_length_exceeded"))
        let store = ChatStore(chats: [], throttle: .zero)
        store.send("go", using: backend, model: "m")
        await store.generationTask?.value

        #expect(store.errorMessage?.contains("context_length_exceeded") == true)
        #expect(store.selectedChat?.messages.last?.content == "partial")
        #expect(!store.isGenerating)
    }

    @Test func sidebarActions() {
        let store = ChatStore(chats: MockChats.seed(), throttle: .zero)
        let first = store.chats[0]

        store.toggleStar(id: first.id)
        #expect(store.chats[0].starred != first.starred)

        store.renameChat(id: first.id, title: "  Renamed  ")
        #expect(store.chats[0].title == "Renamed")
        store.renameChat(id: first.id, title: "   ")
        #expect(store.chats[0].title == "Renamed")

        store.selectedChatID = first.id
        store.deleteChat(id: first.id)
        #expect(store.selectedChatID == nil)
        #expect(!store.chats.contains { $0.id == first.id })
    }

    @Test func tracksGeneratingChatID() async {
        let store = ChatStore(chats: [], throttle: .zero)
        #expect(store.generatingChatID == nil)
        store.send("go", using: ScriptedBackend(tokens: ["x"]), model: "m")
        #expect(store.generatingChatID == store.selectedChatID)
        await store.generationTask?.value
        #expect(store.generatingChatID == nil)
    }

    @Test func deletingGeneratingChatCancelsGeneration() async {
        let store = ChatStore(chats: [], throttle: .zero)
        store.send("go", using: ScriptedBackend(
            tokens: ["a", "b", "c"], tokenDelay: .milliseconds(40)), model: "m")
        let chatID = store.selectedChatID!
        let task = store.generationTask

        store.deleteChat(id: chatID)
        await task?.value

        #expect(!store.isGenerating)
        #expect(store.generatingChatID == nil)
        #expect(store.chats.isEmpty)
    }

    @Test func mockBackendSplitsKeepingWhitespace() {
        let words = MockChatBackend.splitKeepingWhitespace("a b\n\nc ")
        #expect(words == ["a ", "b\n\n", "c "])
        #expect(words.joined() == "a b\n\nc ")
    }
}
