import Foundation
import Observation

/// Chat session state: sidebar chats, the active transcript, and the
/// streaming lifecycle. Streaming text publishes at most once per throttle
/// interval (~30 fps), mirroring the template's `createStreamingStore` +
/// `STREAMING_THROTTLE_MS`.
@MainActor
@Observable
public final class ChatStore {
    public private(set) var chats: [Chat]
    public var selectedChatID: UUID?
    public private(set) var isGenerating = false
    public private(set) var streamingText = ""
    public var errorMessage: String?
    public private(set) var generationTask: Task<Void, Never>?
    /// The chat receiving the in-flight response, so views don't render
    /// `streamingText` into a different chat selected mid-generation.
    public private(set) var generatingChatID: UUID?

    private let throttle: Duration

    public init(
        chats: [Chat] = MockChats.seed(),
        throttle: Duration = .milliseconds(32)
    ) {
        self.chats = chats
        self.throttle = throttle
    }

    public var selectedChat: Chat? {
        guard let selectedChatID else { return nil }
        return chats.first { $0.id == selectedChatID }
    }

    // MARK: - Sidebar actions

    public func newChat() {
        selectedChatID = nil
        errorMessage = nil
    }

    public func deleteChat(id: UUID) {
        if generatingChatID == id { generationTask?.cancel() }
        chats.removeAll { $0.id == id }
        if selectedChatID == id { selectedChatID = nil }
    }

    public func renameChat(id: UUID, title: String) {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        update(id: id) { $0.title = trimmed }
    }

    public func toggleStar(id: UUID) {
        update(id: id) { $0.starred.toggle() }
    }

    // MARK: - Sending

    public func send(_ text: String, using backend: any ChatBackend, model: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }

        errorMessage = nil
        let chatID = selectedChatID.flatMap { id in chats.contains { $0.id == id } ? id : nil }
            ?? createChat(titledFrom: trimmed)
        selectedChatID = chatID

        update(id: chatID) { chat in
            chat.messages.append(ChatMessage(role: .user, content: trimmed))
            chat.messages.append(ChatMessage(role: .assistant, content: ""))
            chat.lastActivity = Date()
        }
        let history: [ChatMessage] = selectedChat.map { Array($0.messages.dropLast()) } ?? []

        isGenerating = true
        streamingText = ""
        generatingChatID = chatID
        let throttle = self.throttle
        generationTask = Task { [weak self] in
            var buffer = ""
            var failure: Error?
            var lastFlush = ContinuousClock.now
            do {
                for try await token in backend.stream(messages: history, model: model) {
                    buffer += token
                    let now = ContinuousClock.now
                    if now - lastFlush >= throttle {
                        lastFlush = now
                        self?.streamingText = buffer
                    }
                }
            } catch {
                failure = error
            }
            self?.finishGeneration(chatID: chatID, content: buffer, failure: failure)
        }
    }

    public func stopGenerating() {
        generationTask?.cancel()
    }

    // MARK: - Internals

    private func finishGeneration(chatID: UUID, content: String, failure: Error?) {
        update(id: chatID) { chat in
            guard let last = chat.messages.indices.last,
                  chat.messages[last].role == .assistant
            else { return }
            if content.isEmpty, failure == nil {
                chat.messages.removeLast()
            } else {
                chat.messages[last].content = content
            }
        }
        if let failure {
            errorMessage = (failure as? LocalizedError)?.errorDescription
                ?? String(describing: failure)
        }
        streamingText = ""
        isGenerating = false
        generationTask = nil
        generatingChatID = nil
    }

    private func createChat(titledFrom prompt: String) -> UUID {
        let title = String(prompt.prefix(48))
        let chat = Chat(title: title)
        chats.insert(chat, at: 0)
        return chat.id
    }

    private func update(id: UUID, _ transform: (inout Chat) -> Void) {
        guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
        var chat = chats[index]
        transform(&chat)
        chats[index] = chat
    }
}
