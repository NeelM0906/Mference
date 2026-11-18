import Foundation
import Observation

/// One message as the chat shows it. Tool messages and the assistant turns
/// that only called tools are history for the model, not rows on screen;
/// their steps and sources appear on the answer that follows them.
public struct ChatMessageItem: Sendable, Hashable, Identifiable {
    public enum Role: Sendable, Hashable { case user, assistant }

    public var id: UUID
    public var role: Role
    public var content: String
    public var reasoning: String?
    public var steps: [ChatStep]
    public var citations: [ChatCitation]
    public var model: AppModel?
    public var finish: ChatFinish?
    public var isStreaming: Bool

    public init(id: UUID = UUID(), role: Role, content: String, reasoning: String? = nil,
                steps: [ChatStep] = [], citations: [ChatCitation] = [], model: AppModel? = nil,
                finish: ChatFinish? = nil, isStreaming: Bool = false) {
        self.id = id
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.steps = steps
        self.citations = citations
        self.model = model
        self.finish = finish
        self.isStreaming = isStreaming
    }

    /// The header of the steps disclosure, if there is anything to disclose.
    public var stepsTitle: String? {
        ChatStepsSummary.title(steps: steps, hasReasoning: !(reasoning ?? "").isEmpty)
    }

    /// Display rows for stored messages, in order.
    public static func items(from messages: [StoredMessage]) -> [ChatMessageItem] {
        messages.compactMap { message in
            switch message.role {
            case .user:
                return ChatMessageItem(id: message.id, role: .user, content: message.content)
            case .assistant where message.toolCalls.isEmpty || message.finish == .cancelled
                || message.finish == .failed:
                return ChatMessageItem(id: message.id, role: .assistant, content: message.content,
                                       reasoning: message.reasoning, steps: message.steps,
                                       citations: message.citations, model: message.model,
                                       finish: message.finish)
            default:
                return nil
            }
        }
    }
}

/// The chat screen's state and actions (spec 4.3). Streams through
/// `EngineServicing`, batching token updates so SwiftUI re-renders about
/// every `flushInterval` rather than per token (spec 4.3.2).
@MainActor
@Observable
public final class ChatViewModel {
    // MARK: Dependencies

    @ObservationIgnored public let service: any EngineServicing
    @ObservationIgnored public let store: ConversationStore
    @ObservationIgnored private let tools: (any ChatToolboxProvider)?
    /// How often streamed text reaches the view.
    @ObservationIgnored public var flushInterval: Duration = .milliseconds(30)
    /// Tool rounds per answer before the model must answer without tools.
    @ObservationIgnored public var maxToolRounds = 6
    @ObservationIgnored public var now: @MainActor () -> Date = { .now }

    // MARK: State

    public private(set) var groups = ConversationGroups(rows: [])
    public private(set) var selectedID: UUID?
    public private(set) var messages: [ChatMessageItem] = []
    /// The answer being written (in whichever chat started it).
    public private(set) var streaming: ChatMessageItem?
    /// The chat the answer is being written in.
    public private(set) var generatingConversationID: UUID?
    /// The answer being written, if it belongs to the chat on screen.
    public var visibleStreaming: ChatMessageItem? {
        guard let selectedID, selectedID == generatingConversationID else { return nil }
        return streaming
    }
    public var draft = ""
    public private(set) var isGenerating = false
    public var errorMessage: String?
    public private(set) var modelOptions: [ModelOption] = AppModel.allCases.map {
        ModelOption(model: $0, availability: .notInstalled(reason: "Looking for models…"))
    }
    public private(set) var engineStatus = EngineServiceStatus.empty
    public private(set) var rememberedCount = 0
    /// The model for the selected chat (or the next new one).
    public private(set) var model: AppModel = .quick
    public private(set) var webEnabled = false
    public private(set) var isPrivate = false
    public private(set) var title = Conversation.defaultTitle
    /// How many times streamed text was pushed to the view (tests check the
    /// batching with it).
    @ObservationIgnored public private(set) var flushCount = 0

    // MARK: Private

    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var statusTask: Task<Void, Never>?
    @ObservationIgnored private var flushTask: Task<Void, Never>?
    @ObservationIgnored private var pendingContent = ""
    @ObservationIgnored private var pendingReasoning = ""

    public init(service: any EngineServicing, store: ConversationStore,
                tools: (any ChatToolboxProvider)? = nil) {
        self.service = service
        self.store = store
        self.tools = tools
    }

    /// Loads the sidebar and follows engine status. Call once when the
    /// screen appears.
    public func start() {
        reloadConversations()
        rememberedCount = store.rememberedCount()
        guard statusTask == nil else { return }
        let updates = service.statusUpdates()
        statusTask = Task { [weak self] in
            for await status in updates {
                guard let self else { return }
                self.engineStatus = status
                self.modelOptions = await self.service.models()
            }
        }
        Task { [weak self] in
            guard let self else { return }
            self.modelOptions = await self.service.models()
        }
    }

    /// Stops following status; the generation, if any, is stopped too.
    public func stopObserving() {
        statusTask?.cancel()
        statusTask = nil
        stop()
    }

    // MARK: Conversations

    public var selectedConversation: Conversation? {
        selectedID.flatMap { try? store.conversation(id: $0) }
    }

    public func reloadConversations() {
        let rows = ((try? store.conversations()) ?? []).map {
            ConversationGroups.Row(id: $0.id, title: $0.title, updatedAt: $0.updatedAt, isPrivate: $0.isPrivate)
        }
        groups = ConversationGroups(rows: rows, now: now())
    }

    /// Starts an empty chat; it is stored when its first message is sent.
    public func newChat(isPrivate: Bool = false) {
        selectedID = nil
        messages = []
        errorMessage = nil
        self.isPrivate = isPrivate
        title = Conversation.defaultTitle
    }

    public func select(_ id: UUID) {
        guard let conversation = try? store.conversation(id: id) else { return }
        selectedID = id
        errorMessage = nil
        model = conversation.model
        webEnabled = conversation.webEnabled
        isPrivate = conversation.isPrivate
        title = conversation.title
        messages = ChatMessageItem.items(from: conversation.orderedMessages)
        warmUp(model)
    }

    public func delete(_ id: UUID) {
        guard let conversation = try? store.conversation(id: id) else { return }
        if generatingConversationID == id { stop() }
        try? store.delete(conversation)
        if selectedID == id { newChat() }
        reloadConversations()
    }

    // MARK: Model and web

    public func option(for model: AppModel) -> ModelOption? {
        modelOptions.first { $0.model == model }
    }

    public func selectModel(_ model: AppModel) {
        if case .notInstalled = option(for: model)?.availability { return }
        self.model = model
        if let conversation = selectedConversation {
            try? store.update(conversation) { $0.model = model }
        }
        warmUp(model)
    }

    public func setWebEnabled(_ enabled: Bool) {
        webEnabled = enabled
        if let conversation = selectedConversation {
            try? store.update(conversation) { $0.webEnabled = enabled }
        }
        tools?.webUsageChanged(enabled: enabled)
    }

    public func setPrivate(_ value: Bool) {
        isPrivate = value
        if let conversation = selectedConversation {
            try? store.update(conversation) { $0.isPrivate = value }
            reloadConversations()
        }
    }

    /// Makes the model resident when a chat opens with it (spec 4.3.1).
    private func warmUp(_ model: AppModel) {
        guard !isGenerating, option(for: model)?.availability == .ready else { return }
        let service = self.service
        Task { [weak self] in
            do { try await service.load(model) } catch {
                self?.errorMessage = Self.message(for: error)
            }
        }
    }

    // MARK: Sending

    public var canSend: Bool {
        !isGenerating && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isGenerating, !text.isEmpty else { return }
        errorMessage = nil
        do {
            let conversation = try selectedConversation
                ?? store.createConversation(model: model, webEnabled: webEnabled, isPrivate: isPrivate, now: now())
            if selectedID != conversation.id {
                selectedID = conversation.id
            }
            let message = try store.append(to: conversation, role: .user, content: text, now: now())
            draft = ""
            title = conversation.title
            messages.append(ChatMessageItem(id: message.id, role: .user, content: text))
            reloadConversations()
            answer(in: conversation)
        } catch {
            errorMessage = "Mference could not save this message. \(error.localizedDescription)"
        }
    }

    /// Writes the last answer again.
    public func regenerate() {
        guard !isGenerating, let conversation = selectedConversation else { return }
        let ordered = conversation.orderedMessages
        guard let lastUser = ordered.last(where: { $0.role == .user }) else { return }
        errorMessage = nil
        do {
            try store.deleteMessages(in: conversation, from: lastUser.index + 1)
        } catch {
            errorMessage = "Mference could not change this chat. \(error.localizedDescription)"
            return
        }
        messages = ChatMessageItem.items(from: conversation.orderedMessages)
        answer(in: conversation)
    }

    /// Stops the answer being written; what was written so far is kept.
    public func stop() {
        generationTask?.cancel()
    }

    /// Waits for the running answer, if any (tests, quit).
    public func waitUntilIdle() async {
        await generationTask?.value
    }

    // MARK: Generation

    private func answer(in conversation: Conversation) {
        let model = conversation.model
        let toolbox = tools?.makeToolbox(webEnabled: conversation.webEnabled)
        isGenerating = true
        generatingConversationID = conversation.id
        streaming = ChatMessageItem(role: .assistant, content: "", model: model, isStreaming: true)
        pendingContent = ""
        pendingReasoning = ""
        generationTask = Task { [weak self] in
            await self?.runAnswer(conversation: conversation, model: model, toolbox: toolbox)
        }
    }

    private func runAnswer(conversation: Conversation, model: AppModel, toolbox: (any ChatToolbox)?) async {
        var steps: [ChatStep] = []
        var citations: [ChatCitation] = []
        var round = 0
        var content = ""
        var reasoning = ""
        defer { finishGeneration(conversationID: conversation.id) }

        do {
            while true {
                var turns = store.turns(for: conversation)
                if let prompt = toolbox?.systemPrompt(now: now()) {
                    turns.insert(ChatTurn(role: .system, content: prompt), at: 0)
                }
                let offerTools = toolbox != nil && round < maxToolRounds
                let options = GenerationOptions(tools: offerTools ? toolbox?.definitions ?? [] : [])
                content = ""
                reasoning = ""
                var summary: GenerationSummary?
                for try await update in service.generate(model: model, turns: turns, options: options) {
                    switch update {
                    case .prepared, .toolCall:
                        break
                    case .content(let text):
                        content += text
                        enqueue(content: text)
                    case .reasoning(let text):
                        reasoning += text
                        enqueue(reasoning: text)
                    case .completed(let done):
                        summary = done
                    }
                }
                try Task.checkCancellation()
                guard let summary else { throw ChatError.endedEarly }
                flushNow()

                if offerTools, let toolbox, summary.finish == .toolCalls, !summary.toolCalls.isEmpty {
                    try store.append(to: conversation, role: .assistant, content: summary.content,
                                     reasoning: summary.reasoning, model: model, now: now()) {
                        $0.toolCalls = summary.toolCalls
                        $0.finishRawValue = ChatFinish.toolCalls.rawValue
                    }
                    for call in summary.toolCalls {
                        let outcome = await toolbox.run(call)
                        try Task.checkCancellation()
                        steps += outcome.steps
                        Self.merge(outcome.citations, into: &citations)
                        try store.append(to: conversation, role: .tool, content: outcome.content, now: now()) {
                            $0.toolCallID = call.id
                            $0.toolName = call.name
                        }
                        updateStreaming { $0.steps = steps; $0.citations = citations }
                    }
                    round += 1
                    updateStreaming { $0.content = ""; $0.reasoning = nil }
                    continue
                }

                let saved = try store.append(to: conversation, role: .assistant, content: summary.content,
                                             reasoning: summary.reasoning, model: model, now: now()) {
                    $0.steps = steps
                    $0.citations = citations
                    $0.finishRawValue = ChatFinish(summary.finish).rawValue
                    $0.promptTokens = summary.promptTokens
                    $0.completionTokens = summary.completionTokens
                }
                appendFinished(saved, conversationID: conversation.id)
                return
            }
        } catch is CancellationError {
            savePartial(conversation: conversation, model: model, content: content, reasoning: reasoning,
                        steps: steps, citations: citations, finish: .cancelled)
        } catch {
            if Task.isCancelled {
                savePartial(conversation: conversation, model: model, content: content, reasoning: reasoning,
                            steps: steps, citations: citations, finish: .cancelled)
                return
            }
            savePartial(conversation: conversation, model: model, content: content, reasoning: reasoning,
                        steps: steps, citations: citations, finish: .failed)
            if selectedID == conversation.id { errorMessage = Self.message(for: error) }
        }
    }

    private func savePartial(conversation: Conversation, model: AppModel, content: String, reasoning: String,
                             steps: [ChatStep], citations: [ChatCitation], finish: ChatFinish) {
        flushNow()
        guard !content.isEmpty || !reasoning.isEmpty || !steps.isEmpty else { return }
        guard let saved = try? store.append(to: conversation, role: .assistant, content: content,
                                            reasoning: reasoning, model: model, now: now(), configure: {
                                                $0.steps = steps
                                                $0.citations = citations
                                                $0.finishRawValue = finish.rawValue
                                            }) else { return }
        appendFinished(saved, conversationID: conversation.id)
    }

    private func appendFinished(_ message: StoredMessage, conversationID: UUID) {
        guard selectedID == conversationID, let item = ChatMessageItem.items(from: [message]).first else { return }
        messages.append(item)
    }

    private func finishGeneration(conversationID: UUID) {
        flushTask?.cancel()
        flushTask = nil
        pendingContent = ""
        pendingReasoning = ""
        streaming = nil
        isGenerating = false
        generatingConversationID = nil
        generationTask = nil
        reloadConversations()
    }

    // MARK: Batching

    private func enqueue(content: String = "", reasoning: String = "") {
        pendingContent += content
        pendingReasoning += reasoning
        guard flushTask == nil else { return }
        let interval = flushInterval
        flushTask = Task { [weak self] in
            try? await Task.sleep(for: interval)
            guard !Task.isCancelled else { return }
            self?.flushTask = nil
            self?.flushNow()
        }
    }

    private func flushNow() {
        flushTask?.cancel()
        flushTask = nil
        guard !pendingContent.isEmpty || !pendingReasoning.isEmpty else { return }
        let content = pendingContent
        let reasoning = pendingReasoning
        pendingContent = ""
        pendingReasoning = ""
        flushCount += 1
        updateStreaming { item in
            item.content += content
            if !reasoning.isEmpty { item.reasoning = (item.reasoning ?? "") + reasoning }
        }
    }

    private func updateStreaming(_ change: (inout ChatMessageItem) -> Void) {
        guard var item = streaming else { return }
        change(&item)
        streaming = item
    }

    // MARK: Helpers

    static func merge(_ new: [ChatCitation], into citations: inout [ChatCitation]) {
        for citation in new where !citations.contains(where: { $0.number == citation.number }) {
            citations.append(citation)
        }
        citations.sort { $0.number < $1.number }
    }

    /// The sentence the chat shows for an error.
    public static func message(for error: any Error) -> String {
        if let error = error as? EngineServiceError { return error.userMessage }
        if let error = error as? ChatError { return error.userMessage }
        return "Something went wrong while answering. \(error.localizedDescription)"
    }
}

enum ChatError: Error {
    case endedEarly

    var userMessage: String {
        switch self {
        case .endedEarly: "The answer ended before it was finished. Try again."
        }
    }
}
