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
    /// The model a new chat starts with (Settings' default model), read each
    /// time a chat is started so a change applies to the next one.
    @ObservationIgnored private let defaultModel: @MainActor () -> AppModel
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
    public var errorMessage: String? {
        didSet { if errorMessage == nil { canRetryAfterError = false } }
    }
    /// The error came from writing an answer, so "Try again" regenerates it.
    public private(set) var canRetryAfterError = false
    public private(set) var modelOptions: [ModelOption] = AppModel.engineModels.map {
        ModelOption(model: $0, availability: .notInstalled(reason: "Looking for models…"))
    }
    public private(set) var engineStatus = EngineServiceStatus.empty
    /// The last answer in this chat left out its oldest messages to fit the
    /// model's context (Apple's model); the chat shows a notice.
    public private(set) var historyTrimmed = false
    public private(set) var rememberedCount = 0
    /// The model for the selected chat (or the next new one).
    public private(set) var model: AppModel = .quick
    public private(set) var webEnabled = false
    public private(set) var isPrivate = false
    public private(set) var title = Conversation.defaultTitle
    /// The person turned web on for the first time: ask before enabling (L7).
    public private(set) var webConsentRequested = false
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
                tools: (any ChatToolboxProvider)? = nil,
                defaultModel: @escaping @MainActor () -> AppModel = { .quick }) {
        self.service = service
        self.store = store
        self.tools = tools
        self.defaultModel = defaultModel
        model = defaultModel()
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
                self.applyOptions(await self.service.models())
            }
        }
        Task { [weak self] in
            guard let self else { return }
            self.applyOptions(await self.service.models())
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
    /// It uses the default model unless that one is not installed. Web
    /// search starts off: each chat opts in on its own (L7), so one chat's
    /// opt-in never carries into another, private or not.
    public func newChat(isPrivate: Bool = false) {
        selectedID = nil
        messages = []
        errorMessage = nil
        historyTrimmed = false
        let preferred = defaultModel()
        if case .notInstalled = option(for: preferred)?.availability {} else { model = preferred }
        webEnabled = false
        self.isPrivate = isPrivate
        title = Conversation.defaultTitle
    }

    public func select(_ id: UUID) {
        guard let conversation = try? store.conversation(id: id) else { return }
        selectedID = id
        errorMessage = nil
        historyTrimmed = false
        model = conversation.model
        fallBackIfModelHidden()
        webEnabled = conversation.webEnabled
        isPrivate = conversation.isPrivate
        title = conversation.title
        messages = ChatMessageItem.items(from: conversation.orderedMessages)
        warmUp(model)
    }

    /// Deletes a chat. If its answer is being written, that answer is stopped
    /// and finishes saving first, so nothing writes to a deleted chat.
    public func delete(_ id: UUID) async {
        if generatingConversationID == id {
            stop()
            await waitUntilIdle()
        }
        guard let conversation = try? store.conversation(id: id) else { return }
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
        guard option(for: model) != nil || model.isEngineModel else { return }
        historyTrimmed = false
        self.model = model
        if let conversation = selectedConversation {
            try? store.update(conversation) { $0.model = model }
        }
        warmUp(model)
    }

    /// Why the Web toggle cannot turn on now (offline mode, or Apple's model,
    /// which answers without tools), if it cannot.
    public var webBlockedReason: String? {
        if model == .apple, tools != nil { return Self.appleWebReason }
        return tools?.webUnavailableReason
    }

    /// The Web chip's text while Apple's model is chosen.
    public static let appleWebReason = "Web needs Quick or Deep"

    /// Web is off because of the chosen model, not offline mode.
    public var webBlockedByModel: Bool { model == .apple && tools != nil }

    /// The model the trimmed-chat notice offers: Deep where it can run, else
    /// Quick; nil when neither can.
    public var longChatModel: AppModel? {
        [AppModel.deep, .quick].first { candidate in
            guard let availability = option(for: candidate)?.availability else { return false }
            if case .notInstalled = availability { return false }
            return true
        }
    }

    /// Re-reads the model list (the picker opening): Apple Intelligence may
    /// have been turned on since.
    public func refreshModels() {
        Task { [weak self] in
            guard let self else { return }
            self.applyOptions(await self.service.models())
        }
    }

    /// Takes the model list from the service. A chat on Apple's model whose
    /// row is now hidden (the Settings switch went off) moves to the default
    /// model.
    private func applyOptions(_ options: [ModelOption]) {
        modelOptions = options
        hasLoadedOptions = true
        fallBackIfModelHidden()
    }

    @ObservationIgnored private var hasLoadedOptions = false

    private func fallBackIfModelHidden() {
        guard hasLoadedOptions, !model.isEngineModel, option(for: model) == nil else { return }
        historyTrimmed = false
        let preferred = defaultModel()
        model = preferred.isEngineModel ? preferred : .quick
        // Saved at once, even mid-answer: the running answer keeps the model
        // it started with, and the next send must not reach Apple's model.
        if let conversation = selectedConversation {
            let fallback = model
            try? store.update(conversation) { $0.model = fallback }
        }
    }

    public func setWebEnabled(_ enabled: Bool) {
        if enabled, webBlockedReason != nil { return }
        if enabled, tools?.needsWebConsent == true {
            webConsentRequested = true
            return
        }
        webEnabled = enabled
        if let conversation = selectedConversation {
            try? store.update(conversation) { $0.webEnabled = enabled }
        }
        tools?.webUsageChanged(enabled: enabled)
    }

    /// "Turn on" in the web consent prompt.
    public func confirmWebConsent() {
        webConsentRequested = false
        tools?.grantWebConsent()
        setWebEnabled(true)
    }

    /// "Not now" in the web consent prompt.
    public func declineWebConsent() {
        webConsentRequested = false
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
        // Apple's model answers without tools (web needs Quick or Deep).
        let toolbox = model.isEngineModel ? tools?.makeToolbox(webEnabled: conversation.webEnabled) : nil
        historyTrimmed = false
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
        // Calls the model made that have no tool result yet; a stop answers
        // them with "Stopped." so the history stays well formed.
        var unanswered: [ToolCall] = []
        defer { finishGeneration(conversationID: conversation.id) }

        do {
            // What the person asked Mference to remember (Settings > Memory),
            // read once per answer.
            let memory = ChatMemoryPrompt.text(facts: store.rememberedFacts())
            while true {
                var turns = store.turns(for: conversation)
                let system = [toolbox?.systemPrompt(now: now()), memory].compactMap { $0 }
                if !system.isEmpty {
                    turns.insert(ChatTurn(role: .system, content: system.joined(separator: "\n\n")), at: 0)
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
                    unanswered = summary.toolCalls
                    for call in summary.toolCalls {
                        try Task.checkCancellation()
                        let outcome = await toolbox.run(call)
                        try store.append(to: conversation, role: .tool, content: outcome.content, now: now()) {
                            $0.toolCallID = call.id
                            $0.toolName = call.name
                        }
                        unanswered.removeFirst()
                        steps += outcome.steps
                        Self.merge(outcome.citations, into: &citations)
                        updateStreaming { $0.steps = steps; $0.citations = citations }
                    }
                    try Task.checkCancellation()
                    round += 1
                    updateStreaming { $0.content = ""; $0.reasoning = nil }
                    continue
                }

                let sources = Self.cited(citations, in: summary.content)
                let saved = try store.append(to: conversation, role: .assistant, content: summary.content,
                                             reasoning: summary.reasoning, model: model, now: now()) {
                    $0.steps = steps
                    $0.citations = sources
                    $0.finishRawValue = ChatFinish(summary.finish).rawValue
                    $0.promptTokens = summary.promptTokens
                    $0.completionTokens = summary.completionTokens
                }
                appendFinished(saved, conversationID: conversation.id)
                if summary.historyTrimmed, selectedID == conversation.id { historyTrimmed = true }
                return
            }
        } catch {
            let cancelled = error is CancellationError || Task.isCancelled
            answerUnanswered(unanswered, in: conversation, with: cancelled ? "Stopped." : "Failed.")
            savePartial(conversation: conversation, model: model, content: content, reasoning: reasoning,
                        steps: steps, citations: citations, finish: cancelled ? .cancelled : .failed)
            if !cancelled, selectedID == conversation.id {
                errorMessage = Self.message(for: error)
                canRetryAfterError = true
            }
        }
    }

    private func answerUnanswered(_ calls: [ToolCall], in conversation: Conversation, with text: String) {
        for call in calls {
            _ = try? store.append(to: conversation, role: .tool, content: text, now: now()) {
                $0.toolCallID = call.id
                $0.toolName = call.name
            }
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

    /// The sources an answer shows: those it cites, in number order. Numbers
    /// stay as the model saw them, so "[3]" still opens source 3.
    static func cited(_ citations: [ChatCitation], in answer: String) -> [ChatCitation] {
        let valid = Set(citations.map(\.number))
        var used: Set<Int> = []
        for segment in CitationMarkers.segments(answer, valid: valid) {
            if case .citation(let number) = segment { used.insert(number) }
        }
        return citations.filter { used.contains($0.number) }
    }

    /// The sentence the chat shows for an error.
    public static func message(for error: any Error) -> String {
        if let error = error as? EngineServiceError { return error.userMessage }
        if let error = error as? ChatError { return error.userMessage }
        if let error = error as? AssistantModelError { return error.userMessage }
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

/// The system-prompt section that carries the person's enabled memories.
public enum ChatMemoryPrompt {
    /// Facts beyond these limits are left out, newest changes kept, so
    /// memory never crowds out the conversation.
    public static let maxFacts = 50
    public static let maxCharacters = 4_000

    /// Nil when nothing is remembered.
    public static func text(facts: [String]) -> String? {
        var lines: [String] = []
        var used = 0
        for fact in facts.prefix(maxFacts) {
            let line = "- " + fact.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.count > 2, used + line.count <= maxCharacters else { continue }
            lines.append(line)
            used += line.count + 1
        }
        guard !lines.isEmpty else { return nil }
        return "Things the person asked you to remember about them. Use them when they are relevant; "
            + "do not mention this list unless asked.\n" + lines.joined(separator: "\n")
    }
}
