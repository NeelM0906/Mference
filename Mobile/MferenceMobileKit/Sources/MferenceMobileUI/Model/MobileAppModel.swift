import Foundation
import Observation

/// The paired Mac as the phone sees it (same Wi-Fi only, by decision).
public struct PairedMac: Hashable, Sendable {
    public var name: String
    public var isOnSameWiFi: Bool
    public var models: [PhoneModel]

    public static let sample = PairedMac(name: "MacBook Pro", isOnSameWiFi: true,
                                         models: [.deep, .qwen36, .miniCPM, .maple])
}

/// The whole phone app's state. One instance per window scene, so folding or
/// unfolding a Duo keeps the chat, draft, calendar day and scan.
///
/// This first slice runs on sample data and `ScriptedReplies`; nothing here
/// loads a model or touches the network.
@Observable
public final class MobileAppModel {
    // MARK: Shell
    public var selection: MobileSection = .chat
    public var isFirstRunPresented: Bool

    // MARK: Chat
    public var conversations: [Conversation]
    /// nil = a new, empty chat (Chat opens to one, by decision).
    public var currentConversationID: UUID?
    public var draft = ""
    public var selectedModel: PhoneModel = .quick
    public var isModelPickerPresented = false
    public var searchText = ""

    // MARK: Settings
    /// "Web search via paired Mac". Off until the person turns it on.
    public var webSearchViaMac = false
    public var appleModelEnabled = true
    public var unloadAfterMinutes = 10
    public var pairedMac: PairedMac?
    public var memoryCount = 12

    // MARK: Calendar
    public var calendarMode: CalendarMode = .day
    public var selectedDay: Date
    public var events: [PhoneEvent]
    public var todos: [PhoneTodo]
    public var suggestion: CalendarSuggestion?

    // MARK: Scan
    public var scanKind: ScanKind = .receipt
    public var scanPhase: ScanPhase = .read
    public var scanFields: [ScanField]
    public var scanMerchant = "Blue Door Cafe"

    let calendar: Calendar
    let replies: ScriptedReplies
    /// Seconds between streamed words; tests set 0.
    let wordDelay: Duration

    public init(now: Date = .now, calendar: Calendar = .current, showFirstRun: Bool = false,
                pairedMac: PairedMac? = .sample, wordDelay: Duration = .milliseconds(28)) {
        self.calendar = calendar
        self.replies = ScriptedReplies()
        self.wordDelay = wordDelay
        self.isFirstRunPresented = showFirstRun
        self.pairedMac = pairedMac
        self.conversations = SampleContent.conversations()
        self.selectedDay = calendar.startOfDay(for: now)
        let planner = SampleContent.planner(today: now, calendar: calendar)
        self.events = planner.events
        self.todos = planner.todos
        self.suggestion = planner.suggestion
        self.scanFields = SampleContent.scanFields(today: now)
    }

    // MARK: - Chat

    public var currentConversation: Conversation? {
        conversations.first { $0.id == currentConversationID }
    }

    public var currentMessages: [ChatMessage] { currentConversation?.messages ?? [] }

    /// The nav title: the conversation's title or "New chat".
    public var chatTitle: String { currentConversation?.title ?? "New chat" }

    /// Models the picker offers on the Mac (none while unpaired or away).
    public var macModels: [PhoneModel] {
        guard let pairedMac, pairedMac.isOnSameWiFi else { return [] }
        return pairedMac.models
    }

    public func startNewChat() {
        currentConversationID = nil
        draft = ""
    }

    public func open(_ conversation: Conversation) {
        currentConversationID = conversation.id
    }

    public func conversations(in group: Conversation.ListGroup) -> [Conversation] {
        let query = searchText.trimmingCharacters(in: .whitespaces).lowercased()
        return conversations.filter { conversation in
            conversation.group == group && (query.isEmpty || conversation.title.lowercased().contains(query))
        }
    }

    /// Sends the draft. A request the Mac should take gets the hand-off card
    /// instead of an answer; anything else gets a streamed scripted reply.
    public func send(_ text: String? = nil) {
        let request = (text ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !request.isEmpty else { return }
        draft = ""
        let conversationID = ensureConversation(titledFrom: request)
        append(ChatMessage(role: .user, text: request), to: conversationID)

        if selectedModel.place == .thisIPhone, pairedMac?.isOnSameWiFi == true,
           let reason = HandoffAdvisor.reason(for: request) {
            append(ChatMessage(role: .assistant, text: "", handoff: HandoffOffer(reason: reason)),
                   to: conversationID)
            return
        }
        stream(replies.reply(to: request, webSearch: webSearchViaMac, model: selectedModel),
               into: conversationID)
    }

    /// The person chose on the hand-off card.
    public func resolveHandoff(_ messageID: UUID, sendToMac: Bool) {
        guard let conversationID = conversationContaining(messageID),
              let request = conversation(conversationID)?.messages.last(where: { $0.role == .user })?.text
        else { return }
        update(messageID, in: conversationID) { $0.handoff?.state = sendToMac ? .sentToMac : .keptOnIPhone }
        let model: PhoneModel = sendToMac ? (pairedMac?.models.first ?? .deep) : selectedModel
        if sendToMac {
            mutate(conversationID) { $0.isRunningOnMac = true }
        }
        stream(replies.reply(to: request, webSearch: webSearchViaMac, model: model), into: conversationID)
    }

    // MARK: - Calendar

    public func events(on day: Date) -> [PhoneEvent] {
        events.filter { calendar.isDate($0.start, inSameDayAs: day) }.sorted { $0.start < $1.start }
    }

    public func hasEvents(on day: Date) -> Bool { !events(on: day).isEmpty }

    public func select(day: Date, switchToDay: Bool = false) {
        selectedDay = calendar.startOfDay(for: day)
        if switchToDay { calendarMode = .day }
    }

    public func goToToday() { selectedDay = calendar.startOfDay(for: .now) }

    public func acceptSuggestion() {
        guard let suggestion else { return }
        if let index = events.firstIndex(where: { $0.id == suggestion.event.id }) {
            events[index] = suggestion.event
        }
        self.suggestion = nil
    }

    public func dismissSuggestion() { suggestion = nil }

    public func toggle(_ todo: PhoneTodo) {
        guard let index = todos.firstIndex(where: { $0.id == todo.id }) else { return }
        todos[index].isDone.toggle()
    }

    public func addTodo(_ title: String) {
        let trimmed = title.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        todos.insert(PhoneTodo(title: trimmed, detail: "Added on iPhone"), at: 0)
    }

    public var openTodoCount: Int { todos.filter { !$0.isDone }.count }

    // MARK: - Scan

    /// The shutter: read the sample receipt after a short pause.
    public func capture() {
        scanPhase = .reading
        Task {
            try? await Task.sleep(for: .milliseconds(900))
            if scanPhase == .reading { scanPhase = .read }
        }
    }

    public func retake() { scanPhase = .viewfinder }

    public func addScanRow() { scanPhase = .added }

    // MARK: - Private

    private func ensureConversation(titledFrom request: String) -> UUID {
        if let currentConversationID { return currentConversationID }
        let words = request.split(separator: " ").prefix(5).joined(separator: " ")
        let conversation = Conversation(title: words.isEmpty ? "New chat" : words, group: .today,
                                        updatedLabel: "Now")
        conversations.insert(conversation, at: 0)
        currentConversationID = conversation.id
        return conversation.id
    }

    private func stream(_ reply: ChatMessage, into conversationID: UUID) {
        var message = reply
        let fullText = message.text
        let fullClosing = message.closing
        message.text = ""
        message.closing = nil
        message.table = nil
        message.sources = []
        message.actions = []
        message.isStreaming = true
        append(message, to: conversationID)
        let id = message.id
        Task {
            var shown = ""
            for word in fullText.split(separator: " ", omittingEmptySubsequences: false) {
                shown += shown.isEmpty ? String(word) : " " + word
                update(id, in: conversationID) { $0.text = shown }
                if wordDelay > .zero { try? await Task.sleep(for: wordDelay) }
            }
            update(id, in: conversationID) {
                $0.table = reply.table
                $0.closing = fullClosing
                $0.sources = reply.sources
                $0.actions = reply.actions
                $0.isStreaming = false
            }
            mutate(conversationID) { conversation in
                conversation.isRunningOnMac = false
                for index in conversation.messages.indices where conversation.messages[index].handoff?.state == .sentToMac {
                    conversation.messages[index].handoff?.state = .returnedFromMac
                }
            }
        }
    }

    private func conversation(_ id: UUID) -> Conversation? { conversations.first { $0.id == id } }

    private func conversationContaining(_ messageID: UUID) -> UUID? {
        conversations.first { $0.messages.contains { $0.id == messageID } }?.id
    }

    private func append(_ message: ChatMessage, to conversationID: UUID) {
        mutate(conversationID) {
            $0.messages.append(message)
            $0.updatedLabel = "Now"
        }
    }

    private func update(_ messageID: UUID, in conversationID: UUID, _ change: (inout ChatMessage) -> Void) {
        mutate(conversationID) { conversation in
            guard let index = conversation.messages.firstIndex(where: { $0.id == messageID }) else { return }
            change(&conversation.messages[index])
        }
    }

    private func mutate(_ conversationID: UUID, _ change: (inout Conversation) -> Void) {
        guard let index = conversations.firstIndex(where: { $0.id == conversationID }) else { return }
        change(&conversations[index])
    }
}
