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
/// Calendar and Scan come in as their screen models (`CalendarScreenModel`,
/// `ScanScreenModel`), live or sample, chosen by whoever builds the app.
/// Chat still runs on `ScriptedReplies` until its own screen model lands;
/// nothing here loads a model or touches the network.
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

    // MARK: Calendar and Scan
    public let calendar: any CalendarScreenModel
    public let scan: any ScanScreenModel

    let replies: ScriptedReplies
    /// Seconds between streamed words; tests set 0.
    let wordDelay: Duration

    public init(calendar: any CalendarScreenModel, scan: any ScanScreenModel, showFirstRun: Bool = false,
                pairedMac: PairedMac? = .sample, wordDelay: Duration = .milliseconds(28)) {
        self.calendar = calendar
        self.scan = scan
        self.replies = ScriptedReplies()
        self.wordDelay = wordDelay
        self.isFirstRunPresented = showFirstRun
        self.pairedMac = pairedMac
        self.conversations = SampleContent.conversations()
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
              let request = request(answeredBy: messageID, in: conversationID)
        else { return }
        update(messageID, in: conversationID) { $0.handoff?.state = sendToMac ? .sentToMac : .keptOnIPhone }
        let model: PhoneModel = sendToMac ? (pairedMac?.models.first ?? .deep) : selectedModel
        if sendToMac {
            mutate(conversationID) { $0.isRunningOnMac = true }
        }
        stream(replies.reply(to: request, webSearch: webSearchViaMac, model: model), into: conversationID)
    }

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

    /// The user message the given answer (or hand-off card) replies to: the
    /// nearest user message before it, not the newest one, since the person
    /// may keep chatting while an offer waits.
    private func request(answeredBy messageID: UUID, in conversationID: UUID) -> String? {
        guard let messages = conversation(conversationID)?.messages,
              let index = messages.firstIndex(where: { $0.id == messageID })
        else { return nil }
        return messages[..<index].last(where: { $0.role == .user })?.text
    }

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
