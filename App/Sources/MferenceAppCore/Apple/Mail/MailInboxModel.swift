import Foundation
import Observation
import SwiftData

/// The Mail screen's state (mockup 03): buckets, the list, the open message,
/// its event card and the reply draft.
@MainActor @Observable
public final class MailInboxModel {
    /// What the left column selects.
    public enum Folder: Hashable, Sendable {
        case bucket(String)
        case done
        case drafts
        case sent
    }

    public enum DraftState: Sendable, Hashable {
        case idle
        case writing
        case ready
        case sending
        case sent
        case failed(String)
    }

    public private(set) var status: MailStoreStatus?
    public private(set) var buckets: [MailBucketDefaults.Bucket] = []
    public var folder: Folder = .bucket(MailBucketDefaults.all[0].key)
    public private(set) var accounts: [MailAccount] = []
    public private(set) var selected: MailMessageSummary?
    public private(set) var body: MailBody?
    public private(set) var eventCard: ProposedEvent?
    public var draft: String = ""
    public private(set) var draftState: DraftState = .idle
    public private(set) var draftModelLabel: String?
    public private(set) var isSorting = false
    /// Some mail is unsorted because Apple's model is off; the list offers
    /// "Sort with Quick".
    public private(set) var needsManualSort = false
    /// A text field is being edited (draft or bucket names): E and Z type
    /// letters then, they do not mark Done or undo.
    public var isEditingDraft = false
    public var isRenamingBuckets = false
    public var isEditingText: Bool { isEditingDraft || isRenamingBuckets || isComposing }
    public private(set) var errorMessage: String?

    private var inbox: [MailMessageSummary] = []
    private var drafts: [MailMessageSummary] = []
    private var sent: [MailMessageSummary] = []
    private var assignment: [String: String] = [:]
    @ObservationIgnored private var lastMoved: (messageID: String, previous: String?)?
    @ObservationIgnored private var sendToken = 0
    /// Replies already sent from this window, so reopening one shows it sent.
    @ObservationIgnored private var sentMessageIDs: Set<String> = []

    private let source: any MailSource
    private let store: MailBucketStore
    private let assistant: any AppleModelServicing
    private let drafter: MailReplyDrafter?
    private let sender: any MailSending
    private let broker: any PermissionBroker
    private let onEventSuggested: @MainActor (ProposedEvent) -> Void
    private let timeZone: TimeZone

    public init(source: any MailSource, context: ModelContext, assistant: any AppleModelServicing,
                drafter: MailReplyDrafter?, sender: any MailSending, broker: any PermissionBroker,
                timeZone: TimeZone = .current,
                onEventSuggested: @escaping @MainActor (ProposedEvent) -> Void = { _ in }) {
        self.source = source
        self.store = MailBucketStore(context: context)
        self.assistant = assistant
        self.drafter = drafter
        self.sender = sender
        self.broker = broker
        self.timeZone = timeZone
        self.onEventSuggested = onEventSuggested
    }

    // MARK: Loading

    /// Reads without asking; the screen shows the reason and a button when
    /// Full Disk Access is missing.
    public func load(limit: Int = 300) async {
        await source.reload()
        let status = await source.status()
        self.status = status
        refreshBuckets()
        guard case .available = status else { return }
        do {
            async let inboxMessages = source.messages(in: [.inbox], limit: limit)
            async let draftMessages = source.messages(in: [.drafts], limit: 50)
            async let sentMessages = source.messages(in: [.sent], limit: 50)
            async let accountList = source.accounts()
            (inbox, drafts, sent, accounts) = try await (inboxMessages, draftMessages, sentMessages, accountList)
            let stored = try store.assignments(for: inbox.map(\.messageID))
            assignment = stored.mapValues(\.bucketKey)
            errorMessage = nil
        } catch {
            errorMessage = "Mail could not be read: \(error)"
        }
        // Show the newest message first, then sort in the background with
        // Apple's model only: Quick would replace the person's chat model,
        // so sorting with it waits for a click (`sortWithQuick`).
        if selected == nil, let first = inbox.first { await open(first) }
        await sortUnassigned(allowFallback: false)
    }

    /// The person clicked "Sort with Quick" (Apple Intelligence is off).
    public func sortWithQuick() async {
        await sortUnassigned(allowFallback: true)
    }

    /// The person clicked the button under the reason: the broker opens the
    /// Full Disk Access pane (it cannot be asked in a dialog).
    public func requestAccess() async {
        let state = await broker.request(.fullDiskAccess)
        if state != .granted { await broker.openSystemSettings(for: .fullDiskAccess) }
        await load()
    }

    private func refreshBuckets() {
        do {
            let stored = try store.buckets()
            buckets = stored.map { model in
                let fixed = MailBucketDefaults.all.first { $0.key == model.key }
                return MailBucketDefaults.Bucket(key: model.key, name: model.name, description: fixed?.description ?? "")
            }
        } catch {
            buckets = MailBucketDefaults.all
        }
    }

    /// Sorts messages no one placed yet. The person's own moves stick.
    public func sortUnassigned(allowFallback: Bool) async {
        let unsorted = inbox.filter { assignment[$0.messageID] == nil }
        guard !unsorted.isEmpty else { needsManualSort = false; return }
        isSorting = true
        defer { isSorting = false }
        let items = unsorted.map {
            MailSortItem(messageID: $0.messageID, sender: $0.senderName.isEmpty ? $0.senderAddress : $0.senderName,
                         subject: $0.subject, snippet: $0.snippet)
        }
        let choices = MailBucketDefaults.all.map { MailSortBucket(key: $0.key, description: $0.description) }
        let result: AssistantResult<[String: String]>
        do {
            result = try await assistant.sortMail(items, into: choices, allowFallback: allowFallback)
        } catch {
            // Apple's model is off: offer Quick on a click instead.
            needsManualSort = !allowFallback
            return
        }
        needsManualSort = false
        for (messageID, key) in result.value {
            // The person may have moved it while the sorter ran: theirs wins.
            guard assignment[messageID] == nil else { continue }
            assignment[messageID] = key
            try? store.assign(messageID, to: key, manual: false)
        }
    }

    // MARK: Lists

    public var messages: [MailMessageSummary] { messages(in: folder) }

    public func messages(in folder: Folder) -> [MailMessageSummary] {
        switch folder {
        case .bucket(let key):
            let fallback = MailBucketDefaults.all.last?.key
            return inbox.filter { (assignment[$0.messageID] ?? fallback) == key }
        case .done: return inbox.filter { assignment[$0.messageID] == MailBucketDefaults.doneKey }
        case .drafts: return drafts
        case .sent: return sent
        }
    }

    public func count(in folder: Folder) -> Int { messages(in: folder).count }

    public var folderTitle: String {
        switch folder {
        case .bucket(let key): buckets.first { $0.key == key }?.name ?? key
        case .done: "Done"
        case .drafts: "Drafts"
        case .sent: "Sent"
        }
    }

    public func select(_ folder: Folder) async {
        self.folder = folder
        if let first = messages.first { await open(first) } else { clearSelection() }
    }

    public func rename(bucket key: String, to name: String) {
        do { try store.rename(key: key, to: name) } catch { errorMessage = "The bucket could not be renamed." }
        refreshBuckets()
    }

    /// The person moved a message (sticks), or marked it done (E).
    public func move(_ message: MailMessageSummary, to key: String) {
        lastMoved = (message.messageID, assignment[message.messageID])
        assignment[message.messageID] = key
        try? store.assign(message.messageID, to: key, manual: true)
        if selected?.messageID == message.messageID { clearSelection() }
    }

    public func markDone(_ message: MailMessageSummary) {
        move(message, to: MailBucketDefaults.doneKey)
    }

    /// Z: undoes the last move.
    public func undo() {
        guard let (messageID, previous) = lastMoved else { return }
        lastMoved = nil
        if let previous {
            assignment[messageID] = previous
            try? store.assign(messageID, to: previous, manual: true)
        } else {
            assignment[messageID] = nil
            try? store.removeAssignment(messageID)
        }
    }

    // MARK: Reading

    public func open(_ message: MailMessageSummary) async {
        isEditingDraft = false
        selected = message
        body = nil
        eventCard = nil
        draft = ""
        sendToken += 1
        draftState = sentMessageIDs.contains(message.messageID) ? .sent : .idle
        do {
            let body = try await source.body(of: message)
            guard selected?.messageID == message.messageID else { return }
            self.body = body
            await suggestEvent(for: message, body: body)
        } catch {
            guard selected?.messageID == message.messageID else { return }
            self.body = MailBody(text: message.snippet)
        }
    }

    private func clearSelection() {
        selected = nil
        body = nil
        eventCard = nil
        draft = ""
        draftState = .idle
    }

    private func suggestEvent(for message: MailMessageSummary, body: MailBody) async {
        // Automatic: Apple's model only, never Quick over the chat model.
        guard let result = try? await assistant.suggestEvent(subject: message.subject, body: body.text,
                                                              received: message.date, timeZone: timeZone,
                                                              allowFallback: false),
              let suggestion = result.value, selected?.messageID == message.messageID else { return }
        let name = message.senderName.split(separator: " ").first.map(String.init) ?? message.senderAddress
        eventCard = ProposedEvent(title: suggestion.title, start: suggestion.start, end: suggestion.end,
                                  source: ItemSource(kind: .mail, label: message.senderName.isEmpty
                                                     ? message.senderAddress : message.senderName,
                                                     reference: message.messageID),
                                  note: "From \(name)'s email. Nothing changes until you add it.")
    }

    /// "Add to calendar" hands the card to the Calendar screen's proposals.
    public func addEventCardToCalendar() {
        guard let eventCard else { return }
        onEventSuggested(eventCard)
        self.eventCard = nil
    }

    // MARK: Draft and send

    /// Replies are offered for received mail only, not for Drafts or Sent.
    public var canReply: Bool {
        guard let selected else { return false }
        return selected.mailbox.kind != .drafts && selected.mailbox.kind != .sent
    }

    /// Who the reply goes to and from, and its subject, for the draft card.
    public var replyDraft: MailDraft? {
        guard let message = selected, canReply else { return nil }
        return MailDraft.reply(to: message, headers: body?.headers ?? [:], accountAddresses: accounts.map(\.name),
                               body: draft)
    }

    /// Send is possible only for a finished draft that is not being
    /// rewritten, sent or already sent.
    public var canSend: Bool {
        guard canReply, !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        switch draftState {
        case .writing, .sending, .sent: return false
        case .idle, .ready, .failed: return true
        }
    }

    public func writeDraft(tone: DraftTone = .standard) async {
        guard canReply, draftState != .writing, draftState != .sending, draftState != .sent else { return }
        guard let message = selected, let drafter else {
            draftState = .failed("Quick is needed to write drafts. Install it from the model library.")
            return
        }
        let previous = tone == .standard ? nil : draft
        draftState = .writing
        do {
            let text = try await drafter.draft(replyTo: message, body: body?.text ?? message.snippet, tone: tone,
                                               previous: previous)
            // A different message, or a draft that changed state meanwhile,
            // keeps what it has.
            guard selected?.messageID == message.messageID, draftState == .writing else { return }
            draft = text
            draftModelLabel = drafter.model.shortName
            draftState = .ready
        } catch {
            guard selected?.messageID == message.messageID, draftState == .writing else { return }
            switch error {
            case let error as EngineServiceError: draftState = .failed(error.userMessage)
            case let error as AssistantModelError: draftState = .failed(error.userMessage)
            default: draftState = .failed("The draft could not be written.")
            }
        }
    }

    /// Compose (C): a new message, sent through Mail only when the person
    /// clicks Send. Returns nil when sent, else the reason it was not.
    public func sendNewMessage(to: String, subject: String, body: String) async -> String? {
        let recipient = to.trimmingCharacters(in: .whitespacesAndNewlines)
        guard AppleMailSender.isPlausibleAddress(recipient) else { return MailSendError.invalidRecipient.userMessage }
        let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "Write the message first." }
        do {
            try await sender.send(MailDraft(to: recipient, subject: subject.trimmingCharacters(in: .whitespaces),
                                            body: text))
            return nil
        } catch let error as MailSendError {
            return error.userMessage
        } catch {
            return "Mail could not send the message."
        }
    }

    /// A compose window is open: C and E/Z type letters.
    public var isComposing = false

    /// Only the Send button calls this.
    public func send() async {
        guard canSend, let reply = replyDraft, let message = selected else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        sendToken += 1
        let token = sendToken
        draftState = .sending
        let outcome: DraftState
        do {
            var outgoing = reply
            outgoing.body = text
            try await sender.send(outgoing)
            outcome = .sent
            sentMessageIDs.insert(message.messageID)
        } catch let error as MailSendError {
            outcome = .failed(error.userMessage)
        } catch {
            outcome = .failed("Mail could not send the reply.")
        }
        // Another message may be open by now: its draft is not this one.
        guard selected?.messageID == message.messageID, token == sendToken else { return }
        draftState = outcome
    }
}
