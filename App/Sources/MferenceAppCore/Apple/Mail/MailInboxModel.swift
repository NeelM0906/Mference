import Foundation
import Observation

/// The Mail screen's state (mockup 03): Apple Mail's own mailboxes, the
/// list, the open message, its event card and the reply draft. Mail's store
/// is read, never written; the screen follows it as Mail saves changes.
@MainActor @Observable
public final class MailInboxModel {
    /// What the left column selects.
    public enum Folder: Hashable, Sendable {
        /// Favorites: every account's mailbox with this role (Inbox, Drafts, Sent).
        case favorite(MailboxRef.Kind)
        /// Flagged mail anywhere.
        case flagged
        /// One mailbox, by its URL.
        case mailbox(String)
    }

    /// The line at the bottom of the sidebar.
    public enum SyncState: Sendable, Hashable {
        /// Mail is running: what it downloads shows up here within seconds.
        case upToDate
        /// Mail is not running, so nothing new arrives.
        case mailClosed
    }

    /// One account's mailboxes in the sidebar.
    public struct AccountSection: Identifiable, Sendable, Hashable {
        public var account: MailAccount
        public var mailboxes: [Mailbox]
        public var id: String { account.id }
    }

    public enum DraftState: Sendable, Hashable {
        case idle
        case writing
        case ready
        case sending
        case sent
        case failed(String)
    }

    /// Favorites, in Mail's order (Flagged sits after Inbox when the store
    /// records flags).
    public static let favoriteKinds: [MailboxRef.Kind] = [.inbox, .drafts, .sent]

    public private(set) var status: MailStoreStatus?
    public private(set) var folder: Folder = .favorite(.inbox)
    public private(set) var accounts: [MailAccount] = []
    public private(set) var listing = MailboxListing()
    /// The selected folder's messages, newest first.
    public private(set) var messages: [MailMessageSummary] = []
    public private(set) var selected: MailMessageSummary?
    public private(set) var body: MailBody?
    public private(set) var eventCard: ProposedEvent?
    public var draft: String = ""
    public private(set) var draftState: DraftState = .idle
    public private(set) var draftModelLabel: String?
    public private(set) var syncState: SyncState = .upToDate
    /// When the lists were last read from Mail's store.
    public private(set) var lastRefreshed: Date?
    /// Accounts whose mailboxes are folded away in the sidebar.
    public var collapsedAccounts: Set<String> = []
    /// A text field is being edited: bare-letter shortcuts type letters then.
    public var isEditingDraft = false
    public var isEditingText: Bool { isEditingDraft || isComposing }
    public private(set) var errorMessage: String?

    @ObservationIgnored private var changeToken: String?
    /// Replies being sent and already sent, per message: reopening one shows
    /// its real state, and it cannot be sent twice.
    @ObservationIgnored private var sendingMessageIDs: Set<String> = []
    @ObservationIgnored private var sentMessageIDs: Set<String> = []

    private let source: any MailSource
    private let assistant: any AppleModelServicing
    private let drafter: MailReplyDrafter?
    private let sender: any MailSending
    private let broker: any PermissionBroker
    private let mailApp: any MailAppControlling
    private let onEventSuggested: @MainActor (ProposedEvent) -> Void
    private let timeZone: TimeZone
    private let limit: Int
    private let pollInterval: Duration
    private let now: @MainActor () -> Date

    public init(source: any MailSource, assistant: any AppleModelServicing,
                drafter: MailReplyDrafter?, sender: any MailSending, broker: any PermissionBroker,
                mailApp: any MailAppControlling = SystemMailApp(), timeZone: TimeZone = .current, limit: Int = 300,
                pollInterval: Duration = .seconds(3), now: @escaping @MainActor () -> Date = { .now },
                onEventSuggested: @escaping @MainActor (ProposedEvent) -> Void = { _ in }) {
        self.source = source
        self.assistant = assistant
        self.drafter = drafter
        self.sender = sender
        self.broker = broker
        self.mailApp = mailApp
        self.timeZone = timeZone
        self.limit = limit
        self.pollInterval = pollInterval
        self.now = now
        self.onEventSuggested = onEventSuggested
    }

    // MARK: Loading and following Mail

    /// Reads without asking; the screen shows the reason and a button when
    /// Full Disk Access is missing.
    public func load() async {
        await source.reload()
        let status = await source.status()
        self.status = status
        guard case .available = status else { return }
        changeToken = await source.changeToken()
        await refreshLists()
        if selected == nil, let first = messages.first { await open(first) }
    }

    /// Follows Mail while the screen is shown: every `pollInterval` it looks
    /// whether Mail saved anything and re-reads the lists if so. Ends when
    /// the calling task is cancelled.
    public func watch() async {
        while !Task.isCancelled {
            do { try await Task.sleep(for: pollInterval) } catch { return }
            await refreshIfChanged()
        }
    }

    /// Re-reads the mailboxes and the open list when Mail's store changed;
    /// keeps the folder and the open message.
    public func refreshIfChanged() async {
        guard case .available = status else { return }
        let token = await source.changeToken()
        guard token != changeToken else {
            syncState = mailApp.isRunning() ? .upToDate : .mailClosed
            return
        }
        changeToken = token
        await source.reload()
        await refreshLists()
    }

    /// Mference came back to the front: look again for access, or for mail.
    public func appBecameActive() async {
        if status == .needsFullDiskAccess || status == nil {
            await recheckAccess()
        } else {
            await refreshIfChanged()
        }
    }

    /// "Open Mail" under "Apple Mail is closed".
    public func openMail() {
        mailApp.open()
    }

    private func refreshLists() async {
        let shown = folder
        do {
            async let listing = source.mailboxes()
            async let accountList = source.accounts()
            (self.listing, accounts) = try await (listing, accountList)
            if !exists(folder) { folder = .favorite(.inbox) }
            let fresh = try await source.messages(in: selection(for: folder), limit: limit)
            // The person picked another folder meanwhile: that load wins.
            if folder == shown || !exists(shown) { messages = fresh }
            if let open = selected, let updated = fresh.first(where: { $0.messageID == open.messageID }) {
                selected = updated
            }
            errorMessage = nil
        } catch {
            errorMessage = "Mail could not be read: \(error)"
        }
        lastRefreshed = now()
        syncState = mailApp.isRunning() ? .upToDate : .mailClosed
    }

    /// The person clicked the button under the reason: the broker shows the
    /// reason, then opens the Full Disk Access pane itself (it cannot be
    /// asked in a dialog).
    public func requestAccess() async {
        settingsVisit.begin()
        _ = await broker.request(.fullDiskAccess)
        settingsVisit.end()
        await load()
    }

    /// Mference left the foreground (the view forwards
    /// `NSApplication.didResignActiveNotification`): during a request, that
    /// is the Full Disk Access pane opening rather than "Not now".
    public func appResignedActive() {
        settingsVisit.appLeft()
    }

    private var settingsVisit = SettingsVisit()

    /// The person went to the Full Disk Access pane from here at least once.
    public var hasOpenedFullDiskAccessSettings: Bool { settingsVisit.happened }

    /// Full Disk Access was opened but Mail is still not readable: macOS
    /// applies the grant only from the next launch (and an ad hoc rebuild is
    /// a new app to it), so the screen offers Quit and Reopen.
    public var needsRelaunchForAccess: Bool {
        status == .needsFullDiskAccess && hasOpenedFullDiskAccessSettings
    }

    /// "Check again": re-reads the store while access is missing.
    public func recheckAccess() async {
        guard status == .needsFullDiskAccess || status == nil else { return }
        await load()
    }

    // MARK: Mailboxes

    /// On My Mac.
    public var localMailboxes: [Mailbox] { listing.mailboxes.filter(\.isLocal) }

    /// Each account with its mailboxes, in the store's order.
    public var accountSections: [AccountSection] {
        var sections: [AccountSection] = []
        for mailbox in listing.mailboxes where !mailbox.isLocal {
            let id = mailbox.ref.accountID
            if let index = sections.firstIndex(where: { $0.id == id }) {
                sections[index].mailboxes.append(mailbox)
            } else {
                let account = accounts.first { $0.id == id } ?? MailAccount(id: id, name: MailAccount.unnamed)
                sections.append(AccountSection(account: account, mailboxes: [mailbox]))
            }
        }
        return sections
    }

    /// Flagged is offered only when Mail's store records flags.
    public var offersFlagged: Bool { listing.flaggedCount != nil }

    public func mailbox(_ url: String) -> Mailbox? {
        listing.mailboxes.first { $0.ref.url == url }
    }

    /// The sidebar's number: unread mail (every draft for Drafts, every
    /// flagged message for Flagged), as Mail shows it.
    public func count(in folder: Folder) -> Int {
        switch folder {
        case .favorite(let kind): listing.mailboxes.filter { $0.kind == kind }.reduce(0) { $0 + $1.badgeCount }
        case .flagged: listing.flaggedCount ?? 0
        case .mailbox(let url): mailbox(url)?.badgeCount ?? 0
        }
    }

    /// Unread mail in the open folder, for the list header.
    public var unreadInFolder: Int {
        switch folder {
        case .favorite(let kind): listing.mailboxes.filter { $0.kind == kind }.reduce(0) { $0 + $1.unreadCount }
        case .flagged: 0
        case .mailbox(let url): mailbox(url)?.unreadCount ?? 0
        }
    }

    public var folderTitle: String {
        switch folder {
        case .favorite(let kind): MailboxClassifier.standardName(kind) ?? "Mail"
        case .flagged: "Flagged"
        case .mailbox(let url): mailbox(url)?.name ?? "Mail"
        }
    }

    /// Where the open list comes from: "All accounts", an account, or On My Mac.
    public var folderScope: String {
        switch folder {
        case .favorite, .flagged: return "All accounts"
        case .mailbox(let url):
            guard let mailbox = mailbox(url) else { return "" }
            if mailbox.isLocal { return "On My Mac" }
            return accounts.first { $0.id == mailbox.ref.accountID }?.title ?? MailAccount.unnamed
        }
    }

    /// "Google · Inbox": where the open message is, for the reader.
    public func location(of message: MailMessageSummary) -> String {
        let place = message.mailbox.isLocal ? "On My Mac"
            : accounts.first { $0.id == message.mailbox.accountID }?.title ?? MailAccount.unnamed
        return "\(place) · \(folderTitle)"
    }

    public func select(_ folder: Folder) async {
        self.folder = folder
        guard case .available = status else { return }
        do {
            let fresh = try await source.messages(in: selection(for: folder), limit: limit)
            guard self.folder == folder else { return }
            messages = fresh
        } catch {
            guard self.folder == folder else { return }
            messages = []
            errorMessage = "Mail could not be read: \(error)"
        }
        if let first = messages.first { await open(first) } else { clearSelection() }
    }

    private func selection(for folder: Folder) -> MailSelection {
        switch folder {
        case .favorite(let kind): .kinds([kind])
        case .flagged: .flagged
        case .mailbox(let url): .mailbox(url)
        }
    }

    private func exists(_ folder: Folder) -> Bool {
        switch folder {
        case .favorite: true
        case .flagged: offersFlagged
        case .mailbox(let url): mailbox(url) != nil
        }
    }

    // MARK: Reading

    public func open(_ message: MailMessageSummary) async {
        isEditingDraft = false
        selected = message
        body = nil
        eventCard = nil
        draft = ""
        draftState = sentMessageIDs.contains(message.messageID) ? .sent
            : sendingMessageIDs.contains(message.messageID) ? .sending : .idle
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

    /// Replies are offered for received mail only: not for Drafts or Sent
    /// (where it is listed or stored), nor for Junk or Trash.
    public var canReply: Bool {
        guard let selected, !selected.isOutgoing else { return false }
        return ![selected.listedKind, selected.mailbox.kind].contains { $0 == .junk || $0 == .trash }
    }

    /// Who the reply goes to and from, and its subject, for the draft card.
    public var replyDraft: MailDraft? {
        guard let message = selected, canReply else { return nil }
        return MailDraft.reply(to: message, headers: body?.headers ?? [:], accountAddresses: accounts.compactMap(\.address),
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
        guard canSend, let reply = replyDraft, let message = selected,
              !sendingMessageIDs.contains(message.messageID), !sentMessageIDs.contains(message.messageID) else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        sendingMessageIDs.insert(message.messageID)
        defer { sendingMessageIDs.remove(message.messageID) }
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
        // Whatever is open now: if it is this message (even reopened), it
        // shows the outcome; another message's draft is left alone.
        guard selected?.messageID == message.messageID else { return }
        draftState = outcome
    }
}
