import Foundation

/// A mail store in memory: previews and tests. Never reads Mail's files.
public actor InMemoryMailSource: MailSource {
    private var storeStatus: MailStoreStatus
    private var stored: [MailMessageSummary]
    private var bodies: [String: MailBody]
    private var accountList: [MailAccount]

    public init(status: MailStoreStatus = .available(version: "V10"), messages: [MailMessageSummary] = [],
                bodies: [String: MailBody] = [:], accounts: [MailAccount] = []) {
        self.storeStatus = status
        self.stored = messages
        self.bodies = bodies
        self.accountList = accounts
    }

    public func reload() async {}

    public func status() async -> MailStoreStatus { storeStatus }

    /// Tests: the store becomes readable (Full Disk Access applied) or not.
    public func setStatus(_ status: MailStoreStatus) {
        storeStatus = status
    }

    /// Tests: Mail saved a change (new mail arrives).
    public func setMessages(_ messages: [MailMessageSummary]) {
        stored = messages
        revision += 1
    }

    private var revision = 0

    public func messages(in selection: MailSelection, limit: Int) async throws -> [MailMessageSummary] {
        guard case .available = storeStatus else { throw MailSourceError.unavailable(storeStatus) }
        let chosen = stored.filter { message in
            switch selection {
            case .kinds(let kinds): kinds.contains(message.mailbox.kind)
            case .flagged: message.isFlagged
            case .mailbox(let url): message.mailbox.url == url
            }
        }
        let listed: MailboxRef.Kind? = switch selection {
        case .kinds(let kinds): kinds.count == 1 ? kinds.first : nil
        case .flagged: nil
        case .mailbox: chosen.first?.mailbox.kind
        }
        return Array(chosen.sorted { $0.date > $1.date }.prefix(limit)).map { message in
            var message = message
            message.listedKind = listed
            return message
        }
    }

    /// One mailbox per distinct mailbox of the stored messages.
    public func mailboxes() async throws -> MailboxListing {
        guard case .available = storeStatus else { throw MailSourceError.unavailable(storeStatus) }
        var urls: [String] = []
        for message in stored where !urls.contains(message.mailbox.url) { urls.append(message.mailbox.url) }
        let classified = MailboxClassifier.classify(urls.enumerated().map { (id: Int64($0.offset), url: $0.element) })
        let mailboxes = classified.map { _, mailbox in
            var mailbox = mailbox
            let inside = stored.filter { $0.mailbox.url == mailbox.ref.url }
            mailbox.totalCount = inside.count
            mailbox.unreadCount = inside.filter { !$0.isRead }.count
            return mailbox
        }
        return MailboxListing(mailboxes: mailboxes, flaggedCount: stored.filter(\.isFlagged).count)
    }

    public func changeToken() async -> String? {
        guard case .available = storeStatus else { return nil }
        return "\(revision)"
    }

    public func body(of message: MailMessageSummary) async throws -> MailBody {
        bodies[message.messageID] ?? MailBody(text: message.snippet)
    }

    public func accounts() async throws -> [MailAccount] { accountList }
}

/// A broker that never shows a prompt or opens System Settings: previews and
/// tests. Every request comes back `.notDetermined`.
public struct PreviewPermissionBroker: PermissionBroker {
    public init() {}
    public func state(of permission: AppPermission) async -> PermissionState { .notDetermined }
    public func request(_ permission: AppPermission) async -> PermissionState { .notDetermined }
    public func openSystemSettings(for permission: AppPermission) async {}
}
