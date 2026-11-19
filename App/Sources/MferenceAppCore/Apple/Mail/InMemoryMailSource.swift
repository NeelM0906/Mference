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

    public func messages(in kinds: Set<MailboxRef.Kind>, limit: Int) async throws -> [MailMessageSummary] {
        guard case .available = storeStatus else { throw MailSourceError.unavailable(storeStatus) }
        return Array(stored.filter { kinds.contains($0.mailbox.kind) }.sorted { $0.date > $1.date }.prefix(limit))
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
