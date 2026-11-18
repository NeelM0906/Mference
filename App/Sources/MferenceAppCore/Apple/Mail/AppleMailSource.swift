import Foundation

/// One message in a list, from Mail's Envelope Index.
public struct MailMessageSummary: Sendable, Hashable, Identifiable {
    /// The Envelope Index row (also the `.emlx` file name).
    public var rowID: Int64
    /// The Message-ID header, or `mail-row:<rowID>` when the store lacks it.
    public var messageID: String
    public var senderName: String
    public var senderAddress: String
    public var subject: String
    /// Mail's own short summary, if it made one.
    public var snippet: String
    public var date: Date
    public var isRead: Bool
    public var mailbox: MailboxRef

    public var id: String { messageID }

    public init(rowID: Int64, messageID: String, senderName: String, senderAddress: String, subject: String,
                snippet: String, date: Date, isRead: Bool, mailbox: MailboxRef) {
        self.rowID = rowID
        self.messageID = messageID
        self.senderName = senderName
        self.senderAddress = senderAddress
        self.subject = subject
        self.snippet = snippet
        self.date = date
        self.isRead = isRead
        self.mailbox = mailbox
    }

    /// "PR" for the avatar.
    public var initials: String {
        let words = (senderName.isEmpty ? senderAddress : senderName).split(separator: " ").prefix(2)
        return words.compactMap(\.first).map(String.init).joined().uppercased()
    }
}

/// A mailbox, from its Envelope Index URL ("imap://ACCOUNT/INBOX").
public struct MailboxRef: Sendable, Hashable {
    public enum Kind: String, Sendable, Hashable, CaseIterable { case inbox, drafts, sent, archive, trash, junk, other }

    public var url: String
    /// The account's directory name under the Mail folder.
    public var accountID: String
    /// Path components below the account ("INBOX", "[Gmail]", "All Mail").
    public var path: [String]

    public init(url: String) {
        self.url = url
        let components = URLComponents(string: url)
        accountID = components?.host ?? ""
        path = (components?.percentEncodedPath ?? "").split(separator: "/")
            .map { $0.removingPercentEncoding ?? String($0) }
    }

    public var kind: Kind {
        let last = (path.last ?? "").lowercased()
        switch last {
        case "inbox": return .inbox
        case "drafts": return .drafts
        case "sent", "sent messages", "sent items", "sent mail": return .sent
        case "archive", "all mail": return .archive
        case "trash", "deleted messages", "deleted items", "bin": return .trash
        case "junk", "spam": return .junk
        default: return .other
        }
    }
}

/// What the Mail screen can do with this Mac's Mail store.
public enum MailStoreStatus: Sendable, Hashable {
    /// Readable; `version` is the store folder ("V10").
    case available(version: String)
    /// The store exists but macOS blocks reading it: Full Disk Access.
    case needsFullDiskAccess
    /// Mail has never been set up on this Mac.
    case notFound
    /// The store's layout is not one Mference knows; `detail` says what.
    case unsupported(detail: String)

    public var userMessage: String? {
        switch self {
        case .available: nil
        case .needsFullDiskAccess: AppPermission.fullDiskAccess.reason
        case .notFound: "Apple Mail is not set up on this Mac."
        case .unsupported: "This version of Apple Mail stores mail in a way Mference cannot read yet."
        }
    }
}

public enum MailSourceError: Error, Sendable, Hashable {
    case unavailable(MailStoreStatus)
    case messageNotFound
    case unreadable(String)
}

/// Mail on this Mac, read-only.
public protocol MailSource: Sendable {
    func status() async -> MailStoreStatus
    /// Newest first, from mailboxes of `kinds`, skipping deleted messages.
    func messages(in kinds: Set<MailboxRef.Kind>, limit: Int) async throws -> [MailMessageSummary]
    func body(of message: MailMessageSummary) async throws -> MailBody
    /// Account directory ids with an address seen in their mail, for the sidebar.
    func accounts() async throws -> [MailAccount]
}

public struct MailAccount: Sendable, Hashable, Identifiable {
    public var id: String
    public var name: String

    public init(id: String, name: String) {
        self.id = id
        self.name = name
    }
}

/// Reads Apple Mail's local store: the Envelope Index (SQLite) for lists and
/// the `.emlx` files for bodies. Opens everything read-only; needs Full Disk
/// Access (asked through `PermissionBroker`). Probes the schema first and
/// reports `.unsupported` instead of guessing.
public actor AppleMailSource: MailSource {
    /// Tables and columns the reader relies on.
    static let requiredSchema: [String: Set<String>] = [
        "messages": ["ROWID", "sender", "subject", "date_received", "mailbox", "read", "deleted"],
        "subjects": ["ROWID", "subject"],
        "addresses": ["ROWID", "address", "comment"],
        "mailboxes": ["ROWID", "url"],
    ]

    private let mailRoot: URL
    private var database: ReadOnlySQLite?
    private var versionDirectory: URL?
    private var features: (summaries: Bool, globalData: Bool, subjectPrefix: Bool) = (false, false, false)

    /// `mailRoot` is `~/Library/Mail` in the app, a fixture folder in tests.
    public init(mailRoot: URL = URL.libraryDirectory.appending(path: "Mail", directoryHint: .isDirectory)) {
        self.mailRoot = mailRoot
    }

    public func status() async -> MailStoreStatus {
        do {
            _ = try open()
            return .available(version: versionDirectory?.lastPathComponent ?? "")
        } catch let MailSourceError.unavailable(status) {
            return status
        } catch {
            return .unsupported(detail: "\(error)")
        }
    }

    /// The newest `V<n>` folder holding `MailData/Envelope Index`.
    static func findStore(in root: URL) throws -> (version: URL, index: URL)? {
        let entries: [URL]
        do {
            entries = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        } catch let error as CocoaError where error.code == .fileReadNoPermission {
            throw MailSourceError.unavailable(.needsFullDiskAccess)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            return nil
        }
        let versions = entries.compactMap { url -> (Int, URL)? in
            let name = url.lastPathComponent
            guard name.hasPrefix("V"), let number = Int(name.dropFirst()) else { return nil }
            return (number, url)
        }.sorted { $0.0 > $1.0 }
        for (_, version) in versions {
            let index = version.appending(path: "MailData/Envelope Index")
            if FileManager.default.fileExists(atPath: index.path) { return (version, index) }
        }
        return nil
    }

    private func open() throws -> ReadOnlySQLite {
        if let database { return database }
        guard let (version, index) = try Self.findStore(in: mailRoot) else {
            throw MailSourceError.unavailable(.notFound)
        }
        // macOS privacy protection refuses the read itself without Full Disk
        // Access, whatever the file mode says.
        do {
            let handle = try FileHandle(forReadingFrom: index)
            defer { try? handle.close() }
            _ = try handle.read(upToCount: 16)
        } catch {
            throw MailSourceError.unavailable(.needsFullDiskAccess)
        }
        let connection: ReadOnlySQLite
        do {
            connection = try Self.connect(index, immutable: false)
        } catch {
            // The WAL index cannot be opened read-only while Mail is closed.
            do {
                connection = try Self.connect(index, immutable: true)
            } catch let failure as ReadOnlySQLite.Failure {
                throw MailSourceError.unavailable(.unsupported(detail: failure.description))
            }
        }
        for (table, columns) in Self.requiredSchema {
            let present = try connection.columns(of: table).union(["ROWID"])
            let missing = columns.subtracting(present)
            guard missing.isEmpty else {
                throw MailSourceError.unavailable(.unsupported(
                    detail: "\(version.lastPathComponent): \(table) lacks \(missing.sorted().joined(separator: ", "))"))
            }
        }
        features = (summaries: try connection.columns(of: "summaries").contains("summary")
                        && connection.columns(of: "messages").contains("summary"),
                    globalData: try connection.columns(of: "message_global_data").contains("message_id_header")
                        && connection.columns(of: "messages").contains("global_message_id"),
                    subjectPrefix: try connection.columns(of: "messages").contains("subject_prefix"))
        database = connection
        versionDirectory = version
        return connection
    }

    private static func connect(_ index: URL, immutable: Bool) throws -> ReadOnlySQLite {
        let connection = try ReadOnlySQLite(url: index, immutable: immutable)
        _ = try connection.query("SELECT count(*) FROM sqlite_master")
        return connection
    }

    public func messages(in kinds: Set<MailboxRef.Kind>, limit: Int) async throws -> [MailMessageSummary] {
        let database = try open()
        let summaryColumn = features.summaries ? "su.summary AS summary" : "NULL AS summary"
        let summaryJoin = features.summaries ? "LEFT JOIN summaries su ON su.ROWID = m.summary" : ""
        let idColumn = features.globalData ? "g.message_id_header AS message_id_header" : "NULL AS message_id_header"
        let idJoin = features.globalData ? "LEFT JOIN message_global_data g ON g.ROWID = m.global_message_id" : ""
        let prefixColumn = features.subjectPrefix ? "m.subject_prefix AS subject_prefix" : "NULL AS subject_prefix"
        // Mailbox kinds come from URLs, so filter in Swift over a bounded,
        // newest-first window.
        let sql = """
            SELECT m.ROWID AS rowid, m.date_received AS date_received, m.read AS read,
                   a.address AS address, a.comment AS comment, s.subject AS subject, mb.url AS url,
                   \(summaryColumn), \(idColumn), \(prefixColumn)
            FROM messages m
            LEFT JOIN addresses a ON a.ROWID = m.sender
            LEFT JOIN subjects s ON s.ROWID = m.subject
            LEFT JOIN mailboxes mb ON mb.ROWID = m.mailbox
            \(summaryJoin) \(idJoin)
            WHERE m.deleted = 0
            ORDER BY m.date_received DESC
            LIMIT ?
            """
        let rows = try database.query(sql, [.integer(Int64(max(limit, 1) * 4))])
        var result: [MailMessageSummary] = []
        for row in rows {
            let mailbox = MailboxRef(url: row["url"]?.string ?? "")
            guard kinds.contains(mailbox.kind), let rowID = row["rowid"]?.int else { continue }
            let subject = (row["subject_prefix"]?.string ?? "") + (row["subject"]?.string ?? "")
            let header = row["message_id_header"]?.string.flatMap { $0.isEmpty ? nil : $0 }
            result.append(MailMessageSummary(
                rowID: rowID,
                messageID: header ?? "mail-row:\(rowID)",
                senderName: MailMessageParser.decodeHeader(row["comment"]?.string ?? ""),
                senderAddress: row["address"]?.string ?? "",
                subject: MailMessageParser.decodeHeader(subject),
                snippet: row["summary"]?.string ?? "",
                date: Date(timeIntervalSince1970: row["date_received"]?.double ?? 0),
                isRead: (row["read"]?.int ?? 0) != 0,
                mailbox: mailbox))
            if result.count == limit { break }
        }
        return result
    }

    public func accounts() async throws -> [MailAccount] {
        let database = try open()
        let rows = try database.query("SELECT DISTINCT url FROM mailboxes")
        var ids: [String] = []
        for url in rows.compactMap({ $0["url"]?.string }) {
            let id = MailboxRef(url: url).accountID
            if !id.isEmpty, !ids.contains(id) { ids.append(id) }
        }
        return ids.map { MailAccount(id: $0, name: accountName(for: $0)) }
    }

    /// The account's own address if Mail recorded it, else its folder name.
    private func accountName(for id: String) -> String {
        guard let version = versionDirectory else { return id }
        let plist = version.appending(path: "MailData/Accounts.plist")
        if let data = try? Data(contentsOf: plist),
           let object = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
           let accounts = object["MailAccounts"] as? [[String: Any]],
           let match = accounts.first(where: { ($0["AccountPath"] as? String)?.hasSuffix(id) == true || $0["uniqueId"] as? String == id }),
           let address = (match["EmailAddresses"] as? [String])?.first ?? match["AccountName"] as? String {
            return address
        }
        return id
    }

    public func body(of message: MailMessageSummary) async throws -> MailBody {
        _ = try open()
        guard let url = emlxURL(for: message) else { throw MailSourceError.messageNotFound }
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: url) } catch {
            throw MailSourceError.unreadable(error.localizedDescription)
        }
        defer { try? handle.close() }
        let data = try handle.read(upToCount: MailMessageParser.maximumBytes) ?? Data()
        guard let body = MailMessageParser.parseEMLX(data) else {
            throw MailSourceError.unreadable("The message could not be read.")
        }
        return body
    }

    /// The mailbox folder ("<V>/<account>/INBOX.mbox"), then its
    /// `<store>/Data/<digits>/Messages/<row>.emlx`.
    func emlxURL(for message: MailMessageSummary) -> URL? {
        guard let version = versionDirectory else { return nil }
        var mailboxDirectory = version.appending(path: message.mailbox.accountID.isEmpty
                                                 ? "Mailboxes" : message.mailbox.accountID)
        for component in message.mailbox.path { mailboxDirectory.append(path: "\(component).mbox") }
        let names = ["\(message.rowID).emlx", "\(message.rowID).partial.emlx"]
        let shard = Self.shardPath(for: message.rowID)
        let stores = (try? FileManager.default.contentsOfDirectory(at: mailboxDirectory, includingPropertiesForKeys: nil))
            ?? []
        for store in stores {
            for name in names {
                let candidate = store.appending(path: "Data\(shard)/Messages/\(name)")
                if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            }
        }
        // Layouts differ between versions: search the mailbox, bounded.
        guard let enumerator = FileManager.default.enumerator(at: mailboxDirectory, includingPropertiesForKeys: nil)
        else { return nil }
        var visited = 0
        for case let url as URL in enumerator {
            visited += 1
            if visited > 200_000 { break }
            if names.contains(url.lastPathComponent) { return url }
        }
        return nil
    }

    /// Mail shards messages by the thousands of their row id, digits
    /// reversed: row 123456 lives in `Data/3/2/1/Messages`.
    static func shardPath(for rowID: Int64) -> String {
        let thousands = rowID / 1_000
        guard thousands > 0 else { return "" }
        return String(thousands).reversed().map { "/\($0)" }.joined()
    }
}
