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
    /// The mailbox that stores it (its `.emlx`). For Gmail that is usually
    /// All Mail, whatever list it was found in.
    public var mailbox: MailboxRef
    public var isFlagged: Bool
    /// To first, then Cc; read for outgoing mail only.
    public var recipients: [MailParticipant]
    /// The role of the list it was found in (Sent for a Gmail message stored
    /// in All Mail but labelled Sent); nil when it came from Flagged.
    public var listedKind: MailboxRef.Kind?

    public var id: String { messageID }

    public init(rowID: Int64, messageID: String, senderName: String, senderAddress: String, subject: String,
                snippet: String, date: Date, isRead: Bool, mailbox: MailboxRef, isFlagged: Bool = false,
                recipients: [MailParticipant] = [], listedKind: MailboxRef.Kind? = nil) {
        self.rowID = rowID
        self.messageID = messageID
        self.senderName = senderName
        self.senderAddress = senderAddress
        self.subject = subject
        self.snippet = snippet
        self.date = date
        self.isRead = isRead
        self.mailbox = mailbox
        self.isFlagged = isFlagged
        self.recipients = recipients
        self.listedKind = listedKind
    }

    /// Mail the person wrote: listed in, or stored in, Sent or Drafts.
    public var isOutgoing: Bool {
        [listedKind, mailbox.kind].contains { $0 == .sent || $0 == .drafts }
    }

    /// The sender's name or address.
    public var senderDisplay: String { senderName.isEmpty ? senderAddress : senderName }

    /// Who the list row names: the first recipient (and how many more) for
    /// outgoing mail, else the sender.
    public var counterpart: String {
        guard isOutgoing, let first = recipients.first else { return senderDisplay }
        return recipients.count > 1 ? "\(first.display), +\(recipients.count - 1)" : first.display
    }

    /// "PR" for the avatar.
    public var initials: String {
        let words = (senderName.isEmpty ? senderAddress : senderName).split(separator: " ").prefix(2)
        return words.compactMap(\.first).map(String.init).joined().uppercased()
    }
}

/// A mailbox, from its Envelope Index URL ("imap://ACCOUNT/INBOX").
public struct MailboxRef: Sendable, Hashable {
    public enum Kind: String, Sendable, Hashable, CaseIterable {
        case inbox, important, drafts, sent, archive, trash, junk, other
    }

    public var url: String
    /// "imap", "ews", "local" (On My Mac)...
    public var scheme: String
    /// The account's directory name under the Mail folder.
    public var accountID: String
    /// Path components below the account ("INBOX", "[Gmail]", "All Mail").
    public var path: [String]

    public init(url: String) {
        self.url = url
        let components = URLComponents(string: url)
        scheme = components?.scheme?.lowercased() ?? ""
        accountID = components?.host ?? ""
        path = (components?.percentEncodedPath ?? "").split(separator: "/")
            .map { $0.removingPercentEncoding ?? String($0) }
    }

    /// An On My Mac mailbox rather than an account's.
    public var isLocal: Bool { scheme == "local" || accountID.isEmpty }

    /// Groups an account's mailboxes; every local mailbox shares one.
    var accountKey: String { isLocal ? "local" : accountID }

    /// False for an account or folder name that could leave the Mail
    /// folder ("..", or a "/" smuggled in as %2F): such mail is not opened.
    public var isSafePath: Bool {
        ([accountID] + path).allSatisfy { component in
            !component.contains("/") && component != ".." && component != "." && !component.contains("\0")
        } && path.allSatisfy { !$0.isEmpty }
    }

    public var kind: Kind {
        let last = (path.last ?? "").lowercased()
        switch last {
        case "inbox": return .inbox
        case "important": return .important
        case "drafts": return .drafts
        case "sent", "sent messages", "sent items", "sent mail": return .sent
        case "archive", "all mail": return .archive
        case "trash", "deleted messages", "deleted items", "deleted", "bin": return .trash
        case "junk", "spam", "junk e-mail", "junk email", "bulk mail": return .junk
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
    /// Takes a fresh look at the store (new mail since the last load).
    func reload() async
    func status() async -> MailStoreStatus
    /// Newest first, skipping deleted messages. A message stored in one
    /// mailbox and labelled into another (Gmail) is listed in both.
    func messages(in selection: MailSelection, limit: Int) async throws -> [MailMessageSummary]
    /// The sidebar's mailboxes with their counts.
    func mailboxes() async throws -> MailboxListing
    func body(of message: MailMessageSummary) async throws -> MailBody
    /// Account directory ids with an address seen in their mail, for the sidebar.
    func accounts() async throws -> [MailAccount]
    /// Changes whenever Mail writes its store; nil when it cannot be read.
    /// Cheap: polled every few seconds.
    func changeToken() async -> String?
}

extension MailSource {
    /// Newest first, from every mailbox with one of `kinds`.
    public func messages(in kinds: Set<MailboxRef.Kind>, limit: Int) async throws -> [MailMessageSummary] {
        try await messages(in: .kinds(kinds), limit: limit)
    }
}

public struct MailAccount: Sendable, Hashable, Identifiable {
    /// Mail's account folder (a UUID in current versions).
    public var id: String
    /// The address, else the account's name, else "Mail account". Never the
    /// folder UUID.
    public var name: String
    /// The account's own address when known: replies are sent from it.
    public var address: String?
    /// The account's description in Mail ("Google", "iCloud"), when known.
    public var label: String?

    public init(id: String, name: String, address: String? = nil, label: String? = nil) {
        self.id = id
        self.name = name
        self.address = address ?? (name.contains("@") ? name : nil)
        self.label = label
    }

    /// The sidebar section header: Mail's description, else `name`.
    public var title: String { label ?? name }

    /// The line under the header: the address, unless the header already is it.
    public var subtitle: String? {
        guard let address, address != title else { return nil }
        return address
    }

    /// Shown when nothing names the account.
    public static let unnamed = "Mail account"
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
    /// The system's account list (`~/Library/Accounts/Accounts4.sqlite`),
    /// which names Mail's account folders; nil reads only Mail's own store.
    private let accountsDatabase: URL?
    private var database: ReadOnlySQLite?
    /// A private copy of the Envelope Index (with its -wal and -shm) that
    /// this source reads; Mail's own files are only ever copied, never opened
    /// by SQLite.
    private var snapshotDirectory: URL?
    private var versionDirectory: URL?
    /// Optional parts of the schema, probed when the store is opened.
    struct Features {
        var summaries = false
        var globalData = false
        var subjectPrefix = false
        /// `labels(message_id, mailbox_id)`: Gmail's extra mailboxes per message.
        var labels = false
        /// `messages.flagged`.
        var flagged = false
        /// `recipients(message, address, type, position)`.
        var recipients = false
        var recipientPosition = false
    }
    private var features = Features()

    public static let defaultMailRoot = URL.libraryDirectory.appending(path: "Mail", directoryHint: .isDirectory)

    /// `mailRoot` is `~/Library/Mail` in the app, a fixture folder in tests.
    /// `accountsDatabase` defaults to the system's account list only for the
    /// real Mail folder, so a fixture never reads this Mac's accounts.
    public init(mailRoot: URL = AppleMailSource.defaultMailRoot, accountsDatabase: URL? = nil) {
        self.mailRoot = mailRoot
        self.accountsDatabase = accountsDatabase ?? (mailRoot.standardizedFileURL == Self.defaultMailRoot.standardizedFileURL
            ? URL.libraryDirectory.appending(path: "Accounts/Accounts4.sqlite") : nil)
    }

    deinit {
        if let snapshotDirectory { try? FileManager.default.removeItem(at: snapshotDirectory) }
    }

    public func reload() async {
        database = nil
        if let snapshotDirectory { try? FileManager.default.removeItem(at: snapshotDirectory) }
        snapshotDirectory = nil
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
        // Copy the index, its write-ahead log and shared memory into a
        // private 0700 folder, fold the log into the copy and read the copy
        // immutable: SQLite never opens, locks or writes Mail's live files. On APFS the copy is a clone (no data
        // copied). macOS privacy protection refuses the copy without Full
        // Disk Access, whatever the file mode says.
        let snapshot = try Self.snapshot(index)
        let connection: ReadOnlySQLite
        do {
            let copy = snapshot.appending(path: index.lastPathComponent)
            try ReadOnlySQLite.checkpointPrivateCopy(at: copy)
            connection = try Self.connect(copy)
        } catch let failure as ReadOnlySQLite.Failure {
            try? FileManager.default.removeItem(at: snapshot)
            throw MailSourceError.unavailable(.unsupported(detail: failure.description))
        }
        snapshotDirectory = snapshot
        for (table, columns) in Self.requiredSchema {
            let present = try connection.columns(of: table).union(["ROWID"])
            let missing = columns.subtracting(present)
            guard missing.isEmpty else {
                throw MailSourceError.unavailable(.unsupported(
                    detail: "\(version.lastPathComponent): \(table) lacks \(missing.sorted().joined(separator: ", "))"))
            }
        }
        let messageColumns = try connection.columns(of: "messages")
        let recipientColumns = try connection.columns(of: "recipients")
        features = Features(
            summaries: try connection.columns(of: "summaries").contains("summary") && messageColumns.contains("summary"),
            globalData: try connection.columns(of: "message_global_data").contains("message_id_header")
                && messageColumns.contains("global_message_id"),
            subjectPrefix: messageColumns.contains("subject_prefix"),
            labels: try connection.columns(of: "labels").isSuperset(of: ["message_id", "mailbox_id"]),
            flagged: messageColumns.contains("flagged"),
            recipients: recipientColumns.isSuperset(of: ["message", "address", "type"]),
            recipientPosition: recipientColumns.contains("position"))
        database = connection
        versionDirectory = version
        return connection
    }

    /// Copies `index` and its "-wal" / "-shm" companions into a new private
    /// folder and returns that folder.
    static func snapshot(_ index: URL) throws -> URL {
        let directory = URL.temporaryDirectory.appending(path: "mference-mail-\(UUID().uuidString)",
                                                         directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        do {
            for suffix in ["", "-wal", "-shm"] {
                let source = index.deletingLastPathComponent().appending(path: index.lastPathComponent + suffix)
                guard suffix.isEmpty || FileManager.default.fileExists(atPath: source.path) else { continue }
                try FileManager.default.copyItem(at: source, to: directory.appending(path: source.lastPathComponent))
            }
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw MailSourceError.unavailable(.needsFullDiskAccess)
        }
        return directory
    }

    /// Opens the private copy `mode=ro&immutable=1`, after its write-ahead
    /// log was folded into it (`checkpointPrivateCopy`), so the newest mail
    /// is there.
    private static func connect(_ index: URL) throws -> ReadOnlySQLite {
        let connection = try ReadOnlySQLite(url: index)
        _ = try connection.query("SELECT count(*) FROM sqlite_master")
        return connection
    }

    /// Visible mailboxes with their Envelope Index row ids.
    private func classifiedMailboxes(_ database: ReadOnlySQLite) throws -> [(id: Int64, mailbox: Mailbox)] {
        let rows = try database.query("SELECT ROWID AS id, url FROM mailboxes").compactMap { row -> (id: Int64, url: String)? in
            guard let id = row["id"]?.int, let url = row["url"]?.string else { return nil }
            return (id, url)
        }
        return MailboxClassifier.classify(rows)
    }

    public func mailboxes() async throws -> MailboxListing {
        let database = try open()
        let classified = try classifiedMailboxes(database)
        var unread: [Int64: Int] = [:]
        var total: [Int64: Int] = [:]
        func add(_ rows: [[String: ReadOnlySQLite.Value]]) {
            for row in rows {
                guard let id = row["id"]?.int else { continue }
                unread[id, default: 0] += Int(row["unread"]?.int ?? 0)
                total[id, default: 0] += Int(row["total"]?.int ?? 0)
            }
        }
        add(try database.query("""
            SELECT mailbox AS id, sum(CASE WHEN read = 0 THEN 1 ELSE 0 END) AS unread, count(*) AS total
            FROM messages WHERE deleted = 0 GROUP BY mailbox
            """))
        if features.labels {
            // A label naming the storing mailbox again is not counted twice.
            add(try database.query("""
                SELECT l.mailbox_id AS id, sum(CASE WHEN m.read = 0 THEN 1 ELSE 0 END) AS unread, count(*) AS total
                FROM labels l JOIN messages m ON m.ROWID = l.message_id
                WHERE m.deleted = 0 AND l.mailbox_id != m.mailbox GROUP BY l.mailbox_id
                """))
        }
        var flagged: Int?
        if features.flagged {
            let rows = try database.query("SELECT count(*) AS n FROM messages WHERE deleted = 0 AND flagged != 0")
            flagged = Int(rows.first?["n"]?.int ?? 0)
        }
        let mailboxes = classified.compactMap { id, mailbox -> Mailbox? in
            var mailbox = mailbox
            mailbox.unreadCount = unread[id] ?? 0
            mailbox.totalCount = total[id] ?? 0
            // Mail shows the Outbox only while something waits in it.
            if mailbox.isLocal, mailbox.kind == .other, mailbox.name.lowercased() == "outbox",
               mailbox.totalCount == 0 { return nil }
            return mailbox
        }
        return MailboxListing(mailboxes: mailboxes, flaggedCount: flagged)
    }

    public func messages(in selection: MailSelection, limit: Int) async throws -> [MailMessageSummary] {
        let database = try open()
        let summaryColumn = features.summaries ? "su.summary AS summary" : "NULL AS summary"
        let summaryJoin = features.summaries ? "LEFT JOIN summaries su ON su.ROWID = m.summary" : ""
        let idColumn = features.globalData ? "g.message_id_header AS message_id_header" : "NULL AS message_id_header"
        let idJoin = features.globalData ? "LEFT JOIN message_global_data g ON g.ROWID = m.global_message_id" : ""
        let prefixColumn = features.subjectPrefix ? "m.subject_prefix AS subject_prefix" : "NULL AS subject_prefix"
        let flaggedColumn = features.flagged ? "m.flagged AS flagged" : "0 AS flagged"
        // Pick the mailboxes first, then let SQL filter and limit, so a quiet
        // mailbox (Sent) is never crowded out by newer mail elsewhere.
        let filter: String
        var parameters: [ReadOnlySQLite.Value] = []
        let listedKind: MailboxRef.Kind?
        switch selection {
        case .flagged:
            guard features.flagged else { return [] }
            filter = "m.flagged != 0"
            listedKind = nil
        case .kinds, .mailbox:
            let classified = try classifiedMailboxes(database)
            let chosen = classified.filter { _, mailbox in
                if case .kinds(let kinds) = selection { return kinds.contains(mailbox.kind) }
                if case .mailbox(let url) = selection { return mailbox.ref.url == url }
                return false
            }
            guard !chosen.isEmpty else { return [] }
            let kinds = Set(chosen.map(\.mailbox.kind))
            listedKind = kinds.count == 1 ? kinds.first : nil
            let placeholders = Array(repeating: "?", count: chosen.count).joined(separator: ", ")
            parameters = chosen.map { .integer($0.id) }
            if features.labels {
                // Gmail stores a message once (usually in All Mail) and
                // labels it into Inbox, Sent, Important...
                filter = """
                    (m.mailbox IN (\(placeholders))
                     OR m.ROWID IN (SELECT message_id FROM labels WHERE mailbox_id IN (\(placeholders))))
                    """
                parameters += parameters
            } else {
                filter = "m.mailbox IN (\(placeholders))"
            }
        }
        let sql = """
            SELECT m.ROWID AS rowid, m.date_received AS date_received, m.read AS read,
                   a.address AS address, a.comment AS comment, s.subject AS subject, mb.url AS url,
                   \(summaryColumn), \(idColumn), \(prefixColumn), \(flaggedColumn)
            FROM messages m
            LEFT JOIN addresses a ON a.ROWID = m.sender
            LEFT JOIN subjects s ON s.ROWID = m.subject
            LEFT JOIN mailboxes mb ON mb.ROWID = m.mailbox
            \(summaryJoin) \(idJoin)
            WHERE m.deleted = 0 AND \(filter)
            ORDER BY m.date_received DESC
            LIMIT ?
            """
        let rows = try database.query(sql, parameters + [.integer(Int64(max(limit, 1)))])
        var result: [MailMessageSummary] = []
        for row in rows {
            let mailbox = MailboxRef(url: row["url"]?.string ?? "")
            guard let rowID = row["rowid"]?.int else { continue }
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
                mailbox: mailbox,
                isFlagged: (row["flagged"]?.int ?? 0) != 0,
                listedKind: listedKind))
            if result.count == limit { break }
        }
        let outgoing = result.indices.filter { result[$0].isOutgoing }
        if !outgoing.isEmpty {
            let found = try recipients(of: outgoing.map { result[$0].rowID }, database: database)
            for index in outgoing { result[index].recipients = found[result[index].rowID] ?? [] }
        }
        return result
    }

    /// To, then Cc, in header order, per message row.
    private func recipients(of rowIDs: [Int64], database: ReadOnlySQLite) throws -> [Int64: [MailParticipant]] {
        guard features.recipients, !rowIDs.isEmpty else { return [:] }
        let placeholders = Array(repeating: "?", count: rowIDs.count).joined(separator: ", ")
        let position = features.recipientPosition ? ", r.position" : ""
        let rows = try database.query("""
            SELECT r.message AS message, a.address AS address, a.comment AS comment
            FROM recipients r JOIN addresses a ON a.ROWID = r.address
            WHERE r.message IN (\(placeholders))
            ORDER BY r.message, r.type\(position), r.ROWID
            """, rowIDs.map { .integer($0) })
        var result: [Int64: [MailParticipant]] = [:]
        for row in rows {
            guard let message = row["message"]?.int else { continue }
            result[message, default: []].append(MailParticipant(
                name: MailMessageParser.decodeHeader(row["comment"]?.string ?? ""),
                address: row["address"]?.string ?? ""))
        }
        return result
    }

    /// The size and modification time of the index and its write-ahead
    /// log: Mail touches one of them for every change it saves.
    public func changeToken() async -> String? {
        let index: URL
        if let versionDirectory {
            index = versionDirectory.appending(path: "MailData/Envelope Index")
        } else {
            guard let found = try? Self.findStore(in: mailRoot) else { return nil }
            index = found.index
        }
        let parts = ["", "-wal"].map { suffix -> String in
            let path = index.path + suffix
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path) else { return "-" }
            let size = (attributes[.size] as? NSNumber)?.int64Value ?? -1
            let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            return "\(size)@\(modified)"
        }
        return parts.joined(separator: "|")
    }

    public func accounts() async throws -> [MailAccount] {
        let database = try open()
        let mailboxes = try database.query("SELECT ROWID AS id, url FROM mailboxes").compactMap { row -> (Int64, MailboxRef)? in
            guard let id = row["id"]?.int else { return nil }
            return (id, MailboxRef(url: row["url"]?.string ?? ""))
        }
        var ids: [String] = []
        for (_, mailbox) in mailboxes where !mailbox.isLocal && !ids.contains(mailbox.accountID) {
            ids.append(mailbox.accountID)
        }
        let system = systemAccountNames()
        return ids.map { id in
            let named = system[id]
            let sentFrom = try? sentFromAddress(account: id, mailboxes: mailboxes, database: database)
            let address = named?.address ?? sentFrom
            return MailAccount(id: id, name: address ?? named?.name ?? MailAccount.unnamed, address: address,
                               label: named?.name)
        }
    }

    /// The address the account's Sent mailboxes were sent from most often.
    private func sentFromAddress(account: String, mailboxes: [(Int64, MailboxRef)],
                                 database: ReadOnlySQLite) throws -> String? {
        let sent = mailboxes.filter { $0.1.accountID == account && $0.1.kind == .sent }.map(\.0)
        guard !sent.isEmpty else { return nil }
        let placeholders = Array(repeating: "?", count: sent.count).joined(separator: ", ")
        let rows = try database.query("""
            SELECT a.address AS address, count(*) AS n FROM messages m
            JOIN addresses a ON a.ROWID = m.sender
            WHERE m.mailbox IN (\(placeholders)) AND a.address LIKE '%@%'
            GROUP BY lower(a.address) ORDER BY n DESC LIMIT 1
            """, sent.map { .integer($0) })
        return rows.first?["address"]?.string
    }

    /// Mail's account folders are named by the system account's identifier:
    /// its user name (an address) or description ("iCloud"), or its parent
    /// account's. Read from a private immutable copy; any failure means no
    /// names, never an error.
    func systemAccountNames() -> [String: (name: String?, address: String?)] {
        guard let accountsDatabase, FileManager.default.fileExists(atPath: accountsDatabase.path),
              let snapshot = try? Self.snapshot(accountsDatabase) else { return [:] }
        defer { try? FileManager.default.removeItem(at: snapshot) }
        let copy = snapshot.appending(path: accountsDatabase.lastPathComponent)
        guard (try? ReadOnlySQLite.checkpointPrivateCopy(at: copy)) != nil,
              let connection = try? ReadOnlySQLite(url: copy),
              let columns = try? connection.columns(of: "ZACCOUNT"),
              columns.isSuperset(of: ["ZIDENTIFIER", "ZUSERNAME", "ZACCOUNTDESCRIPTION", "ZPARENTACCOUNT"]),
              let rows = try? connection.query("""
                  SELECT a.ZIDENTIFIER AS id, a.ZUSERNAME AS user, a.ZACCOUNTDESCRIPTION AS description,
                         p.ZUSERNAME AS parent_user, p.ZACCOUNTDESCRIPTION AS parent_description
                  FROM ZACCOUNT a LEFT JOIN ZACCOUNT p ON p.Z_PK = a.ZPARENTACCOUNT
                  """) else { return [:] }
        var names: [String: (name: String?, address: String?)] = [:]
        for row in rows {
            guard let id = row["id"]?.string, !id.isEmpty else { continue }
            let text = { (key: String) in row[key]?.string.flatMap { $0.isEmpty ? nil : $0 } }
            let address = [text("user"), text("parent_user")].compactMap { $0 }.first { $0.contains("@") }
            let name = text("description") ?? text("parent_description") ?? text("user") ?? text("parent_user")
            names[id] = (name, address)
        }
        return names
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
        guard let version = versionDirectory, message.mailbox.isSafePath else { return nil }
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
