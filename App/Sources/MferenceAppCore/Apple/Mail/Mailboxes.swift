import Foundation

/// One mailbox as Apple Mail's sidebar shows it: its standard role (or
/// `.other` for the person's own folders), its name and its counts.
public struct Mailbox: Sendable, Hashable, Identifiable {
    public var ref: MailboxRef
    /// The role in its account. Name-based like `MailboxRef.kind`, but a
    /// Gmail folder outside "[Gmail]" named "Drafts" is the person's own
    /// (`.other`), and only one mailbox per account holds each role.
    public var kind: MailboxRef.Kind
    public var name: String
    public var unreadCount: Int
    public var totalCount: Int

    public var id: String { ref.url }
    public var isLocal: Bool { ref.isLocal }

    /// What the sidebar shows next to it: Drafts counts every draft, the
    /// rest count unread mail, as Mail does.
    public var badgeCount: Int { kind == .drafts ? totalCount : unreadCount }

    public init(ref: MailboxRef, kind: MailboxRef.Kind, name: String, unreadCount: Int = 0, totalCount: Int = 0) {
        self.ref = ref
        self.kind = kind
        self.name = name
        self.unreadCount = unreadCount
        self.totalCount = totalCount
    }
}

/// Every mailbox, plus the flagged count when the store records flags.
public struct MailboxListing: Sendable, Hashable {
    public var mailboxes: [Mailbox]
    /// Nil when this Mail store has no flag column: Flagged is not offered.
    public var flaggedCount: Int?

    public init(mailboxes: [Mailbox] = [], flaggedCount: Int? = nil) {
        self.mailboxes = mailboxes
        self.flaggedCount = flaggedCount
    }
}

/// What a message list shows.
public enum MailSelection: Sendable, Hashable {
    /// Every mailbox with one of these roles, across accounts.
    case kinds(Set<MailboxRef.Kind>)
    /// Flagged mail anywhere.
    case flagged
    /// One mailbox, by its URL.
    case mailbox(String)
}

/// A sender or recipient.
public struct MailParticipant: Sendable, Hashable {
    public var name: String
    public var address: String

    public init(name: String, address: String) {
        self.name = name
        self.address = address
    }

    /// The name, else the address.
    public var display: String { name.isEmpty ? address : name }
}

/// Turns Mail's mailbox URLs into the sidebar's mailboxes: which are
/// standard, what they are called, which are hidden, and in what order.
enum MailboxClassifier {
    /// Gmail keeps its standard mailboxes under one of these.
    static let gmailContainers: Set<String> = ["[gmail]", "[google mail]"]

    /// The order of standard mailboxes inside an account (Mail's).
    static let order: [MailboxRef.Kind] = [.important, .inbox, .drafts, .sent, .junk, .trash, .archive]

    static func standardName(_ kind: MailboxRef.Kind) -> String? {
        switch kind {
        case .inbox: "Inbox"
        case .important: "Important"
        case .drafts: "Drafts"
        case .sent: "Sent"
        case .junk: "Junk"
        case .trash: "Trash"
        case .archive: "Archive"
        case .other: nil
        }
    }

    /// `rows` are (Envelope Index ROWID, URL). Returns the visible mailboxes
    /// with their row ids, accounts in order of first appearance, standard
    /// mailboxes first in Mail's order, then the person's folders by name.
    static func classify(_ rows: [(id: Int64, url: String)]) -> [(id: Int64, mailbox: Mailbox)] {
        let refs = rows.sorted { $0.id < $1.id }.map { (id: $0.id, ref: MailboxRef(url: $0.url)) }
        var accountOrder: [String] = []
        var byAccount: [String: [(id: Int64, ref: MailboxRef)]] = [:]
        for entry in refs where !entry.ref.path.isEmpty {
            let key = entry.ref.accountKey
            if byAccount[key] == nil { accountOrder.append(key) }
            byAccount[key, default: []].append(entry)
        }
        var result: [(id: Int64, mailbox: Mailbox)] = []
        for key in accountOrder {
            let entries = byAccount[key] ?? []
            let container = entries.lazy.compactMap { $0.ref.path.first }
                .first { gmailContainers.contains($0.lowercased()) }
            var taken: Set<MailboxRef.Kind> = []
            var standard: [(id: Int64, mailbox: Mailbox)] = []
            var folders: [(id: Int64, mailbox: Mailbox)] = []
            for (id, ref) in entries {
                var path = ref.path
                if let container, path.first == container {
                    // "[Gmail]" itself only groups; Starred is Mail's Flagged.
                    if path.count == 1 || (path.count == 2 && path[1].lowercased() == "starred") { continue }
                    path.removeFirst()
                }
                var kind = ref.kind
                // In Gmail, only "INBOX" and the "[Gmail]" mailboxes are standard.
                if container != nil, kind != .inbox, kind != .other, ref.path.first != container { kind = .other }
                if kind != .other, taken.contains(kind) { kind = .other }
                if kind != .other {
                    taken.insert(kind)
                    standard.append((id, Mailbox(ref: ref, kind: kind, name: standardName(kind) ?? path.joined(separator: " / "))))
                } else {
                    folders.append((id, Mailbox(ref: ref, kind: .other, name: path.joined(separator: " / "))))
                }
            }
            standard.sort { (order.firstIndex(of: $0.mailbox.kind) ?? 99) < (order.firstIndex(of: $1.mailbox.kind) ?? 99) }
            folders.sort { $0.mailbox.name.localizedStandardCompare($1.mailbox.name) == .orderedAscending }
            result += standard + folders
        }
        return result
    }
}
