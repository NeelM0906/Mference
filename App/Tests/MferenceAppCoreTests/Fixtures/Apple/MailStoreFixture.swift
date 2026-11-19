import Foundation
@testable import MferenceAppCore
import SQLite3

/// A tiny, synthetic Apple Mail store: an Envelope Index with the tables the
/// reader uses and a few `.emlx` files. Invented people at example.com; never
/// the real ~/Library/Mail.
struct MailStoreFixture {
    struct Message {
        var rowID: Int64
        var messageID: String
        var name: String
        var address: String
        var subject: String
        var summary: String
        var date: Date
        var mailbox: String
        var read = false
        var deleted = false
        var flagged = false
        /// To first, then Cc: (name, address, type 0 To / 1 Cc).
        var recipients: [(name: String, address: String, type: Int)] = []
        /// Further mailboxes the message is labelled into (Gmail).
        var labels: [String] = []
        var raw: String
    }

    static let account = "4F7A2C1E-0000-4000-8000-0000000000AA"
    /// A second account with only a Junk mailbox: nothing in Mail names it.
    static let secondAccount = "7C3D9B20-0000-4000-8000-0000000000BB"
    /// A Gmail account: every message stored in All Mail, labelled into
    /// Inbox, Sent, Important and a folder of the person's own.
    static let gmailAccount = "2B8E5F70-0000-4000-8000-0000000000CC"
    static let allMail = "imap://\(gmailAccount)/%5BGmail%5D/All%20Mail"
    static let received = Date(timeIntervalSince1970: 1_789_465_320)  // 2026-09-15 09:42 UTC

    let root: URL

    static var messages: [Message] {
        [
            Message(rowID: 101, messageID: "<review@example.com>", name: "Priya Raman", address: "priya@example.com",
                    subject: "Wednesday design review",
                    summary: "Could we move it to 11? I'll bring the new onboarding screens.",
                    date: received, mailbox: "imap://\(account)/INBOX",
                    raw: """
                    From: Priya Raman <priya@example.com>
                    To: You <me@example.com>
                    Subject: Wednesday design review
                    Message-ID: <review@example.com>
                    MIME-Version: 1.0
                    Content-Type: multipart/alternative; boundary="b1"

                    --b1
                    Content-Type: text/plain; charset=utf-8
                    Content-Transfer-Encoding: quoted-printable

                    Hi! Could we move Wednesday=E2=80=99s review to 11:00? I=E2=80=99ll bring the new =
                    onboarding screens.
                    Thanks, Priya
                    --b1
                    Content-Type: text/html; charset=utf-8

                    <p>Hi! Could we move Wednesday&rsquo;s review to 11:00?</p>
                    <img src="https://t.example.net/open/abc.gif" width="1" height="1">
                    <img src="https://mail.example.net/wf/open?u=9" style="display:none">
                    <img src="https://cdn.example.net/banner.png" width="600">
                    --b1--
                    """),
            Message(rowID: 2_345, messageID: "<redlines@example.com>", name: "Tom Okafor", address: "tom@example.com",
                    subject: "Contract redlines, round 2", summary: "Two changes left in section 4.",
                    date: received.addingTimeInterval(-24 * 60), mailbox: "imap://\(account)/INBOX", flagged: true,
                    raw: """
                    From: =?UTF-8?B?VG9tIE9rYWZvcg==?= <tom@example.com>
                    Subject: =?utf-8?Q?Contract_redlines,_round_2?=
                    Content-Type: text/html; charset=iso-8859-1
                    Content-Transfer-Encoding: base64

                    \(Data("<div>Two changes left in section 4. Can you look before Friday?<br>Caf\u{e9} at 3?</div>".data(using: .isoLatin1)!).base64EncodedString())
                    """),
            Message(rowID: 3, messageID: "<dental@example.com>", name: "Northside Dental", address: "hello@example.com",
                    subject: "Time for your six-month checkup", summary: "Reply with a few times that work.",
                    date: received.addingTimeInterval(-97 * 60), mailbox: "imap://\(account)/INBOX", read: true,
                    raw: "Subject: Time for your six-month checkup\n\nReply with a few times that work and we'll confirm.\n"),
            Message(rowID: 4, messageID: "<deleted@example.com>", name: "Gone", address: "gone@example.com",
                    subject: "Deleted", summary: "", date: received, mailbox: "imap://\(account)/INBOX",
                    deleted: true, raw: "Subject: Deleted\n\nx\n"),
            Message(rowID: 5, messageID: "<sent@example.com>", name: "You", address: "me@example.com",
                    subject: "Re: Lisbon flights", summary: "Thanks!", date: received.addingTimeInterval(-3_600),
                    mailbox: "imap://\(account)/Sent%20Messages", read: true,
                    recipients: [("Lena Fischer", "lena@example.com", 0)],
                    raw: "Subject: Re: Lisbon flights\n\nThanks!\n"),
            Message(rowID: 6, messageID: "<prize@example.com>", name: "Prize Desk", address: "win@example.net",
                    subject: "You won!", summary: "Claim now.", date: received.addingTimeInterval(-7_200),
                    mailbox: "imap://\(secondAccount)/Junk", raw: "Subject: You won!\n\nClaim now.\n"),
            Message(rowID: 7, messageID: "<old@example.com>", name: "Old Thread", address: "old@example.com",
                    subject: "Old thread", summary: "Thrown away.", date: received.addingTimeInterval(-9_000),
                    mailbox: "imap://\(account)/Deleted%20Messages", raw: "Subject: Old thread\n\nThrown away.\n"),
        ]
    }

    /// The Gmail account's mail (`make(gmail: true)`).
    static var gmailMessages: [Message] {
        let gmail = { (path: String) in "imap://\(gmailAccount)/\(path)" }
        return [
            Message(rowID: 201, messageID: "<g-inbox@example.com>", name: "Ana Silva", address: "ana@example.com",
                    subject: "Lunch on Friday?", summary: "Noon at the usual place?",
                    date: received.addingTimeInterval(60), mailbox: allMail, flagged: true,
                    labels: [gmail("INBOX"), gmail("%5BGmail%5D/Important"), gmail("%5BGmail%5D/Starred")],
                    raw: "Subject: Lunch on Friday?\n\nNoon at the usual place?\n"),
            Message(rowID: 202, messageID: "<g-sent@example.com>", name: "Me", address: "me@gmail.example",
                    subject: "Trip plans", summary: "Here is the plan.",
                    date: received.addingTimeInterval(-600), mailbox: allMail, read: true,
                    recipients: [("Lena Fischer", "lena@example.com", 0), ("", "tom@example.com", 1)],
                    labels: [gmail("%5BGmail%5D/Sent%20Mail")],
                    raw: "Subject: Trip plans\n\nHere is the plan.\n"),
            Message(rowID: 203, messageID: "<g-archived@example.com>", name: "Old News", address: "news@example.com",
                    subject: "Archived", summary: "", date: received.addingTimeInterval(-86_400), mailbox: allMail,
                    read: true, raw: "Subject: Archived\n\nx\n"),
            Message(rowID: 204, messageID: "<g-label@example.com>", name: "Notes", address: "notes@example.com",
                    subject: "Kept in my Drafts label", summary: "", date: received.addingTimeInterval(-7_200),
                    mailbox: allMail, labels: [gmail("Drafts")], raw: "Subject: Kept\n\nx\n"),
        ]
    }

    /// Gmail mailboxes Mail lists that hold no message of their own.
    static var gmailEmptyMailboxes: [String] {
        ["imap://\(gmailAccount)/%5BGmail%5D", "imap://\(gmailAccount)/archinve"]
    }

    /// `extraInbox` adds that many newer inbox messages (bodies not written),
    /// to push older mailboxes' mail out of any global window. `gmail` adds
    /// the Gmail account.
    static func make(version: String = "V10", schema: Schema = .current, extraInbox: Int = 0,
                     gmail: Bool = false) throws -> MailStoreFixture {
        let root = URL.temporaryDirectory.appending(path: "mail-fixture-\(UUID().uuidString)", directoryHint: .isDirectory)
        let mailData = root.appending(path: "\(version)/MailData", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: mailData, withIntermediateDirectories: true)
        var db: OpaquePointer?
        guard sqlite3_open(mailData.appending(path: "Envelope Index").path, &db) == SQLITE_OK else {
            throw MailSourceError.unreadable("fixture open")
        }
        defer { sqlite3_close(db) }
        func exec(_ sql: String) throws {
            var error: UnsafeMutablePointer<CChar>?
            guard sqlite3_exec(db, sql, nil, nil, &error) == SQLITE_OK else {
                let message = error.map { String(cString: $0) } ?? "?"
                sqlite3_free(error)
                throw MailSourceError.unreadable(message)
            }
        }
        let messageColumns = schema == .missingDeleted
            ? "ROWID INTEGER PRIMARY KEY, sender INTEGER, subject INTEGER, subject_prefix TEXT, summary INTEGER, date_received INTEGER, mailbox INTEGER, read INTEGER, global_message_id INTEGER"
            : "ROWID INTEGER PRIMARY KEY, sender INTEGER, subject INTEGER, subject_prefix TEXT, summary INTEGER, date_received INTEGER, mailbox INTEGER, read INTEGER, deleted INTEGER, global_message_id INTEGER, flagged INTEGER"
        try exec("""
            CREATE TABLE messages (\(messageColumns));
            CREATE TABLE subjects (ROWID INTEGER PRIMARY KEY, subject TEXT);
            CREATE TABLE addresses (ROWID INTEGER PRIMARY KEY, address TEXT, comment TEXT);
            CREATE TABLE mailboxes (ROWID INTEGER PRIMARY KEY, url TEXT);
            CREATE TABLE summaries (ROWID INTEGER PRIMARY KEY, summary TEXT);
            CREATE TABLE message_global_data (ROWID INTEGER PRIMARY KEY, message_id_header TEXT);
            CREATE TABLE labels (message_id INTEGER NOT NULL, mailbox_id INTEGER NOT NULL,
                                 PRIMARY KEY (message_id, mailbox_id));
            CREATE TABLE recipients (ROWID INTEGER PRIMARY KEY, message INTEGER, address INTEGER, type INTEGER,
                                     position INTEGER);
            """)
        let quote = { (text: String) in "'" + text.replacingOccurrences(of: "'", with: "''") + "'" }
        var extra: [Message] = []
        for index in 0..<extraInbox {
            extra.append(Message(rowID: Int64(10_000 + index), messageID: "<bulk\(index)@example.com>", name: "Bulk",
                                 address: "bulk@example.com", subject: "Bulk \(index)", summary: "",
                                 date: received.addingTimeInterval(Double(index + 1) * 60),
                                 mailbox: "imap://\(account)/INBOX", raw: "Subject: Bulk\n\nx\n"))
        }
        var mailboxIDs: [String: Int] = [:]
        func mailboxID(_ url: String) throws -> Int {
            if let id = mailboxIDs[url] { return id }
            let id = mailboxIDs.count + 1
            mailboxIDs[url] = id
            try exec("INSERT INTO mailboxes VALUES (\(id), \(quote(url)))")
            return id
        }
        let all = messages + extra + (gmail ? gmailMessages : [])
        if gmail { for url in gmailEmptyMailboxes { _ = try mailboxID(url) } }
        var recipientRow = 0
        for (index, message) in all.enumerated() {
            let n = index + 1
            let mailboxID = try mailboxID(message.mailbox)
            for label in message.labels {
                try exec("INSERT INTO labels VALUES (\(message.rowID), \(try mailboxID(label)))")
            }
            for (position, recipient) in message.recipients.enumerated() {
                recipientRow += 1
                let address = 100_000 + recipientRow
                try exec("INSERT INTO addresses VALUES (\(address), \(quote(recipient.address)), \(quote(recipient.name)))")
                try exec("INSERT INTO recipients VALUES (\(recipientRow), \(message.rowID), \(address), \(recipient.type), \(position))")
            }
            try exec("INSERT INTO subjects VALUES (\(n), \(quote(message.subject)))")
            try exec("INSERT INTO addresses VALUES (\(n), \(quote(message.address)), \(quote(message.name)))")
            try exec("INSERT INTO summaries VALUES (\(n), \(quote(message.summary)))")
            try exec("INSERT INTO message_global_data VALUES (\(n), \(quote(message.messageID)))")
            let deleted = schema == .missingDeleted ? "" : ", \(message.deleted ? 1 : 0)"
            let flagged = schema == .missingDeleted ? "" : ", \(message.flagged ? 1 : 0)"
            try exec("""
                INSERT INTO messages VALUES (\(message.rowID), \(n), \(n), '', \(n), \(Int(message.date.timeIntervalSince1970)), \
                \(mailboxID), \(message.read ? 1 : 0)\(deleted), \(n)\(flagged))
                """)
            let mailbox = MailboxRef(url: message.mailbox)
            var directory = root.appending(path: "\(version)/\(mailbox.accountID)")
            for component in mailbox.path { directory.append(path: "\(component).mbox") }
            directory.append(path: "9E3B6A40-1111-4000-8000-000000000001/Data\(AppleMailSource.shardPath(for: message.rowID))/Messages")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let raw = Data(message.raw.utf8)
            let plist = Data("<?xml version=\"1.0\"?><plist version=\"1.0\"><dict/></plist>".utf8)
            try (Data("\(raw.count)\n".utf8) + raw + plist).write(to: directory.appending(path: "\(message.rowID).emlx"))
        }
        return MailStoreFixture(root: root)
    }

    enum Schema { case current, missingDeleted }

    /// A synthetic system account list (the `ZACCOUNT` columns the reader uses).
    static func writeAccounts(at url: URL,
                              rows: [(pk: Int, id: String, user: String?, description: String?, parent: Int?)]) throws {
        var db: OpaquePointer?
        guard sqlite3_open(url.path, &db) == SQLITE_OK else { throw MailSourceError.unreadable("open") }
        defer { sqlite3_close(db) }
        let quote = { (text: String?) in text.map { "'" + $0.replacingOccurrences(of: "'", with: "''") + "'" } ?? "NULL" }
        var sql = "CREATE TABLE ZACCOUNT (Z_PK INTEGER PRIMARY KEY, ZIDENTIFIER TEXT, ZUSERNAME TEXT, "
            + "ZACCOUNTDESCRIPTION TEXT, ZPARENTACCOUNT INTEGER);"
        for row in rows {
            sql += "INSERT INTO ZACCOUNT VALUES (\(row.pk), \(quote(row.id)), \(quote(row.user)), "
                + "\(quote(row.description)), \(row.parent.map(String.init) ?? "NULL"));"
        }
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else { throw MailSourceError.unreadable("accounts") }
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    /// Mail's own writer: puts the index in WAL mode and adds mail that
    /// stays in the write-ahead log (not checkpointed) while open.
    final class WALWriter {
        private var db: OpaquePointer?

        init(index: URL) throws {
            guard sqlite3_open(index.path, &db) == SQLITE_OK else { throw MailSourceError.unreadable("writer") }
            try exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;")
        }

        func insertInboxMessage(rowID: Int64, messageID: String, subject: String) throws {
            try exec("""
                INSERT INTO subjects VALUES (\(rowID), '\(subject)');
                INSERT INTO addresses VALUES (\(rowID), 'fresh@example.com', 'Fresh');
                INSERT INTO summaries VALUES (\(rowID), '');
                INSERT INTO message_global_data VALUES (\(rowID), '\(messageID)');
                INSERT INTO messages (ROWID, sender, subject, subject_prefix, summary, date_received, mailbox, read,
                                      deleted, global_message_id)
                VALUES (\(rowID), \(rowID), \(rowID), '', \(rowID), 1900000000, 1, 0, 0, \(rowID));
                """)
        }

        func exec(_ sql: String) throws {
            guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else {
                throw MailSourceError.unreadable(String(cString: sqlite3_errmsg(db)))
            }
        }

        func close() {
            sqlite3_close(db)
            db = nil
        }
    }
}
