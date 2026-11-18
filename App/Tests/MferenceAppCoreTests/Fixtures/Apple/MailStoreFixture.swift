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
        var raw: String
    }

    static let account = "4F7A2C1E-0000-4000-8000-0000000000AA"
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
                    date: received.addingTimeInterval(-24 * 60), mailbox: "imap://\(account)/INBOX",
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
                    mailbox: "imap://\(account)/Sent%20Messages", raw: "Subject: Re: Lisbon flights\n\nThanks!\n"),
        ]
    }

    /// Builds the store under a new temporary folder.
    static func make(version: String = "V10", schema: Schema = .current) throws -> MailStoreFixture {
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
            : "ROWID INTEGER PRIMARY KEY, sender INTEGER, subject INTEGER, subject_prefix TEXT, summary INTEGER, date_received INTEGER, mailbox INTEGER, read INTEGER, deleted INTEGER, global_message_id INTEGER"
        try exec("""
            CREATE TABLE messages (\(messageColumns));
            CREATE TABLE subjects (ROWID INTEGER PRIMARY KEY, subject TEXT);
            CREATE TABLE addresses (ROWID INTEGER PRIMARY KEY, address TEXT, comment TEXT);
            CREATE TABLE mailboxes (ROWID INTEGER PRIMARY KEY, url TEXT);
            CREATE TABLE summaries (ROWID INTEGER PRIMARY KEY, summary TEXT);
            CREATE TABLE message_global_data (ROWID INTEGER PRIMARY KEY, message_id_header TEXT);
            """)
        let quote = { (text: String) in "'" + text.replacingOccurrences(of: "'", with: "''") + "'" }
        var mailboxIDs: [String: Int] = [:]
        for (index, message) in messages.enumerated() {
            let n = index + 1
            let mailboxID = mailboxIDs[message.mailbox] ?? (mailboxIDs.count + 1)
            if mailboxIDs[message.mailbox] == nil {
                mailboxIDs[message.mailbox] = mailboxID
                try exec("INSERT INTO mailboxes VALUES (\(mailboxID), \(quote(message.mailbox)))")
            }
            try exec("INSERT INTO subjects VALUES (\(n), \(quote(message.subject)))")
            try exec("INSERT INTO addresses VALUES (\(n), \(quote(message.address)), \(quote(message.name)))")
            try exec("INSERT INTO summaries VALUES (\(n), \(quote(message.summary)))")
            try exec("INSERT INTO message_global_data VALUES (\(n), \(quote(message.messageID)))")
            let deleted = schema == .missingDeleted ? "" : ", \(message.deleted ? 1 : 0)"
            try exec("""
                INSERT INTO messages VALUES (\(message.rowID), \(n), \(n), '', \(n), \(Int(message.date.timeIntervalSince1970)), \
                \(mailboxID), \(message.read ? 1 : 0)\(deleted), \(n))
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

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}
