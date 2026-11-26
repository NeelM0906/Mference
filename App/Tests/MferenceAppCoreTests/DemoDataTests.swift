import Foundation
@testable import MferenceAppCore
import Testing

/// `MFERENCE_DEMO_DATA`: demo mail, calendar and to-dos load into the in-memory sources the
/// README captures use; nothing reads Apple Mail, EventKit or the person's planner.
@Suite struct DemoDataTests {
    private func makeDemo() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: "demo-\(UUID().uuidString)")
        let fm = FileManager.default
        try fm.createDirectory(at: root.appending(path: "mail/sent"), withIntermediateDirectories: true)
        try fm.createDirectory(at: root.appending(path: "calendar"), withIntermediateDirectories: true)
        try """
        From: Priya Raman <priya@lumenstudio.example>\r
        To: alex@morgan.example\r
        Subject: Thursday design review\r
        Date: Wed, 25 Nov 2026 09:42:00 -0500\r
        Message-ID: <a1@demo.example>\r
        Content-Type: text/plain; charset="utf-8"\r
        \r
        Could we move Thursday's review to 2:00?\r
        """.write(to: root.appending(path: "mail/00-unseen.eml"), atomically: true, encoding: .utf8)
        try """
        From: "The Weekly Byte" <news@weeklybyte.example>\r
        To: alex@morgan.example\r
        Subject: Five Mac shortcuts\r
        Date: Tue, 24 Nov 2026 07:00:00 -0500\r
        Message-ID: <a2@demo.example>\r
        Content-Type: text/html\r
        \r
        <p>Shortcuts</p><img src="https://weeklybyte.example/track/open.gif" width="1" height="1">\r
        """.write(to: root.appending(path: "mail/01-seen.eml"), atomically: true, encoding: .utf8)
        try """
        From: alex@morgan.example\r
        To: Dana Whitfield <dana@maplecourt.example>\r
        Subject: Leak in the bathroom\r
        Date: Mon, 23 Nov 2026 16:40:00 -0500\r
        Message-ID: <a3@demo.example>\r
        \r
        There's a leak.\r
        """.write(to: root.appending(path: "mail/sent/00-seen.eml"), atomically: true, encoding: .utf8)
        try "BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:e1\r\nDTSTART:20261126T100000\r\nDTEND:20261126T104500\r\n"
            .appending("SUMMARY:Design review\r\nLOCATION:Studio\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n")
            .write(to: root.appending(path: "calendar/demo-week.ics"), atomically: true, encoding: .utf8)
        try "2026-11-27\tRenew passport\nnot a line\n"
            .write(to: root.appending(path: "calendar/todos.tsv"), atomically: true, encoding: .utf8)
        return root
    }

    @Test func mailLoadsInboxAndSentWithReadState() async throws {
        let root = try makeDemo()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = DemoData.mailSource(in: root)

        let inbox = try await source.messages(in: [.inbox], limit: 10)
        #expect(inbox.map(\.subject) == ["Thursday design review", "Five Mac shortcuts"])
        #expect(inbox[0].senderName == "Priya Raman")
        #expect(inbox[0].senderAddress == "priya@lumenstudio.example")
        #expect(inbox[0].isRead == false)
        #expect(inbox[1].senderName == "The Weekly Byte")
        #expect(inbox[1].isRead == true)

        let sent = try await source.messages(in: [.sent], limit: 10)
        #expect(sent.map(\.subject) == ["Leak in the bathroom"])
        #expect(sent[0].isOutgoing)
        #expect(sent[0].recipients.first?.address == "dana@maplecourt.example")

        let accounts = try await source.accounts()
        #expect(accounts.map(\.address) == ["alex@morgan.example"])

        // The same parser as real mail: HTML reduced to text, the pixel counted.
        let body = try await source.body(of: inbox[1])
        #expect(body.trackersRemoved == 1)
        #expect(body.text.contains("Shortcuts"))
    }

    @Test func calendarReadsFloatingTimesAndLocation() async throws {
        let root = try makeDemo()
        defer { try? FileManager.default.removeItem(at: root) }
        let service = DemoData.calendarService(in: root)
        var components = DateComponents(year: 2026, month: 11, day: 26, hour: 10)
        components.timeZone = .current
        let start = try #require(Calendar(identifier: .gregorian).date(from: components))

        let events = try await service.events(from: start.addingTimeInterval(-86_400),
                                              to: start.addingTimeInterval(86_400))
        #expect(events.map(\.title) == ["Design review"])
        #expect(events.first?.start == start)
        #expect(events.first?.end == start.addingTimeInterval(45 * 60))
        #expect(events.first?.location == "Studio")
        #expect(await service.calendarAccess() == .granted)
    }

    @Test func plannerStartsWithTodosOnlyWhenNew() throws {
        let root = try makeDemo()
        defer { try? FileManager.default.removeItem(at: root) }
        let storeURL = root.appending(path: "Planner.json")

        let fresh = DemoData.plannerStore(in: root, at: storeURL)
        #expect(fresh.snapshot.todos.map(\.title) == ["Renew passport"])

        try fresh.update { $0.todos.removeAll() }
        let reopened = DemoData.plannerStore(in: root, at: storeURL)
        #expect(reopened.snapshot.todos.isEmpty)
    }

    @Test func addressesParseWithAndWithoutNames() {
        #expect(DemoData.parseAddress("\"Sam Carter\" <sam@x.example>") == ("Sam Carter", "sam@x.example"))
        #expect(DemoData.parseAddress("Sam <sam@x.example>") == ("Sam", "sam@x.example"))
        #expect(DemoData.parseAddress("sam@x.example") == ("", "sam@x.example"))
    }
}
