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
        try #require(inbox.count == 2)
        #expect(inbox.map(\.subject) == ["Thursday design review", "Five Mac shortcuts"])
        #expect(inbox[0].senderName == "Priya Raman")
        #expect(inbox[0].senderAddress == "priya@lumenstudio.example")
        #expect(inbox[0].isRead == false)
        #expect(inbox[1].senderName == "The Weekly Byte")
        #expect(inbox[1].isRead == true)

        let sent = try await source.messages(in: [.sent], limit: 10)
        try #require(sent.count == 1)
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

    /// Review R1-1: the demo planner must never be the person's own Planner.json, or real to-dos
    /// appear in captures and the in-memory Reminders can delete them.
    @Test func demoPlannerLivesInTheDemoDirectory() throws {
        let root = try makeDemo()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = DemoData.plannerURL(in: root)
        #expect(url.path.hasPrefix(root.path), "demo planner at \(url.path), outside \(root.path)")
    }

    /// Review R1-2: a set but missing demo path fails closed (empty demo data), never falling back
    /// to the person's real Mail and Calendar.
    @Test func setButMissingDemoPathFailsClosed() {
        let missing = "/nonexistent-mference-demo-\(UUID().uuidString)"
        #expect(DemoData.directory(environment: [DemoData.variable: missing])?.path == missing)
        #expect(DemoData.directory(environment: [:]) == nil)
        #expect(DemoData.directory(environment: [DemoData.variable: ""]) == nil)
    }

    /// Review R2-M3, M1, L1, L2: the wiring `AppleServices.live` uses. Demo mode reads and writes
    /// only the demo directory: in-memory Mail and Calendar, a sender that records, Mail.app never
    /// launched, Reminders choices in their own settings, and the person's planner never built.
    @Test func demoWiringTouchesNothingOfThePerson() throws {
        let root = try makeDemo()
        defer { try? FileManager.default.removeItem(at: root) }
        var builtLivePlanner = false
        let sources = AppleSources.make(demo: root, broker: PreviewPermissionBroker(),
                                        livePlanner: { builtLivePlanner = true; return PlannerStore(url: nil) })
        #expect(sources.calendar is InMemoryCalendarService)
        #expect(sources.mail is InMemoryMailSource)
        #expect(sources.sender is RecordingMailSender)
        #expect(sources.mailApp is FixedMailApp, "demo mode must not launch or watch the real Mail.app")
        #expect(sources.settings !== UserDefaults.standard, "demo Reminders choices must not reach the real settings")
        #expect(!builtLivePlanner, "demo mode must not read the person's Planner.json")
        #expect(sources.planner.snapshot.todos.map(\.title) == ["Renew passport"])
    }

    /// A normal launch keeps this Mac's Mail, EventKit, planner and settings.
    @Test func normalWiringIsUnchanged() {
        var builtLivePlanner = false
        let sources = AppleSources.make(demo: nil, broker: PreviewPermissionBroker(),
                                        livePlanner: { builtLivePlanner = true; return PlannerStore(url: nil) })
        #expect(sources.calendar is EventKitService)
        #expect(sources.mail is AppleMailSource)
        #expect(sources.sender is AppleMailSender)
        #expect(sources.mailApp is SystemMailApp)
        #expect(sources.settings === UserDefaults.standard)
        #expect(builtLivePlanner)
    }

    /// Review R3-1: in demo mode the Settings switch and the calendar model share the demo suite,
    /// so flipping Reminders sync agrees on screen and never reaches the person's own settings.
    @MainActor @Test func demoSettingsAgreeAndStayOutOfTheRealDefaults() throws {
        let root = try makeDemo()
        defer { try? FileManager.default.removeItem(at: root) }
        let key = AppSettings.Key.remindersSyncEnabled
        let real = UserDefaults.standard.object(forKey: key)
        defer { UserDefaults.standard.set(real, forKey: key) }
        let sources = AppleSources.make(demo: root, broker: PreviewPermissionBroker(),
                                        livePlanner: { PlannerStore(url: nil) })
        let store = DemoData.settingsStore(demo: root)
        defer { store.removeObject(forKey: key) }
        let settings = AppSettings(store: store)
        settings.remindersSyncEnabled = !sources.settings.bool(forKey: key)
        #expect(sources.settings.bool(forKey: key) == settings.remindersSyncEnabled,
                "the demo calendar must read what demo Settings wrote")
        #expect((UserDefaults.standard.object(forKey: key) as? Bool) == (real as? Bool),
                "demo Settings must not write the real defaults")
        #expect(DemoData.settingsStore(demo: nil) === UserDefaults.standard)
    }

    @Test func addressesParseWithAndWithoutNames() {
        #expect(DemoData.parseAddress("\"Sam Carter\" <sam@x.example>") == ("Sam Carter", "sam@x.example"))
        #expect(DemoData.parseAddress("Sam <sam@x.example>") == ("Sam", "sam@x.example"))
        #expect(DemoData.parseAddress("sam@x.example") == ("", "sam@x.example"))
    }
}
