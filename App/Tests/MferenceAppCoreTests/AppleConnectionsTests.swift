import Foundation
@testable import MferenceAppCore
import Testing

@Suite @MainActor struct AppleConnectionsTests {
    @Test func readingNeverPrompts() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let calendar = InMemoryCalendarService(calendarAccess: .granted, remindersAccess: .notDetermined)
        let broker = RecordingPermissionBroker()
        let connections = AppleConnections(calendar: calendar, mail: AppleMailSource(mailRoot: fixture.root),
                                           broker: broker)
        #expect(connections.status == .unknown)
        await connections.refresh()
        #expect(connections.status == AppleConnectionStatus(calendar: .connected, reminders: .notConnected,
                                                            mail: .connected))
        #expect(await calendar.requests.isEmpty)
        #expect(broker.requests.isEmpty)
    }

    @Test func connectingCalendarAsksThenPointsToSettingsWhenDenied() async {
        let granted = InMemoryCalendarService(calendarAccess: .notDetermined)
        let broker = RecordingPermissionBroker()
        let missingMail = AppleMailSource(mailRoot: URL.temporaryDirectory.appending(path: "none-\(UUID().uuidString)"))
        let connections = AppleConnections(calendar: granted, mail: missingMail, broker: broker)
        await connections.connectCalendar()
        #expect(connections.status.calendar == .connected)
        #expect(connections.status.mail == .notConnected)
        #expect(await granted.requests == [.calendar])
        #expect(broker.openedSettings.isEmpty)

        let denied = InMemoryCalendarService(calendarAccess: .notDetermined, grantOnRequest: false)
        let deniedConnections = AppleConnections(calendar: denied, mail: missingMail, broker: broker)
        await deniedConnections.connectCalendar()
        #expect(deniedConnections.status.calendar == .needsSystemSettings)
        #expect(broker.openedSettings == [.calendar])
    }

    @Test func connectingMailOpensFullDiskAccess() async throws {
        let fixture = try MailStoreFixture.make()
        defer { fixture.remove() }
        let index = fixture.root.appending(path: "V10/MailData/Envelope Index")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: index.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: index.path) }
        let broker = RecordingPermissionBroker([.fullDiskAccess: .needsSystemSettings])
        let connections = AppleConnections(calendar: InMemoryCalendarService(),
                                           mail: AppleMailSource(mailRoot: fixture.root), broker: broker)
        await connections.connectMail()
        #expect(broker.requests == [.fullDiskAccess])
        #expect(broker.openedSettings == [.fullDiskAccess])
        #expect(connections.status.mail == .needsSystemSettings)
    }

    @Test func unsupportedMailIsItsOwnState() async throws {
        let old = try MailStoreFixture.make(schema: .missingDeleted)
        defer { old.remove() }
        let connections = AppleConnections(calendar: InMemoryCalendarService(),
                                           mail: AppleMailSource(mailRoot: old.root), broker: RecordingPermissionBroker())
        await connections.refresh()
        guard case .unsupported = connections.status.mail else { Issue.record("expected unsupported"); return }
    }

    @Test func turningRemindersSyncOnAsksOnceAndOffAsksNothing() async {
        let calendar = InMemoryCalendarService(remindersAccess: .notDetermined)
        let broker = RecordingPermissionBroker()
        let connections = AppleConnections(calendar: calendar, mail: AppleMailSource(mailRoot: URL.temporaryDirectory
            .appending(path: "none-\(UUID().uuidString)")), broker: broker)
        await connections.remindersSyncChanged(false)
        #expect(await calendar.requests.isEmpty)
        await connections.remindersSyncChanged(true)
        #expect(await calendar.requests == [.reminders])
        #expect(connections.status.reminders == .connected)
        await connections.remindersSyncChanged(true)
        #expect(await calendar.requests == [.reminders])
    }

    @Test func liveServicesBuildWithoutReadingOrAsking() throws {
        let container = try AppModelContainer.inMemory()
        let broker = RecordingPermissionBroker()
        let engine = GatedEngine(gate: ScanTestLatch(), answer: "")
        let services = AppleServices.live(engine: engine, broker: broker, container: container,
                                          plannerStore: PlannerStore(url: nil))
        #expect(services.scan.phase == .empty)
        #expect(services.calendar.access == .notAsked)
        #expect(services.mail.status == nil)
        #expect(services.connections.status == .unknown)
        #expect(broker.requests.isEmpty)
    }
}
