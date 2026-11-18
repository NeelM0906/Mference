import EventKit
import Foundation
@testable import MferenceAppCore
import Testing

/// Checkpoint C-B2's live round trip on this Mac's Calendar: create, edit and
/// delete one clearly named event a year from now. Gated: it triggers the
/// Calendar privacy prompt, which the maintainer approves. Run only with
/// `MFERENCE_EVENTKIT_LIVE=1`; writes a short report to
/// `MFERENCE_EVENTKIT_LIVE_REPORT` when set.
@Suite struct AppleCalendarLiveTests {
    static let enabled = ProcessInfo.processInfo.environment["MFERENCE_EVENTKIT_LIVE"] == "1"

    /// Asks EventKit directly (the app's broker is U2's and needs the bundle).
    struct DirectBroker: PermissionBroker {
        func state(of permission: AppPermission) async -> PermissionState { .notDetermined }
        func request(_ permission: AppPermission) async -> PermissionState {
            let store = EKEventStore()
            let granted: Bool
            if permission == .reminders {
                granted = (try? await store.requestFullAccessToReminders()) ?? false
            } else {
                granted = (try? await store.requestFullAccessToEvents()) ?? false
            }
            return granted ? .granted : .denied
        }
        func openSystemSettings(for permission: AppPermission) async {}
    }

    @Test(.enabled(if: enabled))
    func createEditDeleteRoundTrip() async throws {
        let service = EventKitService(broker: DirectBroker())
        let access = await service.requestCalendarAccess()
        try #require(access == .granted, "Calendar access: \(access)")
        let start = Calendar.current.date(byAdding: .year, value: 1, to: .now)!
        let title = "Mference round-trip check \(UUID().uuidString.prefix(8))"
        var log: [String] = []

        let created = try await service.add(ProposedEvent(title: title, start: start, end: start.addingTimeInterval(1_800),
                                                          source: ItemSource(kind: .manual, label: "C-B2 check")))
        log.append("created \(created.id) in \(created.calendarName)")
        var found = try await service.events(from: start.addingTimeInterval(-60), to: start.addingTimeInterval(3_600))
        #expect(found.contains { $0.id == created.id && $0.title == title })

        let edited = try await service.update(eventID: created.id, title: title + " (edited)",
                                              start: start.addingTimeInterval(3_600), end: start.addingTimeInterval(5_400))
        log.append("edited to \(edited.start)")
        found = try await service.events(from: start, to: start.addingTimeInterval(7_200))
        #expect(found.contains { $0.id == created.id && $0.title.hasSuffix("(edited)") })

        try await service.remove(eventID: created.id)
        found = try await service.events(from: start.addingTimeInterval(-60), to: start.addingTimeInterval(7_200))
        #expect(!found.contains { $0.id == created.id })
        log.append("deleted; window empty of the test event: \(!found.contains { $0.id == created.id })")

        if let path = ProcessInfo.processInfo.environment["MFERENCE_EVENTKIT_LIVE_REPORT"] {
            try (log.joined(separator: "\n") + "\n").write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
