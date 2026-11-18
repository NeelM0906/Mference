import Foundation
import Observation

/// What Settings > Connections shows for the Apple apps (U1's rows, read by
/// U2's settings adapter).
public struct AppleConnectionStatus: Sendable, Hashable {
    public enum Access: Sendable, Hashable {
        /// Never asked, or Mail is not set up on this Mac.
        case notConnected
        case connected
        /// Denied, or Full Disk Access: only System Settings can change it.
        case needsSystemSettings
        /// The Mail store's layout is one Mference cannot read; `detail` is
        /// for diagnostics, not the UI.
        case unsupported(detail: String)
    }

    public var calendar: Access
    /// Reminders access; asked only when the person turns sync on (L9).
    public var reminders: Access
    public var mail: Access

    public init(calendar: Access, reminders: Access, mail: Access) {
        self.calendar = calendar
        self.reminders = reminders
        self.mail = mail
    }

    /// Before the first `refresh()`.
    public static let unknown = AppleConnectionStatus(calendar: .notConnected, reminders: .notConnected,
                                                      mail: .notConnected)

    public static func access(_ state: PermissionState) -> Access {
        switch state {
        case .granted: .connected
        case .notDetermined: .notConnected
        case .denied, .needsSystemSettings: .needsSystemSettings
        }
    }

    public static func access(_ status: MailStoreStatus) -> Access {
        switch status {
        case .available: .connected
        case .notFound: .notConnected
        case .needsFullDiskAccess: .needsSystemSettings
        case .unsupported(let detail): .unsupported(detail: detail)
        }
    }
}

/// The connection state and actions for Calendar, Reminders and Mail. Reading
/// never prompts; each `connect…` asks through the `PermissionBroker` (reason
/// first) and falls back to the matching System Settings pane.
@MainActor @Observable
public final class AppleConnections {
    public private(set) var status = AppleConnectionStatus.unknown

    @ObservationIgnored private let calendar: any CalendarServicing
    @ObservationIgnored private let mail: any MailSource
    @ObservationIgnored private let broker: any PermissionBroker

    public init(calendar: any CalendarServicing, mail: any MailSource, broker: any PermissionBroker) {
        self.calendar = calendar
        self.mail = mail
        self.broker = broker
    }

    /// The app's connections: EventKit and this Mac's Apple Mail store.
    public static func live(broker: any PermissionBroker) -> AppleConnections {
        AppleConnections(calendar: EventKitService(broker: broker), mail: AppleMailSource(), broker: broker)
    }

    /// Reads every state without prompting.
    public func refresh() async {
        let calendar = self.calendar, mail = self.mail
        async let calendarState = calendar.calendarAccess()
        async let remindersState = calendar.remindersAccess()
        async let mailStatus = mail.status()
        status = AppleConnectionStatus(calendar: AppleConnectionStatus.access(await calendarState),
                                       reminders: AppleConnectionStatus.access(await remindersState),
                                       mail: AppleConnectionStatus.access(await mailStatus))
    }

    /// "Connect" on the Calendar row.
    public func connectCalendar() async {
        let state = await calendar.requestCalendarAccess()
        if state == .denied || state == .needsSystemSettings {
            await broker.openSystemSettings(for: .calendar)
        }
        await refresh()
    }

    /// "Connect" on the Mail row: Full Disk Access cannot be granted in a
    /// dialog, so after the reason the broker opens its System Settings pane.
    public func connectMail() async {
        if case .available = await mail.status() {
            await refresh()
            return
        }
        if await broker.request(.fullDiskAccess) != .granted {
            await broker.openSystemSettings(for: .fullDiskAccess)
        }
        await refresh()
    }

    /// The Reminders sync switch changed (the switch itself is the
    /// `remindersSyncEnabled` setting). Turning it on asks for Reminders.
    public func remindersSyncChanged(_ enabled: Bool) async {
        if enabled, await calendar.remindersAccess() != .granted {
            let state = await calendar.requestRemindersAccess()
            if state == .denied || state == .needsSystemSettings {
                await broker.openSystemSettings(for: .reminders)
            }
        }
        await refresh()
    }
}
