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
    /// Apple Mail on this Mac; always `.notConnected` on the iPhone.
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

    #if os(macOS)
    public static func access(_ status: MailStoreStatus) -> Access {
        switch status {
        case .available: .connected
        case .notFound: .notConnected
        case .needsFullDiskAccess: .needsSystemSettings
        case .unsupported(let detail): .unsupported(detail: detail)
        }
    }
    #endif
}

/// The connection state and actions for Calendar, Reminders and Mail. Reading
/// never prompts; each `connect…` asks through the `PermissionBroker` (reason
/// first) and falls back to the matching System Settings pane. On the iPhone
/// there is no Mail: Calendar and Reminders only.
@MainActor @Observable
public final class AppleConnections {
    public private(set) var status = AppleConnectionStatus.unknown

    @ObservationIgnored private let calendar: any CalendarServicing
    #if os(macOS)
    @ObservationIgnored private let mail: any MailSource
    #endif
    @ObservationIgnored private let broker: any PermissionBroker

    #if os(macOS)
    public init(calendar: any CalendarServicing, mail: any MailSource, broker: any PermissionBroker) {
        self.calendar = calendar
        self.mail = mail
        self.broker = broker
    }

    /// The app's connections: EventKit and this Mac's Apple Mail store.
    public static func live(broker: any PermissionBroker) -> AppleConnections {
        AppleConnections(calendar: EventKitService(broker: broker), mail: AppleMailSource(), broker: broker)
    }

    /// Reads every state without prompting (and takes a fresh look at the
    /// Mail store, so a Full Disk Access grant shows at once).
    public func refresh() async {
        let calendar = self.calendar, mail = self.mail
        await mail.reload()
        async let calendarState = calendar.calendarAccess()
        async let remindersState = calendar.remindersAccess()
        async let mailStatus = mail.status()
        status = AppleConnectionStatus(calendar: AppleConnectionStatus.access(await calendarState),
                                       reminders: AppleConnectionStatus.access(await remindersState),
                                       mail: AppleConnectionStatus.access(await mailStatus))
    }
    #else
    public init(calendar: any CalendarServicing, broker: any PermissionBroker) {
        self.calendar = calendar
        self.broker = broker
    }

    /// The app's connections: EventKit.
    public static func live(broker: any PermissionBroker) -> AppleConnections {
        AppleConnections(calendar: EventKitService(broker: broker), broker: broker)
    }

    /// Reads every state without prompting.
    public func refresh() async {
        let calendar = self.calendar
        async let calendarState = calendar.calendarAccess()
        async let remindersState = calendar.remindersAccess()
        status = AppleConnectionStatus(calendar: AppleConnectionStatus.access(await calendarState),
                                       reminders: AppleConnectionStatus.access(await remindersState),
                                       mail: .notConnected)
    }
    #endif

    /// "Connect" on the Calendar row.
    public func connectCalendar() async {
        let state = await calendar.requestCalendarAccess()
        if state == .denied || state == .needsSystemSettings {
            await broker.openSystemSettings(for: .calendar)
        }
        await refresh()
    }

    #if os(macOS)
    /// "Connect" on the Mail row: Full Disk Access cannot be granted in a
    /// dialog, so after the reason the broker opens its System Settings pane
    /// (one request; this does not open it a second time).
    public func connectMail() async {
        if case .available = await mail.status() {
            await refresh()
            return
        }
        // The broker shows the reason, then opens the Full Disk Access pane.
        settingsVisit.begin()
        _ = await broker.request(.fullDiskAccess)
        settingsVisit.end()
        await refresh()
    }

    /// Mference left the foreground: during a Mail request, that is the Full
    /// Disk Access pane opening rather than "Not now".
    public func appResignedActive() {
        settingsVisit.appLeft()
    }

    private var settingsVisit = SettingsVisit()

    /// Full Disk Access was opened from Settings but Mail is still not
    /// readable: macOS applies it from the next launch, so Settings offers
    /// Quit and Reopen (`AppRelaunch.quitAndReopen()` in MferenceAppUI).
    public var mailNeedsRelaunch: Bool {
        status.mail == .needsSystemSettings && settingsVisit.happened
    }
    #endif

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
