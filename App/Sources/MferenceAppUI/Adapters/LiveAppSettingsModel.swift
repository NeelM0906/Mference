import AppKit
import Foundation
import MferenceAppCore
import Observation

/// Where Settings > Connections reads Apple connection states: U4's
/// `AppleConnections` in the app, `BrokerConnections` where Apple services
/// are not built.
public protocol AppleConnectionsProviding: AnyObject {
    /// Re-reads every state; never prompts.
    func refresh() async
    func connectionState(_ kind: ConnectionKind) -> ConnectionState
    /// Asks for access (reason first, then the system prompt or pane).
    func connect(_ kind: ConnectionKind)
    /// Reminders sync was switched; turning it on asks for Reminders (L9).
    func remindersSyncSwitched(_ enabled: Bool)
}

/// Connection states straight from the `PermissionBroker`: Calendar access
/// for Apple Calendar, Full Disk Access for Apple Mail. Reads never prompt
/// (L6); `refresh()` re-reads them, for example when the app becomes active.
@Observable
public final class BrokerConnections: AppleConnectionsProviding {
    public private(set) var calendar: PermissionState = .notDetermined
    public private(set) var mail: PermissionState = .notDetermined
    /// Mail shows "Open System Settings" only after the person tried to
    /// connect; before that it is simply not connected.
    public private(set) var askedForMail = false

    @ObservationIgnored private let broker: any PermissionBroker
    /// Only the newest refresh writes: an older one that read before a grant
    /// must not overwrite what a newer one read after it.
    @ObservationIgnored private var refreshGeneration: UInt64 = 0

    public init(broker: any PermissionBroker) {
        self.broker = broker
    }

    public func refresh() async {
        refreshGeneration &+= 1
        let generation = refreshGeneration
        let calendar = await broker.state(of: .calendar)
        let mail = await broker.state(of: .fullDiskAccess)
        guard generation == refreshGeneration else { return }
        self.calendar = calendar
        self.mail = mail
    }

    public func connectionState(_ kind: ConnectionKind) -> ConnectionState {
        switch kind {
        case .appleCalendar:
            Self.state(calendar)
        case .appleMail:
            mail == .granted ? .connected : (askedForMail ? .needsSystemSettings : .notConnected)
        }
    }

    public func connect(_ kind: ConnectionKind) {
        Task {
            switch kind {
            case .appleCalendar:
                calendar = await broker.request(.calendar)
                if calendar == .needsSystemSettings || calendar == .denied {
                    await broker.openSystemSettings(for: .calendar)
                }
            case .appleMail:
                askedForMail = true
                mail = await broker.request(.fullDiskAccess)
            }
        }
    }

    public func remindersSyncSwitched(_ enabled: Bool) {
        guard enabled else { return }
        Task { _ = await broker.request(.reminders) }
    }

    static func state(_ permission: PermissionState) -> ConnectionState {
        switch permission {
        case .granted: .connected
        case .notDetermined: .notConnected
        case .denied, .needsSystemSettings: .needsSystemSettings
        }
    }
}

/// U4's connections behind the settings adapter. An unsupported Mail store
/// shows as Not supported with U4's reason and no button: no setting fixes it.
extension AppleConnections: AppleConnectionsProviding {
    public func connectionState(_ kind: ConnectionKind) -> ConnectionState {
        switch kind {
        case .appleCalendar: Self.connectionState(status.calendar)
        case .appleMail: Self.connectionState(status.mail, kind: .appleMail)
        }
    }

    public func connect(_ kind: ConnectionKind) {
        Task {
            switch kind {
            case .appleCalendar: await connectCalendar()
            case .appleMail: await connectMail()
            }
        }
    }

    public func remindersSyncSwitched(_ enabled: Bool) {
        Task { await remindersSyncChanged(enabled) }
    }

    /// The row shows the person-facing sentence, not the store diagnostic:
    /// for Mail that is `MailStoreStatus`'s own wording.
    static func connectionState(_ access: AppleConnectionStatus.Access,
                                kind: ConnectionKind = .appleCalendar) -> ConnectionState {
        switch access {
        case .notConnected: .notConnected
        case .connected: .connected
        case .needsSystemSettings: .needsSystemSettings
        case .unsupported(let detail):
            .unsupported(detail: kind == .appleMail
                ? MailStoreStatus.unsupported(detail: detail).userMessage ?? detail
                : detail)
        }
    }
}

/// `AppSettingsModel` over U2's `AppSettings` and an Apple connections source.
/// Connection states are re-read (never prompting) when it is built and each
/// time the app becomes active, so a grant or revocation made in System
/// Settings (Full Disk Access above all) shows without another click.
@Observable
public final class LiveAppSettingsModel: AppSettingsModel {
    @ObservationIgnored private let settings: AppSettings
    @ObservationIgnored private let connections: any AppleConnectionsProviding
    @ObservationIgnored private var activation: (any NSObjectProtocol)?

    public init(settings: AppSettings, connections: any AppleConnectionsProviding,
                refreshOn activationNotification: Notification.Name? = NSApplication.didBecomeActiveNotification) {
        self.settings = settings
        self.connections = connections
        Task { await connections.refresh() }
        if let activationNotification {
            // Runs on the posting thread and hops to the main actor itself,
            // so it does not depend on the main operation queue's turn.
            activation = NotificationCenter.default.addObserver(
                forName: activationNotification, object: nil, queue: nil) { [weak self] _ in
                    Task { @MainActor in
                        guard let connections = self?.connections else { return }
                        await connections.refresh()
                    }
                }
        }
    }

    isolated deinit {
        if let activation { NotificationCenter.default.removeObserver(activation) }
    }

    public var webSearchEnabled: Bool {
        get { settings.webSearchEnabled }
        set { settings.webSearchEnabled = newValue }
    }

    public var offlineMode: Bool {
        get { settings.offlineMode }
        set { settings.offlineMode = newValue }
    }

    public var blockRemoteMailContent: Bool {
        get { settings.blockRemoteMailContent }
        set { settings.blockRemoteMailContent = newValue }
    }

    public var remindersSyncEnabled: Bool {
        get { settings.remindersSyncEnabled }
        set {
            settings.remindersSyncEnabled = newValue
            connections.remindersSyncSwitched(newValue)
        }
    }

    public var useAppleModel: Bool {
        get { settings.useAppleModel }
        set { settings.useAppleModel = newValue }
    }

    public func connectionState(_ kind: ConnectionKind) -> ConnectionState {
        connections.connectionState(kind)
    }

    public func connect(_ kind: ConnectionKind) {
        connections.connect(kind)
    }
}

extension ScreenModels {
    /// The app's screen models over U2's composition root. Starts the model
    /// library following the engine and downloads for the app's lifetime;
    /// connection states come from U4's `AppleConnections` (never prompting,
    /// L6).
    public static func live(services: AppServices) -> ScreenModels {
        let library = ModelLibrary(engine: services.engine, downloads: services.downloads,
                                   settings: services.settings)
        Task { await library.run() }
        let connections: any AppleConnectionsProviding = services.apple?.connections
            ?? BrokerConnections(broker: services.permissions)
        return ScreenModels(
            library: LiveModelLibraryModel(library: library, settings: services.settings,
                                           downloads: services.downloads),
            memory: LiveMemoryListModel(store: MemoryStore(context: services.modelContainer.mainContext)),
            settings: LiveAppSettingsModel(settings: services.settings, connections: connections))
    }
}
