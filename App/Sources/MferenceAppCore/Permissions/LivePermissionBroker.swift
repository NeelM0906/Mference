import AppKit
import AVFoundation
import CoreServices
import EventKit
import Foundation

/// The real `PermissionBroker` (decision L6). It never prompts on its own:
/// `state(of:)` only reads, and `request(_:)` runs only when a feature asks,
/// first showing the one-line reason through `presentReason`, then the
/// system prompt.
///
/// The bundle's `Info.plist` must carry the usage strings
/// (`NSCalendarsFullAccessUsageDescription`,
/// `NSRemindersFullAccessUsageDescription`, `NSCameraUsageDescription`,
/// `NSMicrophoneUsageDescription`, `NSAppleEventsUsageDescription`), or the
/// system refuses the request without asking.
public struct LivePermissionBroker: PermissionBroker {
    /// Shows `permission.reason` and returns whether the person chose to go
    /// on. Required: the system prompt never appears without the reason (L6).
    public typealias ReasonPresenter = @Sendable (AppPermission) async -> Bool

    /// The operating-system calls, injectable so tests never reach TCC.
    public struct System: Sendable {
        /// Reads the current state; never prompts.
        public var state: @Sendable (AppPermission) async -> PermissionState
        /// Shows the system prompt (or, for Full Disk Access, nothing: the
        /// person switches it on in System Settings).
        public var prompt: @Sendable (AppPermission) async -> Void
        public var openURL: @Sendable (URL) async -> Void

        public init(state: @escaping @Sendable (AppPermission) async -> PermissionState,
                    prompt: @escaping @Sendable (AppPermission) async -> Void,
                    openURL: @escaping @Sendable (URL) async -> Void) {
            self.state = state
            self.prompt = prompt
            self.openURL = openURL
        }

        public static func live(homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> System {
            System(
                state: { permission in
                    switch permission {
                    case .calendar: LivePermissionBroker.state(EKEventStore.authorizationStatus(for: .event))
                    case .reminders: LivePermissionBroker.state(EKEventStore.authorizationStatus(for: .reminder))
                    case .camera: LivePermissionBroker.state(AVCaptureDevice.authorizationStatus(for: .video))
                    case .microphone: LivePermissionBroker.state(AVCaptureDevice.authorizationStatus(for: .audio))
                    case .fullDiskAccess: LivePermissionBroker.fullDiskAccessState(homeDirectory: homeDirectory)
                    case .mailAutomation: await LivePermissionBroker.mailAutomationState(ask: false)
                    }
                },
                prompt: { permission in
                    switch permission {
                    case .calendar: _ = try? await EKEventStore().requestFullAccessToEvents()
                    case .reminders: _ = try? await EKEventStore().requestFullAccessToReminders()
                    case .camera: _ = await AVCaptureDevice.requestAccess(for: .video)
                    case .microphone: _ = await AVCaptureDevice.requestAccess(for: .audio)
                    case .mailAutomation: _ = await LivePermissionBroker.mailAutomationState(ask: true)
                    case .fullDiskAccess: break
                    }
                },
                openURL: { url in
                    await MainActor.run { _ = NSWorkspace.shared.open(url) }
                })
        }
    }

    private let presentReason: ReasonPresenter
    private let system: System

    public init(presentReason: @escaping ReasonPresenter, system: System = .live()) {
        self.presentReason = presentReason
        self.system = system
    }

    // MARK: PermissionBroker

    public func state(of permission: AppPermission) async -> PermissionState {
        await system.state(permission)
    }

    /// Reason first; the system prompt only if the person goes on. A
    /// declined reason leaves the state as it was and prompts nothing.
    public func request(_ permission: AppPermission) async -> PermissionState {
        let current = await state(of: permission)
        let mayAsk = current == .notDetermined || (permission == .fullDiskAccess && current != .granted)
        guard mayAsk else { return current }
        guard await presentReason(permission) else { return current }
        if permission == .fullDiskAccess {
            // There is no prompt for Full Disk Access: the person switches it
            // on in System Settings, and the app sees it on its next check.
            await openSystemSettings(for: .fullDiskAccess)
        } else {
            await system.prompt(permission)
        }
        return await state(of: permission)
    }

    public func openSystemSettings(for permission: AppPermission) async {
        await system.openURL(Self.settingsURL(for: permission))
    }

    // MARK: Mapping (pure, tested)

    /// EventKit: full access is granted; write-only (calendars a person
    /// downgraded) and a denial change only in System Settings; restricted
    /// (managed Macs) cannot change at all.
    static func state(_ status: EKAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: .notDetermined
        case .fullAccess: .granted
        case .writeOnly, .denied: .needsSystemSettings
        case .restricted: .denied
        @unknown default: .denied
        }
    }

    static func state(_ status: AVAuthorizationStatus) -> PermissionState {
        switch status {
        case .notDetermined: .notDetermined
        case .authorized: .granted
        case .denied: .needsSystemSettings
        case .restricted: .denied
        @unknown default: .denied
        }
    }

    /// `AEDeterminePermissionToAutomateTarget` results. Mail not running
    /// (`procNotFound`) leaves the answer open.
    static func automationState(_ status: OSStatus) -> PermissionState {
        switch status {
        case OSStatus(noErr): .granted
        case OSStatus(errAEEventWouldRequireUserConsent), OSStatus(procNotFound): .notDetermined
        case OSStatus(errAEEventNotPermitted): .needsSystemSettings
        default: .denied
        }
    }

    static func settingsURL(for permission: AppPermission) -> URL {
        let anchor = switch permission {
        case .calendar: "Privacy_Calendars"
        case .reminders: "Privacy_Reminders"
        case .camera: "Privacy_Camera"
        case .microphone: "Privacy_Microphone"
        case .fullDiskAccess: "Privacy_AllFiles"
        case .mailAutomation: "Privacy_Automation"
        }
        return URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)")!
    }

    // MARK: Probes

    /// Full Disk Access cannot be queried and never prompts: the probe opens
    /// a file only it allows (the per-user TCC database, which always
    /// exists) and reads nothing from it.
    static func fullDiskAccessState(homeDirectory: URL) -> PermissionState {
        let probe = homeDirectory.appending(path: "Library/Application Support/com.apple.TCC/TCC.db")
        let descriptor = open(probe.path, O_RDONLY)
        if descriptor >= 0 {
            close(descriptor)
            return .granted
        }
        return .needsSystemSettings
    }

    static let mailBundleID = "com.apple.mail"

    /// Whether the app may send Apple events to Mail. Asking shows the
    /// system prompt and blocks until the person answers, so it runs off the
    /// caller's thread; Mail is launched in the background first if needed,
    /// since the system only answers for a running target.
    static func mailAutomationState(ask: Bool) async -> PermissionState {
        if ask, !(await isMailRunning()) { await launchMailInBackground() }
        return await Task.detached {
            let descriptor = NSAppleEventDescriptor(bundleIdentifier: mailBundleID)
            let status = withExtendedLifetime(descriptor) { () -> OSStatus in
                guard let target = descriptor.aeDesc else { return OSStatus(procNotFound) }
                return AEDeterminePermissionToAutomateTarget(target, AEEventClass(typeWildCard),
                                                             AEEventID(typeWildCard), ask)
            }
            return automationState(status)
        }.value
    }

    @MainActor
    private static func isMailRunning() -> Bool {
        !NSRunningApplication.runningApplications(withBundleIdentifier: mailBundleID).isEmpty
    }

    @MainActor
    private static func launchMailInBackground() async {
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: mailBundleID) else { return }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false
        configuration.hides = true
        _ = try? await NSWorkspace.shared.openApplication(at: url, configuration: configuration)
    }
}
