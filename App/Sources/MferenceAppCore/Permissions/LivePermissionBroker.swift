import AppKit
import AVFoundation
import CoreServices
import EventKit
import Foundation
import os
import Synchronization

/// The real `PermissionBroker` (decision L6). It never prompts on its own:
/// `state(of:)` only reads, and `request(_:)` runs only when a feature asks,
/// first showing the one-line reason through `presentReason`, then the
/// system prompt.
///
/// EventKit requests go through one `EKEventStore` the broker keeps for its
/// lifetime (`eventStore`, which the Calendar service should share), not a
/// temporary store that can be released before the system answers. A system
/// prompt that fails, or leaves the state undetermined, is logged
/// (subsystem `ai.mference.app`, category `permissions`) and answered with
/// `.needsSystemSettings`; for the rest of the session the reason is not
/// shown again for that permission, so a broken prompt can never loop.
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
        /// person switches it on in System Settings). Throws what the system
        /// reports, so it is logged rather than swallowed.
        public var prompt: @Sendable (AppPermission) async throws -> Void
        public var openURL: @Sendable (URL) async -> Void

        public init(state: @escaping @Sendable (AppPermission) async -> PermissionState,
                    prompt: @escaping @Sendable (AppPermission) async throws -> Void,
                    openURL: @escaping @Sendable (URL) async -> Void) {
            self.state = state
            self.prompt = prompt
            self.openURL = openURL
        }

        /// The real calls. EventKit requests go through `eventStore`, which
        /// must outlive the request (the broker keeps it).
        public static func live(eventStore: EventStoreBox = EventStoreBox(),
                                homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser) -> System {
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
                    case .calendar: _ = try await eventStore.store.requestFullAccessToEvents()
                    case .reminders: _ = try await eventStore.store.requestFullAccessToReminders()
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
    private let storeBox: EventStoreBox
    /// Permissions whose system prompt already ran this session.
    private let prompted = PromptedPermissions()

    static let log = Logger(subsystem: "ai.mference.app", category: "permissions")

    /// - Parameters:
    ///   - eventStore: the one EventKit store used for requests; share it
    ///     with the Calendar service.
    ///   - system: nil uses the live calls over `eventStore`.
    public init(presentReason: @escaping ReasonPresenter, eventStore: EventStoreBox = EventStoreBox(),
                system: System? = nil) {
        self.presentReason = presentReason
        self.storeBox = eventStore
        self.system = system ?? .live(eventStore: eventStore)
    }

    /// The EventKit store requests go through.
    public var eventStore: EKEventStore { storeBox.store }

    // MARK: PermissionBroker

    public func state(of permission: AppPermission) async -> PermissionState {
        await system.state(permission)
    }

    /// Reason first; the system prompt only if the person goes on. A
    /// declined reason leaves the state as it was and prompts nothing. A
    /// system prompt that did not settle the state answers
    /// `.needsSystemSettings`, and the reason is not shown again this session.
    public func request(_ permission: AppPermission) async -> PermissionState {
        let current = await state(of: permission)
        let mayAsk = current == .notDetermined || (permission == .fullDiskAccess && current != .granted)
        guard mayAsk else { return current }
        if permission == .fullDiskAccess {
            guard await presentReason(permission) else { return current }
            // There is no prompt for Full Disk Access: the person switches it
            // on in System Settings, and the app sees it on its next check.
            await openSystemSettings(for: .fullDiskAccess)
            return await state(of: permission)
        }
        // The system prompt already ran this session and left it open.
        if prompted.contains(permission) {
            // Mail automation reads as undetermined whenever Mail is not
            // running, not only after a failed prompt: skip our reason, but
            // let the system call launch Mail and check again.
            if permission == .mailAutomation {
                do { try await system.prompt(permission) } catch {
                    Self.log.error("mailAutomation: re-check failed: \(String(describing: error), privacy: .public)")
                }
                let after = await state(of: permission)
                return after == .notDetermined ? .needsSystemSettings : after
            }
            Self.log.notice("\(permission.rawValue, privacy: .public): prompt already shown this session; not asking again")
            return .needsSystemSettings
        }
        guard await presentReason(permission) else { return current }
        prompted.insert(permission)
        do {
            try await system.prompt(permission)
        } catch {
            Self.log.error("\(permission.rawValue, privacy: .public): system prompt failed: \(String(describing: error), privacy: .public)")
        }
        let after = await state(of: permission)
        if after == .notDetermined {
            Self.log.error("\(permission.rawValue, privacy: .public): system prompt left the permission undetermined")
            return .needsSystemSettings
        }
        return after
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

/// The broker's EventKit store, kept for the broker's lifetime. EventKit
/// stores are not `Sendable`; the broker and the Calendar service only use it
/// to request access and read on EventKit's own terms.
public final class EventStoreBox: @unchecked Sendable {
    public let store: EKEventStore

    public init(_ store: EKEventStore = EKEventStore()) {
        self.store = store
    }
}

/// Permissions whose system prompt already ran this session.
final class PromptedPermissions: Sendable {
    private let set = Mutex<Set<AppPermission>>([])

    func contains(_ permission: AppPermission) -> Bool { set.withLock { $0.contains(permission) } }
    func insert(_ permission: AppPermission) { set.withLock { _ = $0.insert(permission) } }
}
