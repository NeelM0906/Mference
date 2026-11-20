import Foundation

// SHARED CONTRACT (decision L6): no permission is asked up front. A feature
// asks the broker the first time it needs access; the broker shows the
// one-line reason, then the system prompt. U2 owns the implementation; U4's
// Apple services call it.

/// A permission the app may ask for, each with its one-line reason.
public enum AppPermission: String, Sendable, Hashable, Codable, CaseIterable {
    case calendar
    case reminders
    case camera
    case fullDiskAccess
    case mailAutomation
    case microphone
    /// Find reads the Photos library (Settings › Find › Photos library).
    case photos

    /// The one line shown before the system prompt.
    public var reason: String {
        switch self {
        case .calendar: "To show your events and add the ones you approve."
        case .reminders: "To sync your to-dos with Reminders, only because you turned it on."
        case .camera: "To scan a receipt or document with the camera."
        case .fullDiskAccess: "To read Apple Mail on this Mac. Mail never leaves this Mac."
        case .mailAutomation: "To send a reply through Mail when you click Send."
        case .microphone: "To take dictation on this Mac."
        case .photos: "To find photos and videos in your Photos library by what is in them, on this Mac."
        }
    }
}

public enum PermissionState: String, Sendable, Hashable, Codable {
    /// Never asked.
    case notDetermined
    case granted
    case denied
    /// Must be changed in System Settings (Full Disk Access, or after a denial).
    case needsSystemSettings
}

public protocol PermissionBroker: Sendable {
    /// The current state, without prompting.
    func state(of permission: AppPermission) async -> PermissionState
    /// Asks if needed (reason first, then the system prompt) and returns the
    /// resulting state. Never called at launch.
    func request(_ permission: AppPermission) async -> PermissionState
    /// Opens the matching pane of System Settings.
    func openSystemSettings(for permission: AppPermission) async
}
