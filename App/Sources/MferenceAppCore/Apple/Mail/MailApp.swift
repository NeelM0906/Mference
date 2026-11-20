#if os(macOS)
import AppKit
#else
import Foundation
#endif

/// Apple Mail itself: whether it runs (it is what downloads new mail) and
/// opening it.
public protocol MailAppControlling: Sendable {
    @MainActor func isRunning() -> Bool
    @MainActor func open()
}

/// The Mail app on this Mac.
public struct SystemMailApp: MailAppControlling {
    public static let bundleIdentifier = "com.apple.mail"

    public init() {}

    @MainActor public func isRunning() -> Bool {
        #if os(macOS)
        !NSRunningApplication.runningApplications(withBundleIdentifier: Self.bundleIdentifier).isEmpty
        #else
        false
        #endif
    }

    /// Opens Mail behind Mference, so the person stays where they are.
    @MainActor public func open() {
        #if os(macOS)
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: Self.bundleIdentifier) else { return }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false
        NSWorkspace.shared.openApplication(at: url, configuration: configuration)
        #endif
    }
}

/// Previews and tests: Mail is running or not, and opening it does nothing.
public struct FixedMailApp: MailAppControlling {
    public var running: Bool

    public init(running: Bool = true) {
        self.running = running
    }

    @MainActor public func isRunning() -> Bool { running }
    @MainActor public func open() {}
}
