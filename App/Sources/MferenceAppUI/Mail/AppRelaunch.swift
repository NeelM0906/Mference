import AppKit
import Foundation

/// Quits Mference and opens it again, so macOS applies a Full Disk Access
/// grant (it takes effect only from the next launch).
public enum AppRelaunch {
    /// Starts a detached `open -n` on this app's bundle after a short delay,
    /// then quits. Outside a bundle (development runs) it only quits.
    @MainActor
    public static func quitAndReopen() {
        let bundle = Bundle.main.bundleURL
        if bundle.pathExtension == "app" {
            let process = Process()
            process.executableURL = URL(filePath: "/bin/sh")
            process.arguments = ["-c", "sleep 1; /usr/bin/open -n \"$0\"", bundle.path]
            try? process.run()
        }
        NSApp.terminate(nil)
    }
}
