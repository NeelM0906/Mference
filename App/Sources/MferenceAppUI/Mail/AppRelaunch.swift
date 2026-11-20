import AppKit
import Foundation

/// Quits Mference and opens it again, so macOS applies a Full Disk Access
/// grant (it takes effect only from the next launch).
public enum AppRelaunch {
    /// Starts a detached helper that waits for this process to exit (an
    /// orderly quit can take seconds while the engine and web search stop),
    /// then opens the bundle again; then quits. If this process is still
    /// running after a minute, the helper gives up rather than start a
    /// second copy. Outside a bundle (development runs) it only quits.
    @MainActor
    public static func quitAndReopen() {
        let bundle = Bundle.main.bundleURL
        if bundle.pathExtension == "app" {
            let process = Process()
            process.executableURL = URL(filePath: "/bin/sh")
            process.arguments = ["-c", reopenScript, bundle.path,
                                 String(ProcessInfo.processInfo.processIdentifier)]
            try? process.run()
        }
        NSApp.terminate(nil)
    }

    /// `$0` is the bundle path, `$1` the process to wait for.
    static let reopenScript = """
        i=0
        while /bin/kill -0 "$1" 2>/dev/null; do
          i=$((i + 1)); [ "$i" -gt 300 ] && exit 0
          /bin/sleep 0.2
        done
        /usr/bin/open -n "$0"
        """
}
