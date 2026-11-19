import AppKit
import Foundation
import ScreenCaptureKit

/// Diagnostic only: with `MFERENCE_CAPTURE_WINDOW=<path.png>` set, the app
/// captures its own main window a moment after launch, writes the PNG and
/// quits. Used for checkpoint evidence (C-A0) without granting Screen
/// Recording: `SCShareableContent.currentProcess` covers only this app's own
/// windows and needs no permission. `MFERENCE_CAPTURE_SECTION=<AppSection raw
/// value>` picks the section first. Not a supported control.
enum WindowCapture {
    static let pathVariable = "MFERENCE_CAPTURE_WINDOW"
    static let sectionVariable = "MFERENCE_CAPTURE_SECTION"

    static var requestedPath: String? {
        ProcessInfo.processInfo.environment[pathVariable].flatMap { $0.isEmpty ? nil : $0 }
    }

    static func captureAndQuit(to path: String) async {
        do {
            // Settle, then wait (up to about 10 s) for the window to be on
            // screen: a cold launch can take longer than a warm one.
            // No activation: a background check must not take the keyboard
            // from whatever the person is typing in.
            try await Task.sleep(for: .seconds(2))
            var window: SCWindow?
            for _ in 0..<16 where window == nil {
                window = try await ownWindow()
                if window == nil { try await Task.sleep(for: .milliseconds(500)) }
            }
            guard let window else {
                FileHandle.standardError.write(Data("capture: no window of this app\n".utf8))
                quit()
                return
            }
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let configuration = SCStreamConfiguration()
            let scale = CGFloat(filter.pointPixelScale)
            configuration.width = Int(window.frame.width * scale)
            configuration.height = Int(window.frame.height * scale)
            configuration.showsCursor = false
            let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: configuration)
            let rep = NSBitmapImageRep(cgImage: image)
            if let png = rep.representation(using: .png, properties: [:]) {
                try png.write(to: URL(fileURLWithPath: path))
                print("capture: wrote \(path) (\(image.width)x\(image.height))")
            }
        } catch {
            FileHandle.standardError.write(Data("capture: \(error)\n".utf8))
        }
        quit()
    }

    private static func ownWindow() async throws -> SCWindow? {
        let content = try await SCShareableContent.currentProcess
        let pid = ProcessInfo.processInfo.processIdentifier
        // The app's own main window (layer 0); a desktop-independent capture
        // works even when the window is on another Space or not reported on
        // screen. Prefer an on-screen one, then the largest.
        let own = content.windows.filter { $0.owningApplication?.processID == pid && $0.windowLayer == 0 }
        return own.first(where: \.isOnScreen)
            ?? own.max { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }
    }

    /// Quits from the run loop, not from inside this main-actor job: the
    /// app delegate's shutdown (`.terminateLater`) replies from another
    /// main-actor job, which could never run while this one sat in
    /// `terminate`.
    private static func quit() {
        NSApp.perform(#selector(NSApplication.terminate(_:)), with: nil, afterDelay: 0)
    }
}
