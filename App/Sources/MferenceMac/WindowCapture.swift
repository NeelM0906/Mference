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
            try await Task.sleep(for: .seconds(2))
            let content = try await SCShareableContent.currentProcess
            let pid = ProcessInfo.processInfo.processIdentifier
            guard let window = content.windows.first(where: {
                $0.owningApplication?.processID == pid && $0.windowLayer == 0 && $0.isOnScreen
            }) else {
                FileHandle.standardError.write(Data("capture: no on-screen window\n".utf8))
                NSApp.terminate(nil)
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
        NSApp.terminate(nil)
    }
}
