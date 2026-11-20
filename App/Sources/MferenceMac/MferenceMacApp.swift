import AppKit
import MferenceAppCore
import MferenceAppUI
import MferenceDesign
import SwiftData
import SwiftUI

struct MferenceMacApp: App {
    /// Owns the app's services, chat and screen models, and runs the quit
    /// sequence.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @Environment(\.openWindow) private var openWindow
    @State private var shell = ShellModel(
        selection: ProcessInfo.processInfo.environment[WindowCapture.sectionVariable]
            .flatMap(AppSection.init(rawValue:)) ?? .chat)

    init() {
        // Before the first view renders, so MFFont resolves to Geist.
        MFFontRegistry.registerBundledFonts()
    }

    var body: some Scene {
        Window("Mference", id: "main") {
            RootView(model: shell)
                // ScreenModels, U4's Scan, Calendar and Mail models and the
                // model container, in one call (U2's helper).
                .mferenceEnvironment(services: delegate.runtime.services,
                                     screens: delegate.runtime.screenModels)
                .environment(delegate.runtime.services)
                .environment(delegate.runtime.chat)
                .environment(delegate.runtime.voice)
                .onAppear {
                    // Calendar's "Open Settings" goes to Settings > Connections.
                    let shell = shell
                    delegate.runtime.services.apple?.calendar.openSettings = {
                        shell.openSettings(.connections)
                    }
                }
                .task {
                    if let path = WindowCapture.requestedPath {
                        await WindowCapture.captureAndQuit(to: path)
                    }
                }
        }
        .defaultSize(MFLayout.defaultWindow)
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            ShellCommands(model: shell)
            VoiceCommands(voice: delegate.runtime.voice)
        }

        // While a meeting records: the time, Open, Pause or Resume, Stop (D7).
        MenuBarExtra(isInserted: Binding(get: { delegate.runtime.voice.liveMeeting != nil }, set: { _ in })) {
            if let live = delegate.runtime.voice.liveMeeting {
                RecordingMenuContent(live: live, controls: delegate.runtime.voice.recordingControls) {
                    openWindow(id: "main")
                    NSApp.activate()
                    shell.show(.notes)
                    delegate.runtime.voice.openLiveMeeting()
                }
            }
        } label: {
            if let live = delegate.runtime.voice.liveMeeting {
                RecordingMenuLabel(live: live)
            }
        }
    }
}
