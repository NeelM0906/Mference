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
                // ⌘W closes the window: a dictation stops (the microphone is
                // released, its last phrase lands); a meeting keeps going (D7).
                .onDisappear {
                    let voice = delegate.runtime.voice
                    Task { await voice.stopDictation() }
                }
                .onAppear {
                    // Leaving Notes mid-dictation: the last phrase lands first.
                    let voice = delegate.runtime.voice
                    shell.finishBeforeLeaving = { section in
                        VoiceModels.finishBeforeLeaving(section, voice: voice)
                    }
                    // Calendar's "Open Settings" goes to Settings > Connections.
                    let shell = shell
                    delegate.runtime.services.apple?.calendar.openSettings = {
                        shell.openSettings(.connections)
                    }
                    // Find's quick panel on ⌥Space, from any app.
                    if let find = delegate.runtime.services.find {
                        let chat = delegate.runtime.chat
                        FindPanelController.shared.install(
                            find: find,
                            askAbout: { url in
                                chat.startChat(about: url)
                                shell.show(.chat)
                            },
                            openFind: { shell.show(.find) })
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
            VoiceCommands(voice: delegate.runtime.voice, shell: shell)
        }

        // While a meeting records: the time, Open, Pause or Resume, Stop (D7).
        MenuBarExtra(isInserted: Binding(get: { delegate.runtime.voice.liveMeeting != nil }, set: { _ in })) {
            if let live = delegate.runtime.voice.liveMeeting {
                let voice = delegate.runtime.voice
                RecordingMenuContent(live: live, controls: voice.recordingControls,
                                     callAudioHint: voice.callAudioHint.map { message in
                                         .init(message: message, openSettings: { voice.openCallAudioSettings() },
                                               dismiss: { voice.dismissCallAudioHint() })
                                     }) {
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

        // The menu-bar extra: engine state at a glance and quick actions,
        // over the same AppServices as the window.
        MenuBarExtra {
            MenuBarContent(runtime: delegate.runtime, shell: shell)
        } label: {
            MenuBarLabel(model: delegate.runtime.menuBar)
        }
        .menuBarExtraStyle(.menu)
    }
}

/// The menu's content, with the actions pointed at the window, the chat and
/// the app's quit sequence.
private struct MenuBarContent: View {
    let runtime: AppRuntime
    let shell: ShellModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        MenuBarMenu(model: runtime.menuBar, library: runtime.screenModels.library,
                    settings: runtime.services.settings,
                    actions: MenuBarActions(
                        openWindow: showWindow,
                        newChat: {
                            runtime.chat.newChat()
                            shell.show(.chat)
                            showWindow()
                        },
                        selectModel: { runtime.chat.selectModel($0) },
                        openSettings: {
                            shell.show(.settings)
                            showWindow()
                        },
                        // Runs applicationShouldTerminate, so the quit
                        // sequence (web, Find, then engine) runs as from ⌘Q.
                        quit: { NSApplication.shared.terminate(nil) }))
    }

    private func showWindow() {
        openWindow(id: "main")
        NSApplication.shared.activate()
    }
}
