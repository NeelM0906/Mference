import AppKit
import ChatTemplateCore
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct ChatTemplateApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @State private var store = ChatStore()

    var body: some Scene {
        Window("Chat", id: "main") {
            RootView(store: store)
        }
        .defaultSize(width: 1100, height: 720)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { store.newChat() }
                    .keyboardShortcut("n", modifiers: .command)
            }
            CommandMenu("Generation") {
                Button("Stop Generating") { store.stopGenerating() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!store.isGenerating)
            }
        }

        Settings {
            SettingsView()
        }
    }
}
