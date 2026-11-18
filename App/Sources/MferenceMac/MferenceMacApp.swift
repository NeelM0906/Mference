import MferenceAppCore
import MferenceAppUI
import MferenceDesign
import SwiftUI

struct MferenceMacApp: App {
    /// Owns the engine and the chat model, and runs the quit sequence.
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
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
                .environment(delegate.runtime.chat)
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
        }
    }
}
