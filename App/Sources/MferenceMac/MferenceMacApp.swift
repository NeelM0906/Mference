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
    /// The view-facing models of U1's screens (model library, memory,
    /// settings). Filled from U2's adapters (MferenceAppUI/Adapters, built
    /// over U2's `AppServices.live()` composition root) once they land; until
    /// then Settings shows its empty states and first run waits for a model
    /// library.
    @State private var screenModels = ScreenModels()

    init() {
        // Before the first view renders, so MFFont resolves to Geist.
        MFFontRegistry.registerBundledFonts()
    }

    var body: some Scene {
        Window("Mference", id: "main") {
            RootView(model: shell)
                .environment(screenModels)
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
