import MferenceAppCore
import MferenceAppUI
import MferenceDesign
import SwiftUI

struct MferenceMacApp: App {
    @State private var shell = ShellModel(
        selection: ProcessInfo.processInfo.environment[WindowCapture.sectionVariable]
            .flatMap(AppSection.init(rawValue:)) ?? .chat)
    /// Filled with U2's live model library and memory store, and U4's
    /// connections, as those land; until then Settings shows empty states
    /// and first run waits for a model library.
    @State private var services = AppServices()

    init() {
        // Before the first view renders, so MFFont resolves to Geist.
        MFFontRegistry.registerBundledFonts()
    }

    var body: some Scene {
        Window("Mference", id: "main") {
            RootView(model: shell)
                .environment(services)
                .task {
                    if let path = WindowCapture.requestedPath {
                        await WindowCapture.captureAndQuit(to: path)
                    }
                }
        }
        .defaultSize(width: 1200, height: 748)
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            ShellCommands(model: shell)
        }
    }
}
