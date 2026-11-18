import MferenceAppCore
import MferenceAppUI
import MferenceDesign
import SwiftUI

struct MferenceMacApp: App {
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
