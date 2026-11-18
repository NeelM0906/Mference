import MferenceAppUI
import MferenceDesign
import SwiftUI

struct MferenceMacApp: App {
    @State private var shell = ShellModel()

    init() {
        // Before the first view renders, so MFFont resolves to Geist.
        MFFontRegistry.registerBundledFonts()
    }

    var body: some Scene {
        Window("Mference", id: "main") {
            RootView(model: shell)
        }
        .defaultSize(width: 1200, height: 748)
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            ShellCommands(model: shell)
        }
    }
}
