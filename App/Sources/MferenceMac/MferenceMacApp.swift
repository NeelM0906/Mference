import MferenceAppUI
import MferenceDesign
import SwiftUI

struct MferenceMacApp: App {
    init() {
        // Before the first view renders, so MFFont resolves to Geist.
        MFFontRegistry.registerBundledFonts()
    }

    var body: some Scene {
        WindowGroup("Mference") {
            RootView()
        }
    }
}
