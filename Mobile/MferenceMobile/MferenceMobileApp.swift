import MferenceMobileUI
import SwiftUI

/// Mference for iPhone and iPhone Duo. A thin entry point over
/// MferenceMobileUI; this first slice runs on sample data.
///
/// Launch arguments: `-MferenceShowFirstRun YES` shows first run again.
@main
struct MferenceMobileApp: App {
    /// Decided once per launch: the scene body can run more than once.
    private let showsFirstRun: Bool

    init() {
        MobileRootView.configureAppearance()
        let defaults = UserDefaults.standard
        let forced = defaults.bool(forKey: "MferenceShowFirstRun")
        let finished = defaults.bool(forKey: "MferenceDidFinishFirstRun")
        showsFirstRun = forced || !finished
        defaults.set(true, forKey: "MferenceDidFinishFirstRun")
    }

    var body: some Scene {
        WindowGroup {
            // Sample build: a finished first run is paired with the sample Mac;
            // a fresh one starts unpaired and pairs in step 3.
            MobileRootView(model: MobileAppModel(showFirstRun: showsFirstRun,
                                                 pairedMac: showsFirstRun ? nil : .sample))
        }
    }
}
