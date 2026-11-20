import MferenceMobileSamples
import MferenceMobileUI
import SwiftUI

/// Mference for iPhone and iPhone Duo. A thin entry point over
/// MferenceMobileUI; this first slice runs on sample data.
///
/// Temporary: the sample screen models come from MferenceMobileSamples until
/// each owner's live model lands (Mobile/CONTRACTS.md); then each is swapped
/// here and the link to MferenceMobileSamples goes.
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
            MobileRootView(model: MobileAppModel(calendar: SampleCalendarModel(),
                                                 scan: SampleScanModel(),
                                                 showFirstRun: showsFirstRun,
                                                 pairedMac: showsFirstRun ? nil : .sample))
        }
    }
}
