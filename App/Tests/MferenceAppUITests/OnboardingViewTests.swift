import Foundation
@testable import MferenceAppUI
import SwiftUI
import Testing

@Suite struct OnboardingViewTests {
    @Test func deepOfferFollowsMemory() {
        let big = OnboardingView(library: FixtureModelLibrary(memoryGB: 32, deep: .notInstalled), onFinish: {})
        #expect(big.offersDeep)
        let small = OnboardingView(library: FixtureModelLibrary(memoryGB: 16, deep: .notInstalled), onFinish: {})
        #expect(!small.offersDeep)
    }

    @Test func noPermissionIsListedWithoutItsReason() {
        for (permission, title) in OnboardingMetrics.permissionRows {
            #expect(!permission.reason.isEmpty && !title.isEmpty)
        }
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func snapshot(colorScheme: ColorScheme) throws {
        let library = FixtureModelLibrary(quick: .downloading(TransferProgress(completedBytes: 1_100_000_000,
                                                                               totalBytes: 2_900_000_000)),
                                          deep: .notInstalled)
        let view = OnboardingView(library: library, step: .models, onFinish: {})
        try Snapshot.render(view, name: "01-first-run", framing: .centered(width: 1040), colorScheme: colorScheme)
        // Showing the Models step never starts Deep: only Quick may download unasked.
        #expect(!library.downloads.contains(OnboardingView.deepID))
    }
}
