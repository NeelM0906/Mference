import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import SwiftUI
import Testing

@Suite struct OnboardingViewTests {
    @Test func deepOfferFollowsMemory() {
        #expect(OnboardingView(library: FixtureModelLibrary(memoryGB: 32, deep: .notInstalled), onFinish: {}).offersDeep)
        #expect(!OnboardingView(library: FixtureModelLibrary(memoryGB: 16, deep: .notInstalled), onFinish: {}).offersDeep)
    }

    /// Quick installs on first run without asking (L4), when the disk has room.
    @Test func quickInstallsOnFirstRun() {
        let library = FixtureModelLibrary(quick: .notInstalled, deep: .notInstalled)
        let view = OnboardingView(library: library, step: .models, onFinish: {})
        #expect(view.startQuickIfNeeded())
        #expect(library.downloads == [AppModel.quick.modelID])
        #expect(!view.startQuickIfNeeded(), "already downloading: not started twice")
    }

    @Test func quickWaitsWhenTheDiskIsTooFull() {
        let library = FixtureModelLibrary(freeDiskGB: 3, quick: .notInstalled, deep: .notInstalled)
        let view = OnboardingView(library: library, step: .models, onFinish: {})
        #expect(!view.startQuickIfNeeded(), "2.9 GB plus the margin does not fit in 3 GB")
        #expect(library.downloads.isEmpty)
    }

    /// Deep downloads only through "Download Deep", and never where L4 rules it out.
    @Test func deepDownloadsOnlyAfterApproval() throws {
        let library = FixtureModelLibrary(quick: .loaded, deep: .notInstalled)
        let view = OnboardingView(library: library, step: .models, onFinish: {})
        try Snapshot.render(view, name: "onboarding-models-check", framing: .centered(width: 1040))
        #expect(!library.downloads.contains(AppModel.deep.modelID), "showing the step never starts Deep")
        view.approveDeep()
        #expect(library.downloads == [AppModel.deep.modelID])

        let small = FixtureModelLibrary(memoryGB: 16, quick: .loaded, deep: .notInstalled)
        OnboardingView(library: small, step: .models, onFinish: {}).approveDeep()
        #expect(small.downloads.isEmpty, "no Deep on a 16 GB Mac, even through the button")
    }

    @Test func downloadingSummaryComesFromState() {
        let library = FixtureModelLibrary(quick: .downloading(TransferProgress(completedBytes: 1, totalBytes: 4)),
                                          deep: .notInstalled)
        #expect(OnboardingView.downloadingSummary(library.entries) == "Quick, 25%")
        #expect(OnboardingView.downloadingSummary(FixtureModelLibrary(deep: .ready).entries) == nil)
        #expect(OnboardingView.machineFacts(library.machine) == "Apple M1 Max, 32 GB memory, 582 GB free")
    }

    @Test func permissionsStepShowsEachReason() {
        for (permission, title) in OnboardingMetrics.permissionRows {
            #expect(!permission.reason.isEmpty && !title.isEmpty)
        }
    }

    @Test func onboardingShowsOnlyWhileIncomplete() throws {
        let suite = "OnboardingViewTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let model = ShellModel(defaults: defaults)
        #expect(!model.needsOnboarding(hasLibrary: false), "nothing to install without a library")
        #expect(model.needsOnboarding(hasLibrary: true))
        model.completeOnboarding()
        #expect(!model.needsOnboarding(hasLibrary: true))
        #expect(ShellModel(defaults: defaults).hasCompletedOnboarding, "completion persists")
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func snapshot(colorScheme: ColorScheme) throws {
        let library = FixtureModelLibrary(quick: .downloading(TransferProgress(completedBytes: 1_100_000_000,
                                                                               totalBytes: 2_900_000_000)),
                                          deep: .notInstalled)
        let view = OnboardingView(library: library, step: .models, onFinish: {})
        try Snapshot.render(view, name: "01-first-run", framing: .centered(width: 1040), colorScheme: colorScheme)
    }
}
