import Foundation
import MferenceAppCore
import MferenceDesign
@testable import MferenceAppUI
import SwiftUI
import Testing

@Suite struct ModelLibraryViewTests {
    @Test func machineLineAndTransferText() {
        let machine = MachineSummary(chipName: "Apple M1 Max", physicalMemoryBytes: 32 * 1_073_741_824,
                                     freeDiskBytes: 582_000_000_000)
        #expect(ModelFormat.machineLine(machine) == "This Mac: Apple M1 Max, 32 GB memory, 582 GB free.")
        let unknown = MachineSummary(chipName: "", physicalMemoryBytes: 16 * 1_073_741_824, freeDiskBytes: nil)
        #expect(ModelFormat.machineLine(unknown) == "This Mac: 16 GB memory.")
        #expect(ModelFormat.transfer(TransferProgress(completedBytes: 4_600_000_000, totalBytes: 12_100_000_000))
                == "4.6 GB of 12.1 GB")
    }

    @Test func deepIsOfferedOnlyAbove16GB() {
        #expect(MachineSummary(chipName: "", physicalMemoryBytes: 16 * 1_073_741_824, freeDiskBytes: nil).canOfferDeep == false)
        #expect(MachineSummary(chipName: "", physicalMemoryBytes: 18 * 1_073_741_824, freeDiskBytes: nil).canOfferDeep)
    }

    @Test func statusBadges() {
        #expect(ModelStatusBadge.kind(for: .loaded) == .loaded)
        #expect(ModelStatusBadge.kind(for: .ready) == .ready)
        #expect(ModelStatusBadge.kind(for: .downloading(TransferProgress(completedBytes: 1, totalBytes: 2))).title == "Downloading")
        #expect(ModelStatusBadge.kind(for: .notInstalled).title == "Not installed")
    }

    @Test func idleText() {
        #expect(ModelLibraryView.idleText(10) == "10 minutes")
        #expect(ModelLibraryView.idleText(60) == "1 hour")
        #expect(ModelLibraryView.idleShort(10) == "10 min")
    }

    @Test func approvalNamesTheSize() throws {
        let deep = try #require(FixtureModelLibrary().entries.first { $0.id == OnboardingView.deepID })
        #expect(ModelLibraryView.approvalMessage(for: deep).contains("12.1 GB"))
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func snapshot(colorScheme: ColorScheme) throws {
        let library = FixtureModelLibrary()
        let services = AppServices(library: library)
        try Snapshot.render(ShellBoard(selection: .settings, services: services) { ModelLibraryView(library: library) },
                            name: "08b-model-library", canvas: CGSize(width: 1280, height: 1000),
                            colorScheme: colorScheme)
    }
}

/// The chrome bar over a screen, for boards whose screen needs inputs.
struct ShellBoard<Content: View>: View {
    let model: ShellModel
    let services: AppServices
    let content: Content

    init(selection: AppSection, services: AppServices, @ViewBuilder content: () -> Content) {
        model = ShellModel(selection: selection, defaults: nil)
        self.services = services
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 0) {
            ShellChromeBar(model: model)
            content.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .mfWindowBackground()
        .environment(services)
    }
}
