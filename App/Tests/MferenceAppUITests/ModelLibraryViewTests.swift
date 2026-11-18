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

    /// L4 in one predicate: Deep only above 16 GB; Quick everywhere; never a too-large model.
    @Test(arguments: [UInt64(16), 32])
    func offeredModelsFollowL4(memoryGB: UInt64) throws {
        let library = FixtureModelLibrary(memoryGB: memoryGB, quick: .notInstalled, deep: .notInstalled)
        let deep = try #require(library.entries.first { $0.id == AppModel.deep.modelID })
        let quick = try #require(library.entries.first { $0.id == AppModel.quick.modelID })
        #expect(deep.isOffered(on: library.machine) == (memoryGB > 16))
        #expect(quick.isOffered(on: library.machine))
        var huge = quick
        huge.fit = .tooLarge("Needs more memory")
        #expect(!huge.isOffered(on: library.machine))
        let offered = ModelLibraryView.downloadable(library.entries + [huge], on: library.machine).map(\.id)
        #expect(offered.contains(AppModel.deep.modelID) == (memoryGB > 16))
        #expect(offered.filter { $0 == quick.id }.count == 1, "the too-large copy is left out")
    }

    /// L4 holds even if MemoryFit says Deep fits a 16 GB Mac.
    @Test func deepIsNotOfferedAt16GBEvenWhenItFits() throws {
        let library = FixtureModelLibrary(memoryGB: 16, deep: .notInstalled)
        var deep = try #require(library.entries.first { $0.id == AppModel.deep.modelID })
        deep.fit = .fits
        #expect(!deep.isOffered(on: library.machine))
    }

    /// A Deep download that failed was approved when it started: Retry
    /// restarts it without a second dialog, after checking the disk again.
    @Test func retryAfterAnApprovedDownloadNeedsNoSecondApproval() throws {
        let library = FixtureModelLibrary(deep: .downloadFailed("Network"))
        let deep = try #require(library.entries.first { $0.id == AppModel.deep.modelID })
        #expect(ModelLibraryView.retryDownload(deep, in: library))
        #expect(library.downloads == [AppModel.deep.modelID])

        let full = FixtureModelLibrary(freeDiskGB: 5, deep: .downloadFailed("Network"))
        let fullDeep = try #require(full.entries.first { $0.id == AppModel.deep.modelID })
        #expect(!ModelLibraryView.retryDownload(fullDeep, in: full), "12.1 GB does not fit in 5 GB")
        #expect(full.downloads.isEmpty)

        let ready = try #require(FixtureModelLibrary().entries.first { $0.id == AppModel.quick.modelID })
        #expect(!ModelLibraryView.retryDownload(ready, in: library), "only failed downloads retry")
    }

    @Test func diskRoomKeepsAMargin() {
        let machine = MachineSummary(chipName: "", physicalMemoryBytes: 1, freeDiskBytes: 4_000_000_000)
        #expect(machine.hasRoom(for: 1_900_000_000))
        #expect(!machine.hasRoom(for: 2_900_000_000))
        #expect(MachineSummary(chipName: "", physicalMemoryBytes: 1, freeDiskBytes: nil).hasRoom(for: 9_999_999_999))
    }

    @Test func loadFailureStaysInstalledAndDownloadFailureDoesNot() {
        #expect(ModelEntryState.loadFailed("Metal error").isInstalled)
        #expect(!ModelEntryState.downloadFailed("Network").isInstalled)
        #expect(ModelLibraryView.downloadable(
            FixtureModelLibrary(deep: .downloadFailed("Network")).entries,
            on: FixtureModelLibrary().machine).map(\.id) == [AppModel.deep.modelID])
    }

    struct MissingFile: FileRevealingError, LocalizedError {
        let fileToShow: URL?
        var errorDescription: String? { "That file is not a GGUF model." }
    }

    @Test func importFailureOffersShowInFinderOnlyForAFile() {
        let file = URL(fileURLWithPath: "/tmp/model.gguf")
        let withFile = ImportFailure(MissingFile(fileToShow: file))
        #expect(withFile.fileToShow == file)
        #expect(withFile.message == "That file is not a GGUF model.")
        #expect(ImportFailure(CocoaError(.fileReadNoPermission)).fileToShow == nil)
    }

    @Test func approvalIsTheDownloadPath() throws {
        let library = FixtureModelLibrary(deep: .notInstalled)
        let deep = try #require(library.entries.first { $0.id == AppModel.deep.modelID })
        try Snapshot.render(ModelLibraryView(library: library), name: "library-check")
        #expect(library.downloads.isEmpty, "rendering the library starts nothing")
        ModelLibraryView.approveDownload(deep, in: library)
        #expect(library.downloads == [AppModel.deep.modelID])
    }

    @Test func bannerSaysApprovedOnlyForApprovedDownloads() throws {
        let library = FixtureModelLibrary(quick: .downloading(TransferProgress(completedBytes: 0, totalBytes: 1)))
        let quick = try #require(library.entries.first { $0.id == AppModel.quick.modelID })
        let deep = try #require(library.entries.first { $0.id == AppModel.deep.modelID })
        #expect(!DownloadBanner(entry: quick, library: library).detail.contains("approved"))
        #expect(DownloadBanner(entry: deep, library: library).detail.contains("approved by you"))
    }

    @Test func statusBadges() {
        #expect(ModelStatusBadge.kind(for: .loaded) == .loaded)
        #expect(ModelStatusBadge.kind(for: .ready) == .ready)
        #expect(ModelStatusBadge.kind(for: .downloading(TransferProgress(completedBytes: 1, totalBytes: 2))).title == "Downloading")
        #expect(ModelStatusBadge.kind(for: .notInstalled).title == "Not installed")
        #expect(ModelStatusBadge.kind(for: .loadFailed("x")).title == "Didn't load")
        #expect(ModelStatusBadge.kind(for: .downloadFailed("x")).title == "Download failed")
    }

    @Test func idleText() {
        #expect(ModelLibraryView.idleText(10) == "10 minutes")
        #expect(ModelLibraryView.idleText(60) == "1 hour")
        #expect(ModelLibraryView.idleShort(10) == "10 min")
    }

    @Test func approvalNamesTheSize() throws {
        let deep = try #require(FixtureModelLibrary().entries.first { $0.id == AppModel.deep.modelID })
        #expect(ModelLibraryView.approvalMessage(for: deep).contains("12.1 GB"))
    }

    @Test(arguments: [ColorScheme.light, .dark])
    func snapshot(colorScheme: ColorScheme) throws {
        let screenModels = ScreenModels(library: FixtureModelLibrary(), memory: FixtureMemory(), settings: FixtureSettings())
        let view = SettingsView(library: screenModels.library, memory: screenModels.memory, settings: screenModels.settings,
                                showsLibrary: true)
        try Snapshot.render(ShellBoard(selection: .settings, screenModels: screenModels) { view },
                            name: "08b-model-library", canvas: CGSize(width: 1280, height: 1000),
                            colorScheme: colorScheme)
    }
}

/// The chrome bar over a screen, for boards whose screen needs inputs.
struct ShellBoard<Content: View>: View {
    let model: ShellModel
    let screenModels: ScreenModels
    let content: Content

    init(selection: AppSection, screenModels: ScreenModels, @ViewBuilder content: () -> Content) {
        model = ShellModel(selection: selection, defaults: nil)
        self.screenModels = screenModels
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 0) {
            ShellChromeBar(model: model)
            content.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .mfWindowBackground()
        .environment(screenModels)
    }
}
