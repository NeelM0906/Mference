import Foundation
@testable import MferenceAppCore
@testable import MferenceAppUI
import Testing

/// The library's "Image understanding" line: its approval, its labels, and
/// the adapter's mapping from the core library row.
@Suite @MainActor struct ImageUnderstandingViewTests {
    static func library(quick: ModelEntryState = .ready, addOn: ModelEntryState,
                        freeDiskGB: Int64 = 582) -> (FixtureModelLibrary, ModelEntry) {
        let library = FixtureModelLibrary(freeDiskGB: freeDiskGB, quick: quick, deep: .notInstalled)
        var quickEntry = library.entries[0]
        quickEntry.imageUnderstanding = ImageUnderstandingEntry(state: addOn, sizeBytes: 672_423_616)
        return (library, quickEntry)
    }

    @Test func downloadingNeedsTheModelAndRoomAndStartsOnlyOnce() {
        let (library, entry) = Self.library(addOn: .notInstalled)
        #expect(ModelLibraryView.approveImageUnderstanding(entry, in: library))
        #expect(library.imageDownloads == [AppModel.quick.modelID])

        let (installed, installedEntry) = Self.library(addOn: .ready)
        #expect(!ModelLibraryView.approveImageUnderstanding(installedEntry, in: installed))
        let (noModel, noModelEntry) = Self.library(quick: .notInstalled, addOn: .notInstalled)
        #expect(!ModelLibraryView.approveImageUnderstanding(noModelEntry, in: noModel))
        let (full, fullEntry) = Self.library(addOn: .notInstalled, freeDiskGB: 1)
        #expect(!ModelLibraryView.approveImageUnderstanding(fullEntry, in: full))
        #expect(installed.imageDownloads.isEmpty && noModel.imageDownloads.isEmpty && full.imageDownloads.isEmpty)

        // A paused or failed one continues after approval too.
        let (paused, pausedEntry) = Self.library(addOn: .paused(TransferProgress(completedBytes: 1, totalBytes: 2)))
        #expect(ModelLibraryView.approveImageUnderstanding(pausedEntry, in: paused))
    }

    @Test func theLineSaysWhatItDoesAndWhatItCosts() {
        let (_, entry) = Self.library(addOn: .notInstalled)
        let message = ModelLibraryView.imageUnderstandingApprovalMessage(for: entry)
        #expect(message.hasPrefix("Quick’s image understanding downloads (672"))
        #expect(ImageUnderstandingLine.downloadTitle(.notInstalled) == "Download…")
        #expect(ImageUnderstandingLine.downloadTitle(.paused(TransferProgress(completedBytes: 1, totalBytes: 2)))
            == "Resume…")
        #expect(ImageUnderstandingLine.downloadTitle(.downloadFailed("x")) == "Retry…")
        #expect(ImageUnderstandingLine.badge(.ready) == .custom("Installed", .success))
    }

    @Test func theAdapterMapsTheCoreRow() {
        var row = ModelLibraryRow(model: .quick, state: .installed, sizeBytes: 2_912_109_728, isDefault: true,
                                  fit: .fits, availability: .ready)
        row.imageUnderstanding = ImageUnderstandingRow(
            state: .downloading(fraction: 0.5, receivedBytes: 336, totalBytes: 672), sizeBytes: 672)
        let entry = LiveModelLibraryModel.entry(row)
        #expect(entry.imageUnderstanding == ImageUnderstandingEntry(
            state: .downloading(TransferProgress(completedBytes: 336, totalBytes: 672)), sizeBytes: 672))
        row.imageUnderstanding = ImageUnderstandingRow(state: .installed, sizeBytes: 672)
        #expect(LiveModelLibraryModel.entry(row).imageUnderstanding?.state == .ready)
        row.imageUnderstanding = nil
        #expect(LiveModelLibraryModel.entry(row).imageUnderstanding == nil)
    }
}
