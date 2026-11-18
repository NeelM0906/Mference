import Foundation
import MferenceAppCore
@testable import MferenceAppUI
import Observation

/// Fixture models for the A3/A4 view tests and G-A3 snapshots. Values follow
/// the mockup boards (sizes as in 08b); nothing here ships in the app.
@Observable
final class FixtureModelLibrary: ModelLibraryModel {
    var machine: MachineSummary
    var entries: [ModelEntry]
    var defaultModelID = AppModel.quick.modelID
    var idleUnloadMinutes = 10
    private(set) var downloads: [String] = []
    private(set) var paused: [String] = []

    static let gigabyte: Int64 = 1_000_000_000

    init(memoryGB: UInt64 = 32, freeDiskGB: Int64 = 582, quick: ModelEntryState = .loaded, deep: ModelEntryState = .downloading(
        TransferProgress(completedBytes: 4_600_000_000, totalBytes: 12_100_000_000))) {
        machine = MachineSummary(chipName: "Apple M1 Max", physicalMemoryBytes: memoryGB * 1_073_741_824,
                                 freeDiskBytes: freeDiskGB * Self.gigabyte)
        let deepFit: MemoryFitVerdict = memoryGB > 16 ? .offerWithApproval : .tooLarge("Needs more than 16 GB")
        entries = [
            ModelEntry(id: AppModel.quick.modelID, productName: "Quick", variantName: "Qwen3.5 4B",
                       summary: "Default for everyday work. Unsloth UD-Q4_K_XL.", engineName: "llama.cpp",
                       formatDetail: "Unsloth UD-Q4_K_XL GGUF", licence: "Apache 2.0",
                       sizeBytes: 2_900_000_000, location: "~/llm-models/qwen3.5-4b", isRecommended: true,
                       state: quick, fit: .fits),
            ModelEntry(id: AppModel.deep.modelID, productName: "Deep", variantName: "Qwen3.8 27B",
                       summary: "Hard tasks and long documents. Slower than Quick; drafts several words at once.",
                       engineName: "llama.cpp",
                       formatDetail: "ISTA GSQ-RCO IQ3_S with MTP, near-lossless on AIME25, GPQA and LiveCodeBench",
                       sizeBytes: 12_100_000_000, isRecommended: true, state: deep, fit: deepFit),
            ModelEntry(id: AppModel.qwen36.modelID, productName: "Qwen 3.6", variantName: "35B-A3B",
                       summary: "Mixture-of-experts; experts stream from the SSD.", engineName: "Mference",
                       formatDetail: "gturbo int4", sizeBytes: 18 * Self.gigabyte, isRecommended: false,
                       state: .ready, fit: .fits),
            ModelEntry(id: AppModel.minicpm5.modelID, productName: "MiniCPM5", variantName: "2B",
                       summary: "Very small and quick.", engineName: "Mference", formatDetail: "gturbo int4 g64",
                       sizeBytes: 1_300_000_000, isRecommended: false, state: .ready, fit: .fits),
            ModelEntry(id: AppModel.maple.modelID, productName: "Maple", variantName: "",
                       summary: "Experimental.", engineName: "Mference", formatDetail: "2-bit MLX",
                       sizeBytes: 6_100_000_000, isRecommended: false, state: .ready, fit: .fits),
        ]
    }

    func download(_ id: String) {
        downloads.append(id)
        update(id) { $0.state = .downloading(TransferProgress(completedBytes: 0, totalBytes: $0.sizeBytes)) }
    }

    func pause(_ id: String) {
        paused.append(id)
        update(id) { if let progress = $0.state.transfer { $0.state = .paused(progress) } }
    }

    func resume(_ id: String) {
        update(id) { if let progress = $0.state.transfer { $0.state = .downloading(progress) } }
    }

    func cancelDownload(_ id: String) {
        update(id) { $0.state = .notInstalled }
    }

    private(set) var loadRetries: [String] = []
    func retryLoad(_ id: String) { loadRetries.append(id) }

    private(set) var addedFiles: [URL] = []
    func addModelFile(at url: URL) async throws { addedFiles.append(url) }

    private func update(_ id: String, _ change: (inout ModelEntry) -> Void) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        change(&entries[index])
    }
}

@Observable
final class FixtureMemory: MemoryListModel {
    var memories: [MemoryEntry] = [
        MemoryEntry(id: UUID(), text: "Prefers short, friendly replies at work", source: "Learned from 14 sent emails"),
        MemoryEntry(id: UUID(), text: "Dentist is Northside Dental", source: "From Mail, Sep 14"),
        MemoryEntry(id: UUID(), text: "Partner: Lena · planning Lisbon in October", source: "From chat, Sep 10"),
    ] + (1...9).map { MemoryEntry(id: UUID(), text: "Fixture memory \($0)", source: "From chat") }

    func add(_ text: String) { memories.insert(MemoryEntry(id: UUID(), text: text, source: "Added by you"), at: 0) }
    func update(_ id: UUID, text: String) {
        if let index = memories.firstIndex(where: { $0.id == id }) { memories[index].text = text }
    }
    func delete(_ id: UUID) { memories.removeAll { $0.id == id } }
}

@Observable
final class FixtureSettings: AppSettingsModel {
    var webSearchEnabled = false
    var offlineMode = false
    var blockRemoteMailContent = true
    var remindersSyncEnabled = false
    var useAppleModel = true
    var states: [ConnectionKind: ConnectionState] = [.appleMail: .connected, .appleCalendar: .notConnected]
    private(set) var connectRequests: [ConnectionKind] = []

    func connectionState(_ kind: ConnectionKind) -> ConnectionState { states[kind] ?? .notConnected }
    func connect(_ kind: ConnectionKind) { connectRequests.append(kind) }
}
