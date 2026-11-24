import Foundation
import MferenceAppCore
import Observation

/// `ModelLibraryModel` over U2's `ModelLibrary`, `DownloadManager`,
/// `MemoryFit` and `AppSettings`. Views see values only.
///
/// A download starts only through `download(_:)` or `resume(_:)`, which the
/// library view calls from its approval dialog; each one carries a
/// `DownloadApproval` for the pinned source. Offline mode and the memory gate
/// still refuse inside `DownloadManager`.
@Observable
public final class LiveModelLibraryModel: ModelLibraryModel {
    @ObservationIgnored private let library: ModelLibrary
    @ObservationIgnored private let settings: AppSettings
    @ObservationIgnored private let downloads: DownloadManager?
    @ObservationIgnored private let freeDisk: () -> Int64?

    public init(library: ModelLibrary, settings: AppSettings, downloads: DownloadManager?,
                freeDisk: (() -> Int64?)? = nil) {
        self.library = library
        self.settings = settings
        self.downloads = downloads
        // Free space where downloads go; unknown without a download manager
        // (nothing else is probed).
        let destination = downloads?.destination
        self.freeDisk = freeDisk ?? {
            destination.flatMap(DownloadManager.availableCapacity).map { Int64(clamping: $0) }
        }
    }

    public var machine: MachineSummary {
        MachineSummary(chipName: library.chipName ?? "",
                       physicalMemoryBytes: library.memoryFit.physicalMemory,
                       freeDiskBytes: freeDisk())
    }

    public var entries: [ModelEntry] {
        library.rows.map(Self.entry)
    }

    public var defaultModelID: String {
        get { settings.defaultModel.modelID }
        set {
            guard let model = AppModel(modelID: newValue) else { return }
            settings.defaultModel = model
        }
    }

    public var idleUnloadMinutes: Int {
        get { settings.idleUnloadMinutes }
        set { settings.idleUnloadMinutes = newValue }
    }

    /// Called by the approval dialog only.
    public func download(_ id: String) {
        guard let model = AppModel(modelID: id), let source = library.downloadSource(for: model) else { return }
        library.startDownload(DownloadApproval(confirming: source))
    }

    public func pause(_ id: String) {
        guard let model = AppModel(modelID: id) else { return }
        library.pauseDownload(model)
    }

    /// Continues a paused download the person approved earlier.
    public func resume(_ id: String) {
        download(id)
    }

    /// Retry after a load failure. Never starts or continues a download.
    public func retryLoad(_ id: String) {
        guard let model = AppModel(modelID: id) else { return }
        library.retryLoad(model)
    }

    /// Called by the approval dialog (the library's or the chat's offer) only.
    public func downloadImageUnderstanding(_ id: String) {
        guard let model = AppModel(modelID: id), let source = library.imageUnderstandingSource(for: model) else {
            return
        }
        library.startImageUnderstandingDownload(DownloadApproval(confirming: source))
    }

    public func pauseImageUnderstanding(_ id: String) {
        guard let model = AppModel(modelID: id) else { return }
        library.pauseImageUnderstanding(model)
    }

    public func removeImageUnderstanding(_ id: String) {
        guard let model = AppModel(modelID: id) else { return }
        library.removeImageUnderstanding(model)
    }

    public func cancelDownload(_ id: String) {
        guard let model = AppModel(modelID: id) else { return }
        library.discardDownload(model)
    }

    /// "Add a model file": validates the GGUF against the pinned Quick and
    /// Deep files, copies it into ~/llm-models/gguf (atomic rename; the
    /// original stays where it is) and refreshes the catalog.
    public func addModelFile(at url: URL) async throws {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        try await library.importModelFile(at: url)
    }

    // MARK: Mapping

    static func entry(_ row: ModelLibraryRow) -> ModelEntry {
        ModelEntry(id: row.model.modelID,
                   productName: row.model.name,
                   variantName: variantName(row.model),
                   summary: row.model.summary,
                   engineName: row.model.engineLabel,
                   formatDetail: formatDetail(row.model),
                   licence: row.licence,
                   sizeBytes: row.sizeBytes > 0 ? Int64(clamping: row.sizeBytes) : nil,
                   location: row.location?.path,
                   isRecommended: row.model.group == .recommended,
                   state: state(row),
                   fit: fit(row),
                   imageUnderstanding: row.imageUnderstanding.map(imageUnderstanding))
    }

    static func imageUnderstanding(_ addOn: ImageUnderstandingRow) -> ImageUnderstandingEntry {
        let row = ModelLibraryRow(model: .quick, state: addOn.state, sizeBytes: addOn.sizeBytes,
                                  isDefault: false, fit: .fits)
        return ImageUnderstandingEntry(state: state(row), sizeBytes: Int64(clamping: addOn.sizeBytes))
    }

    static func state(_ row: ModelLibraryRow) -> ModelEntryState {
        switch row.state {
        case .installed:
            switch row.availability {
            case .loaded?: .loaded
            case .loading?: .loading
            default: row.loadFailure.map(ModelEntryState.loadFailed) ?? .ready
            }
        case .downloading(_, let received, let total):
            .downloading(TransferProgress(completedBytes: Int64(clamping: received), totalBytes: Int64(clamping: total)))
        case .paused(_, let received, let total):
            .paused(TransferProgress(completedBytes: Int64(clamping: received), totalBytes: Int64(clamping: total)))
        case .verifying:
            .downloading(TransferProgress(completedBytes: Int64(clamping: row.sizeBytes),
                                          totalBytes: Int64(clamping: row.sizeBytes)))
        case .failed(let message):
            .downloadFailed(message)
        case .downloadable, .notInstalled, .notOffered:
            .notInstalled
        }
    }

    static func fit(_ row: ModelLibraryRow) -> MemoryFitVerdict {
        if case .notOffered(let reason) = row.state { return .tooLarge(reason) }
        switch row.fit {
        case .tooLarge: return .tooLarge("Needs more memory than this Mac has.")
        case .fits, .tight:
            // Deep is offered only above 16 GB and only after approval (L4).
            return row.model == .deep && row.state != .installed ? .offerWithApproval : .fits
        }
    }

    static func variantName(_ model: AppModel) -> String {
        switch model {
        case .quick: "Qwen3.5 4B"
        case .deep: "Qwen3.8 27B"
        case .inkling2Mini: "Mini"
        case .qwen36: "35B-A3B"
        case .minicpm5: "2B"
        case .maple, .apple: ""
        }
    }

    static func formatDetail(_ model: AppModel) -> String {
        switch model {
        case .quick: "Unsloth UD-Q4_K_XL"
        case .deep: "ISTA GSQ-RCO IQ3_S with MTP"
        case .inkling2Mini: "Mference install of Unsloth UD-Q4_K_XL"
        case .qwen36: "Mference install, experts stream from the SSD"
        case .minicpm5: "Mference install, 4-bit"
        case .maple: "Mference install, 2-bit"
        case .apple: "Built into macOS"
        }
    }
}
