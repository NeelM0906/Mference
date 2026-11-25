import Foundation
import Observation

/// One row of the model library and of Settings > Models (mockups 08, 08b).
public struct ModelLibraryRow: Sendable, Hashable, Identifiable {
    public enum State: Sendable, Hashable {
        /// On this Mac and runnable.
        case installed
        /// Can be downloaded after the person approves.
        case downloadable
        case downloading(fraction: Double, receivedBytes: UInt64, totalBytes: UInt64)
        case paused(fraction: Double, receivedBytes: UInt64, totalBytes: UInt64)
        case verifying
        case failed(message: String)
        /// Not on this Mac and not downloadable from the app (native installs).
        case notInstalled(reason: String)
        /// Not offered on this Mac (Deep on 16 GB or less, L4).
        case notOffered(reason: String)
    }

    public let model: AppModel
    public var state: State
    /// On disk when installed, the download's size otherwise; 0 when unknown.
    public var sizeBytes: UInt64
    public var isDefault: Bool
    public var fit: MemoryFit.Verdict
    /// Licence of a downloadable model ("Apache 2.0"); nil when unknown.
    public var licence: String?
    /// The engine's state for an installed model (loaded, ready, loading,
    /// unloading); nil when not installed.
    public var availability: ModelAvailability?
    /// Install location; nil when not installed.
    public var location: URL?
    /// The engine's message when the last load of this installed model
    /// failed; cleared by the next successful load.
    public var loadFailure: String?
    /// Quick's and Deep's optional image understanding (the vision
    /// projector): nil for the other models, and while the model itself is
    /// not offered on this Mac.
    public var imageUnderstanding: ImageUnderstandingRow?

    public var id: String { model.modelID }

    /// "Quick · Qwen3.5 4B".
    public var title: String {
        switch model {
        case .quick: "Quick · Qwen3.5 4B"
        case .deep: "Deep · Qwen3.8 27B"
        case .inkling2Mini: "Inkling 2 · Mini"
        case .qwen36: "Qwen 3.6 · 35B-A3B"
        case .minicpm5: "MiniCPM5 · 2B"
        case .minicpm5_1b: "MiniCPM5 · 1B"
        case .maple: "Maple · Preview"
        case .apple: "Apple Intelligence"
        }
    }

    /// The badge ("Installed · default", "Downloading · 62%", ...).
    public var badge: String {
        switch state {
        case .installed: isDefault ? "Installed · default" : "Installed"
        case .downloadable: "Not installed"
        case .downloading(let fraction, _, _): "Downloading · \(Int((fraction * 100).rounded(.down)))%"
        case .paused(let fraction, _, _): "Paused · \(Int((fraction * 100).rounded(.down)))%"
        case .verifying: "Checking"
        case .failed: "Download failed"
        case .notInstalled: "Not installed"
        case .notOffered: "Needs more memory"
        }
    }

    public init(model: AppModel, state: State, sizeBytes: UInt64, isDefault: Bool,
                fit: MemoryFit.Verdict, licence: String? = nil,
                availability: ModelAvailability? = nil, location: URL? = nil) {
        self.model = model
        self.state = state
        self.sizeBytes = sizeBytes
        self.isDefault = isDefault
        self.fit = fit
        self.licence = licence
        self.availability = availability
        self.location = location
    }
}

/// The "Image understanding" line under Quick and Deep: an optional add-on
/// that lets the model see pictures. Until it is installed, pictures reach
/// the model as the text read from them.
public struct ImageUnderstandingRow: Sendable, Hashable {
    /// `.installed`, `.downloadable`, a download's progress, or `.failed`.
    public var state: ModelLibraryRow.State
    /// On disk when installed, the download's size otherwise.
    public var sizeBytes: UInt64
    /// The installed projector the engine paired, when installed.
    public var location: URL?
    /// The library may delete it: it is in the Mference models folder. One
    /// the person put elsewhere stays theirs to remove.
    public var isRemovable: Bool

    public init(state: ModelLibraryRow.State, sizeBytes: UInt64, location: URL? = nil, isRemovable: Bool = true) {
        self.state = state
        self.sizeBytes = sizeBytes
        self.location = location
        self.isRemovable = isRemovable
    }

    public static let title = "Image understanding"

    public var isInstalled: Bool { state == .installed }
}

/// The model library: installed or missing from the engine's catalog, sizes,
/// memory fit, and downloads (spec 4.3, A3). Downloads need the person's
/// approval; this model never starts one on its own.
@MainActor
@Observable
public final class ModelLibrary {
    public private(set) var rows: [ModelLibraryRow] = []
    public let memoryFit: MemoryFit
    /// "Apple M1 Max", when known.
    public let chipName: String?

    @ObservationIgnored private let engine: any EngineServicing
    @ObservationIgnored private let downloads: DownloadManager?
    @ObservationIgnored private let imageDownloads: DownloadManager?
    @ObservationIgnored private let settings: AppSettings
    @ObservationIgnored private var options: [ModelOption] = []
    @ObservationIgnored private var engineStatus = EngineServiceStatus.empty
    @ObservationIgnored private var downloadStates: [AppModel: DownloadState] = [:]
    @ObservationIgnored private var imageDownloadStates: [AppModel: DownloadState] = [:]

    public init(engine: any EngineServicing, downloads: DownloadManager?, imageDownloads: DownloadManager? = nil,
                settings: AppSettings,
                memoryFit: MemoryFit = MemoryFit(), chipName: String? = MemoryFit.chipName) {
        self.engine = engine
        self.downloads = downloads
        self.imageDownloads = imageDownloads
        self.settings = settings
        self.memoryFit = memoryFit
        self.chipName = chipName
    }

    public func row(for model: AppModel) -> ModelLibraryRow? { rows.first { $0.model == model } }

    /// "This Mac: Apple M1 Max, 32 GB memory."
    public var machineSummary: String {
        if let chipName { return "This Mac: \(chipName), \(memoryFit.memoryText) memory." }
        return "This Mac: \(memoryFit.memoryText) memory."
    }

    /// Follows the engine and the downloads; call from the view's `.task`.
    public func run() async {
        await refresh()
        let engine = self.engine
        let downloads = self.downloads
        let imageDownloads = self.imageDownloads
        // Three followers; all stop when the caller's task is cancelled.
        async let engineDone: Void = followEngine(engine.statusUpdates())
        async let downloadsDone: Void = followDownloads(await downloads?.updates())
        async let imagesDone: Void = followImageDownloads(await imageDownloads?.updates())
        _ = await (engineDone, downloadsDone, imagesDone)
    }

    private func followImageDownloads(_ updates: AsyncStream<[AppModel: DownloadState]>?) async {
        guard let updates else { return }
        for await states in updates {
            let changedInstall = states.contains { model, state in
                let installed: Bool = if case .installed = state { true } else { false }
                return installed != (row(for: model)?.imageUnderstanding?.isInstalled ?? false)
            }
            imageDownloadStates = states
            // Installed or removed: the engine's catalog pairs it now.
            if changedInstall { await refreshOptions() } else { rebuild() }
        }
    }

    private func followEngine(_ updates: AsyncStream<EngineServiceStatus>) async {
        for await _ in updates { await refreshOptions() }
    }

    private func followDownloads(_ updates: AsyncStream<[AppModel: DownloadState]>?) async {
        guard let updates else { return }
        for await states in updates {
            downloadStates = states
            // A finished download: the engine's catalog has it now.
            let newlyInstalled = states.contains { model, state in
                if case .installed = state { return row(for: model)?.state != .installed }
                return false
            }
            if newlyInstalled { await refreshOptions() } else { rebuild() }
        }
    }

    public func refresh() async {
        if let downloads {
            for model in AppModel.engineModels where downloads.source(for: model) != nil {
                downloadStates[model] = await downloads.state(of: model)
            }
        }
        if let imageDownloads {
            for model in AppModel.engineModels where imageDownloads.source(for: model) != nil {
                imageDownloadStates[model] = await imageDownloads.state(of: model)
            }
        }
        await refreshOptions()
    }

    private func refreshOptions() async {
        options = await engine.models()
        engineStatus = engine.status
        rebuild()
    }

    private func rebuild() {
        rows = AppModel.engineModels.map { model in
            let option = options.first { $0.model == model }
            let source = downloads?.source(for: model)
            let isInstalled: Bool = if case .notInstalled? = option?.availability { false } else { option != nil }
            let size = isInstalled ? (option?.sizeBytes ?? 0) : (source?.sizeBytes ?? 0)
            let state: ModelLibraryRow.State
            if isInstalled {
                state = .installed
            } else if let reason = memoryFit.notOfferedReason(model) {
                state = .notOffered(reason: reason)
            } else if source != nil {
                state = Self.state(downloadStates[model] ?? .notStarted)
            } else if case .notInstalled(let reason)? = option?.availability {
                state = .notInstalled(reason: reason)
            } else {
                state = .notInstalled(reason: "Not installed on this Mac.")
            }
            let imageRow = imageUnderstandingRow(model, option: option)
            // The projector stays resident beside the model once it is used.
            let fitSize = size + ((imageRow?.isInstalled ?? false) ? (imageRow?.sizeBytes ?? 0) : 0)
            var row = ModelLibraryRow(model: model, state: state, sizeBytes: size,
                                   isDefault: settings.defaultModel == model,
                                   fit: memoryFit.verdict(for: model, sizeBytes: fitSize),
                                   licence: source?.licence,
                                   availability: isInstalled ? option?.availability : nil,
                                   location: isInstalled ? option?.location : nil)
            if isInstalled, engineStatus.lastLoadFailureModel == model, option?.availability == .ready {
                row.loadFailure = engineStatus.lastLoadFailure
            }
            row.imageUnderstanding = imageRow
            return row
        }
    }

    /// Installed when the engine paired a projector with the model (or the
    /// download just finished); otherwise the add-on's download state.
    private func imageUnderstandingRow(_ model: AppModel, option: ModelOption?) -> ImageUnderstandingRow? {
        guard model.supportsImageUnderstanding, memoryFit.isOffered(model),
              let source = imageDownloads?.source(for: model) else { return nil }
        if option?.seesImages == true {
            let location = option?.imageUnderstandingLocation
            return ImageUnderstandingRow(state: .installed,
                                         sizeBytes: option?.imageUnderstandingBytes ?? source.sizeBytes,
                                         location: location,
                                         isRemovable: location.map { imageDownloads?.isInDestination($0) ?? false } ?? true)
        }
        let download = imageDownloadStates[model] ?? .notStarted
        if case .installed = download {
            return ImageUnderstandingRow(state: .installed, sizeBytes: source.sizeBytes)
        }
        return ImageUnderstandingRow(state: Self.state(download), sizeBytes: source.sizeBytes)
    }

    static func state(_ download: DownloadState) -> ModelLibraryRow.State {
        switch download {
        case .notStarted, .installed: .downloadable
        case .downloading(let received, let total):
            .downloading(fraction: download.fraction ?? 0, receivedBytes: received, totalBytes: total)
        case .paused(let received, let total):
            .paused(fraction: download.fraction ?? 0, receivedBytes: received, totalBytes: total)
        case .verifying: .verifying
        case .failed(let message): .failed(message: message)
        }
    }

    // MARK: Actions

    /// The approval dialog's facts for `model`: nil when it cannot download.
    public func downloadSource(for model: AppModel) -> DownloadSource? {
        guard memoryFit.isOffered(model) else { return nil }
        return downloads?.source(for: model)
    }

    /// Starts or continues an approved download; errors land in the row.
    public func startDownload(_ approval: DownloadApproval) {
        guard let downloads else { return }
        Task { try? await downloads.download(approval) }
    }

    /// Adds a model file the person picked: validated against the pinned
    /// product files, copied into the downloads folder, then the engine's
    /// catalog is refreshed so the row shows it installed. The original is
    /// never moved or deleted.
    @discardableResult
    public func importModelFile(at url: URL) async throws -> AppModel {
        // The download manager's folder: ~/llm-models/gguf in the app, a
        // temporary directory in tests and previews.
        guard let downloads else { throw ModelImportError.file("model files cannot be added here") }
        let destination = downloads.destination
        let sources = AppModel.engineModels.compactMap { downloads.source(for: $0) }
        let model = try await ModelFileImporter(destination: destination, sources: sources).importFile(at: url)
        await engine.refreshCatalog()
        await refreshOptions()
        return model
    }

    /// Retry after a failed load: loads an installed model again (the Deep
    /// gate still applies; a local load needs no network, so offline mode
    /// does not block it). It never starts or continues a download: those
    /// go only through the approval paths (`startDownload`), so this is not a
    /// second way to fetch Deep (L4). For anything not installed it does
    /// nothing.
    public func retryLoad(_ model: AppModel) {
        guard memoryFit.isOffered(model), row(for: model)?.state == .installed else { return }
        let engine = self.engine
        Task {
            try? await engine.load(model)
            await refreshOptions()
        }
    }

    // MARK: Image understanding

    /// The approval dialog's facts for `model`'s image understanding: nil
    /// when it has none on this Mac.
    public func imageUnderstandingSource(for model: AppModel) -> DownloadSource? {
        guard model.supportsImageUnderstanding, memoryFit.isOffered(model) else { return nil }
        return imageDownloads?.source(for: model)
    }

    /// Starts or continues an approved image-understanding download.
    public func startImageUnderstandingDownload(_ approval: DownloadApproval) {
        guard let imageDownloads else { return }
        Task { try? await imageDownloads.download(approval) }
    }

    public func pauseImageUnderstanding(_ model: AppModel) {
        guard let imageDownloads else { return }
        Task { await imageDownloads.pause(model) }
    }

    /// Removes the add-on the engine paired (or the partial download).
    /// Pictures then reach the model as text again. A model already loaded
    /// keeps seeing until it is unloaded. A projector outside the Mference
    /// models folder is never deleted (`isRemovable` is false).
    public func removeImageUnderstanding(_ model: AppModel) {
        guard let imageDownloads else { return }
        let addOn = row(for: model)?.imageUnderstanding
        guard addOn?.isRemovable ?? true else { return }
        let paired = addOn?.location
        Task { try? await imageDownloads.remove(model, file: paired) }
    }

    public func pauseDownload(_ model: AppModel) {
        guard let downloads else { return }
        Task { await downloads.pause(model) }
    }

    public func discardDownload(_ model: AppModel) {
        guard let downloads else { return }
        Task { await downloads.discard(model) }
    }
}
