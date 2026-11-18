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

    public var id: String { model.modelID }

    /// "Quick · Qwen3.5 4B".
    public var title: String {
        switch model {
        case .quick: "Quick · Qwen3.5 4B"
        case .deep: "Deep · Qwen3.8 27B"
        case .qwen36: "Qwen 3.6 · 35B-A3B"
        case .minicpm5: "MiniCPM5 · 2B"
        case .maple: "Maple · Preview"
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
                fit: MemoryFit.Verdict, licence: String? = nil) {
        self.model = model
        self.state = state
        self.sizeBytes = sizeBytes
        self.isDefault = isDefault
        self.fit = fit
        self.licence = licence
    }
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
    @ObservationIgnored private let settings: AppSettings
    @ObservationIgnored private var options: [ModelOption] = []
    @ObservationIgnored private var downloadStates: [AppModel: DownloadState] = [:]

    public init(engine: any EngineServicing, downloads: DownloadManager?, settings: AppSettings,
                memoryFit: MemoryFit = MemoryFit(), chipName: String? = MemoryFit.chipName) {
        self.engine = engine
        self.downloads = downloads
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
        // Two followers; both stop when the caller's task is cancelled.
        async let engineDone: Void = followEngine(engine.statusUpdates())
        async let downloadsDone: Void = followDownloads(await downloads?.updates())
        _ = await (engineDone, downloadsDone)
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
            for model in AppModel.allCases where downloads.source(for: model) != nil {
                downloadStates[model] = await downloads.state(of: model)
            }
        }
        await refreshOptions()
    }

    private func refreshOptions() async {
        options = await engine.models()
        rebuild()
    }

    private func rebuild() {
        rows = AppModel.allCases.map { model in
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
            return ModelLibraryRow(model: model, state: state, sizeBytes: size,
                                   isDefault: settings.defaultModel == model,
                                   fit: memoryFit.verdict(for: model, sizeBytes: size),
                                   licence: source?.licence)
        }
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

    public func pauseDownload(_ model: AppModel) {
        guard let downloads else { return }
        Task { await downloads.pause(model) }
    }

    public func discardDownload(_ model: AppModel) {
        guard let downloads else { return }
        Task { await downloads.discard(model) }
    }
}
