import Foundation
import MferenceAppCore

/// What the menu-bar extra shows, worked out from the engine's status, the
/// engine's model list and the model library's downloads. A value, so every
/// state is testable without a menu.
nonisolated public struct MenuBarSummary: Sendable, Hashable {
    /// The top row: "Quick 4B · Loaded", "Loading Deep 27B", "No model in memory".
    public var headline: String
    /// The lines under it: what is running, memory, idle unload, a failure.
    public var details: [String]
    /// The menu-bar icon (SF Symbol) for this state.
    public var symbol: String
    /// The model whose failed load the menu offers to retry.
    public var retryModel: AppModel?
    /// Why Unload Model is off, or nil when it is on.
    public var unloadDisabledReason: String?
    /// Why switching models is off, or nil when it is on.
    public var switchDisabledReason: String?
    /// The Model submenu, in picker order and groups.
    public var modelRows: [ModelRow]
    /// One line per download that is running, paused or failed.
    public var downloads: [String]

    /// One row of the Model submenu.
    public struct ModelRow: Sendable, Hashable, Identifiable {
        public var model: AppModel
        public var group: ModelGroup
        /// "Quick 4B"
        public var title: String
        /// "Loaded · 4.5 GB", "Fits · 8.0 GB", "Tight · 14.8 GB", "Not installed".
        public var detail: String
        /// Resident now (checkmark).
        public var isResident: Bool
        /// Can be picked now.
        public var isEnabled: Bool

        public var id: String { model.modelID }
    }

    public init(status: EngineServiceStatus, options: [ModelOption], entries: [ModelEntry],
                idleUnloadMinutes: Int, physicalMemory: UInt64) {
        let fit = MemoryFit(physicalMemory: physicalMemory)
        let resident = status.residentModel
        let target = status.targetModel

        func memory(_ model: AppModel) -> String? {
            guard let option = options.first(where: { $0.model == model }), option.sizeBytes > 0 else { return nil }
            return Self.gigabytes(fit.estimatedResidentBytes(model, sizeBytes: option.sizeBytes))
        }

        var details: [String] = []
        switch status.phase {
        case .ready:
            let name = resident?.shortName ?? "A model"
            headline = "\(name) · Loaded"
            symbol = status.isGenerating ? "ellipsis.bubble.fill" : "cpu.fill"
            details.append(Self.activity(status))
            var memoryLine = resident.flatMap(memory).map { "About \($0) in memory" } ?? "In memory"
            memoryLine += idleUnloadMinutes > 0
                ? " · unloads after \(idleUnloadMinutes) min idle"
                : " · stays loaded"
            details.append(memoryLine)
        case .loading:
            headline = "Loading \(target?.shortName ?? "a model")"
            symbol = "hourglass"
            if let target, let memory = memory(target) { details.append("About \(memory) in memory when loaded") }
            if status.queuedRequests > 0 { details.append(Self.waiting(status.queuedRequests) + ", then answers") }
        case .unloading:
            headline = "Unloading \(target?.shortName ?? "the model")"
            symbol = "hourglass"
            details.append("Its memory is being released")
        case .empty:
            headline = "No model in memory"
            symbol = "cpu"
            details.append("The model loads with your next message")
        }

        retryModel = nil
        if let failure = status.lastLoadFailure, status.phase != .loading {
            let model = status.lastLoadFailureModel
            details.append("\(model?.shortName ?? "The model") didn't load: \(failure)")
            symbol = "exclamationmark.triangle"
            retryModel = model
        }
        self.details = details

        switch status.phase {
        case .loading: unloadDisabledReason = "Unload after this load"
        case .unloading: unloadDisabledReason = "Unloading now"
        case .empty: unloadDisabledReason = "No model in memory"
        case .ready: unloadDisabledReason = nil
        }
        switchDisabledReason = switch status.phase {
        case .loading: "Switch after this load"
        case .unloading: "Switch after the unload"
        case .ready, .empty: nil
        }

        let canSwitch = switchDisabledReason == nil
        modelRows = options.map { option in
            let gated = EngineService.memoryGateReason(option.model, physicalMemory: physicalMemory)
            let size = memory(option.model)
            let detail: String
            var enabled = canSwitch
            if gated != nil {
                detail = "Too large for this Mac"
                enabled = false
            } else {
                switch option.availability {
                case .loaded:
                    detail = ["Loaded", size].compactMap { $0 }.joined(separator: " · ")
                case .loading:
                    detail = "Loading"
                case .unloading:
                    detail = "Unloading"
                case .notInstalled:
                    detail = "Not installed"
                    enabled = false
                case .ready:
                    let verdict = option.sizeBytes > 0
                        ? fit.verdict(for: option.model, sizeBytes: option.sizeBytes) : .fits
                    let word = switch verdict {
                    case .fits: option.model == .apple ? "Ready" : "Fits"
                    case .tight: "Tight"
                    case .tooLarge: "Too large"
                    }
                    if verdict == .tooLarge { enabled = false }
                    detail = [word, size].compactMap { $0 }.joined(separator: " · ")
                }
            }
            return ModelRow(model: option.model, group: option.model.group, title: option.model.shortName,
                            detail: detail, isResident: option.model == resident && status.phase == .ready,
                            isEnabled: enabled && option.model != resident)
        }

        downloads = entries.compactMap { entry in
            let name = AppModel(modelID: entry.id)?.shortName ?? entry.productName
            switch entry.state {
            case .downloading(let progress):
                let percent = progress.fraction.map { " · \(Int(($0 * 100).rounded(.down)))%" } ?? ""
                return "\(name)\(percent) · \(Self.transfer(progress))"
            case .paused(let progress):
                return "\(name) paused · \(Self.transfer(progress))"
            case .downloadFailed(let message):
                return "\(name) didn't download: \(message)"
            default:
                return nil
            }
        }
    }

    /// "Answering · 2 waiting", "Idle".
    static func activity(_ status: EngineServiceStatus) -> String {
        switch (status.isGenerating, status.queuedRequests) {
        case (true, 0): "Answering"
        case (true, let queued): "Answering · " + waiting(queued)
        case (false, 0): "Idle"
        case (false, let queued): waiting(queued)
        }
    }

    static func waiting(_ count: Int) -> String {
        count == 1 ? "1 request waiting" : "\(count) requests waiting"
    }

    /// "4.5 GB", one decimal, decimal gigabytes like the model library.
    static func gigabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
    }

    /// "4.6 of 12.1 GB"
    static func transfer(_ progress: TransferProgress) -> String {
        let done = Double(progress.completedBytes) / 1_000_000_000
        guard let total = progress.totalBytes else { return String(format: "%.1f GB", done) }
        return String(format: "%.1f of %.1f GB", done, Double(total) / 1_000_000_000)
    }
}
