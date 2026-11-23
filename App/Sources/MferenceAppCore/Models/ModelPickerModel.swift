import Foundation
import Observation

/// What one picker row shows (mockup 02b): Loaded / Ready / Loading /
/// Not installed, with the engine name small under the status.
public struct ModelPickerRow: Sendable, Hashable, Identifiable {
    public enum State: Sendable, Hashable {
        case loaded
        case ready
        /// Being loaded; a load cannot be cancelled (spec 4.3.1).
        case loading
        /// Being released (idle unload, swap or quit).
        case unloading
        case notInstalled(reason: String)
        /// Installed or not, it cannot run on this Mac (Deep on 16 GB or
        /// less, L4).
        case needsMoreMemory(reason: String)
    }

    public let model: AppModel
    public var state: State
    public var sizeBytes: UInt64
    /// The model this chat uses.
    public var isSelected: Bool

    public var id: String { model.modelID }

    /// "Loaded", "Ready", "Loading", "Unloading", "Not installed".
    public var statusText: String {
        switch state {
        case .loaded: "Loaded"
        case .ready: "Ready"
        case .loading: "Loading"
        case .unloading: "Unloading"
        case .notInstalled: "Not installed"
        case .needsMoreMemory:
            switch model {
            case .deep: "Needs more than 16 GB"
            case .inkling2Mini: "Needs 32 GB or more"
            default: "Needs more memory"
            }
        }
    }

    /// Whether picking it can start a chat now (it loads on first use).
    public var isSelectable: Bool {
        switch state {
        case .notInstalled, .needsMoreMemory: false
        case .loaded, .ready, .loading, .unloading: true
        }
    }

    public init(model: AppModel, state: State, sizeBytes: UInt64 = 0, isSelected: Bool = false) {
        self.model = model
        self.state = state
        self.sizeBytes = sizeBytes
        self.isSelected = isSelected
    }

    init(_ option: ModelOption, selected: AppModel?, physicalMemory: UInt64) {
        if let reason = EngineService.memoryGateReason(option.model, physicalMemory: physicalMemory) {
            self.init(model: option.model, state: .needsMoreMemory(reason: reason), sizeBytes: option.sizeBytes,
                      isSelected: option.model == selected)
            return
        }
        let state: State = switch option.availability {
        case .loaded: .loaded
        case .ready: .ready
        case .loading: .loading
        case .unloading: .unloading
        case .notInstalled(let reason): .notInstalled(reason: reason)
        }
        self.init(model: option.model, state: state, sizeBytes: option.sizeBytes,
                  isSelected: option.model == selected)
    }
}

/// The picker's state for one chat, kept current from `EngineServicing`
/// (decision L2: grouped Recommended, with Auto "Coming later" (L3), and Also
/// on this Mac).
///
/// Picking a model warms it up (`load`) so the first answer does not wait for
/// the load; a load cannot be interrupted, so picking another model while one
/// loads queues the second behind it, and the engine keeps one model
/// resident throughout.
@MainActor
@Observable
public final class ModelPickerModel {
    public private(set) var rows: [ModelPickerRow] = []
    public private(set) var status: EngineServiceStatus
    public private(set) var selected: AppModel
    /// The user-facing message of the last failed warm-up, cleared by the
    /// next pick.
    public private(set) var loadError: String?

    /// L3: Auto is shown, but does not pick a model yet.
    public static let autoTitle = "Auto"
    public static let autoDetail = "Coming later. For now you choose the model yourself."

    @ObservationIgnored private let engine: any EngineServicing
    @ObservationIgnored private let physicalMemory: UInt64
    @ObservationIgnored private var warmUp: Task<Void, Never>?

    public init(engine: any EngineServicing, selected: AppModel,
                physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        self.engine = engine
        self.physicalMemory = physicalMemory
        self.selected = selected
        self.status = engine.status
    }

    /// Rows in picker order.
    public var recommended: [ModelPickerRow] { rows.filter { $0.model.group == .recommended } }
    public var alsoOnThisMac: [ModelPickerRow] { rows.filter { $0.model.group == .alsoOnThisMac } }

    public func row(for model: AppModel) -> ModelPickerRow? { rows.first { $0.model == model } }

    /// The composer button: short name, and "Loading" while the chosen model
    /// is being made resident.
    public var isSelectedModelLoading: Bool {
        row(for: selected)?.state == .loading
    }

    /// Starts following the engine. Call from the view's `.task`; it returns
    /// when the task is cancelled or the engine shuts down.
    public func run() async {
        await refresh()
        for await next in engine.statusUpdates() {
            status = next
            await refresh()
        }
    }

    /// Re-reads the rows (after a download, or when the view appears).
    public func refresh() async {
        let options = await engine.models()
        // Read together, so the rows and the status never disagree.
        status = engine.status
        rows = options.map { ModelPickerRow($0, selected: selected, physicalMemory: physicalMemory) }
    }

    /// Chooses `model` for this chat and warms it up. Models not listed yet
    /// (before the first `refresh()`), not installed, or too large for this
    /// Mac are refused.
    @discardableResult
    public func select(_ model: AppModel, warm: Bool = true) -> Bool {
        guard let row = row(for: model), row.isSelectable else { return false }
        selected = model
        loadError = nil
        rows = rows.map { row in
            var row = row
            row.isSelected = row.model == model
            return row
        }
        guard warm else { return true }
        let engine = self.engine
        let previous = warmUp
        warmUp = Task { [weak self] in
            // Loads are serialized by the engine anyway; waiting keeps the
            // last pick last.
            await previous?.value
            // A later pick supersedes this one: do not load a model nobody
            // is waiting for.
            guard let self, self.selected == model else { return }
            do {
                try await engine.load(model)
            } catch is CancellationError {
            } catch {
                guard self.selected == model else { return }
                self.loadError = (error as? EngineServiceError)?.userMessage ?? String(describing: error)
            }
        }
        return true
    }
}
