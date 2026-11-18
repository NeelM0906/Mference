import Foundation

/// Which models the catalog lists.
public enum EngineCatalogScope: String, Sendable, Hashable, CaseIterable {
    /// Only the product models (spec D1). The default.
    case product
    /// Every runnable model found, product or not: a developer switch
    /// (planned as the server's `--catalog all`).
    case all
}

/// A model the catalog found but does not offer, and why. Reported at startup
/// so a partial install or an unsupported file is visible rather than silently
/// missing from the picker.
public struct EngineCatalogSkip: Sendable, Hashable {
    public let location: URL
    public let reason: String
    /// The identifier the model would have had, when one could be derived; a
    /// request for it gets `EngineError.modelNotRunnable` with `reason`.
    public let modelID: String?

    public init(location: URL, reason: String, modelID: String? = nil) {
        self.location = location
        self.reason = reason
        self.modelID = modelID
    }
}

/// The models one engine host can serve, fixed when it is built.
public struct EngineCatalog: Sendable, Hashable {
    /// Sorted by identifier, so listings are stable across restarts.
    public let models: [EngineModelDescriptor]
    public let skipped: [EngineCatalogSkip]

    /// Two descriptors with one identifier are a programming error in the
    /// discovery that built them; the first in identifier order wins and the
    /// rest are reported as skipped.
    public init(models: [EngineModelDescriptor], skipped: [EngineCatalogSkip] = []) {
        var seen = Set<String>()
        var unique: [EngineModelDescriptor] = []
        var duplicates: [EngineCatalogSkip] = []
        for model in models.sorted(by: { ($0.id, $0.location.path) < ($1.id, $1.location.path) }) {
            if seen.insert(model.id).inserted {
                unique.append(model)
            } else {
                duplicates.append(EngineCatalogSkip(location: model.location,
                                                    reason: "duplicate model identifier \(model.id)"))
            }
        }
        self.models = unique
        self.skipped = skipped + duplicates
    }

    public static let empty = EngineCatalog(models: [])

    public func descriptor(for modelID: String) -> EngineModelDescriptor? {
        models.first { $0.id == modelID }
    }

    /// Why a known but declined model cannot be served, or nil.
    public func unavailableReason(for modelID: String) -> String? {
        skipped.first { $0.modelID == modelID }?.reason
    }

    /// The descriptor for `modelID`, or the error a request for it gets.
    public func resolve(_ modelID: String) throws -> EngineModelDescriptor {
        if let descriptor = descriptor(for: modelID) { return descriptor }
        if let reason = unavailableReason(for: modelID) {
            throw EngineError.modelNotRunnable(id: modelID, reason: reason)
        }
        throw EngineError.unknownModel(modelID)
    }
}
