import Foundation
import Mference
import MferenceEngine

/// Builds the engine catalog library mode serves from the server's install
/// discovery (`ServerLibraryDiscovery`).
///
/// - `.product` lists only the product models (spec D1) under their stable
///   `EngineProductSlot` identifiers. Every other runnable install —
///   Flash-Next included — is declined with a reason that names `--catalog
///   all`, so a request for it gets `400 model_not_runnable` rather than a bare
///   `404`.
/// - `.all` lists every runnable install under its discovery identifier.
///
/// Only `.gturbo` installs are discovered here; the GGUF product models join
/// the catalog with the engine's own discovery and llama.cpp backend.
public enum ServerEngineCatalog {
    public static func catalog(
        for index: ServerLibraryIndex,
        scope: EngineCatalogScope,
        installedBytes: (ServerLibraryEntry) -> UInt64? = ServerLibraryListing.receiptInstalledBytes
    ) -> EngineCatalog {
        var skipped = index.skipped.map {
            EngineCatalogSkip(location: $0.directory, reason: $0.reason, modelID: $0.modelID)
        }
        switch scope {
        case .all:
            return EngineCatalog(
                models: index.entries.map {
                    descriptor(for: $0, id: $0.modelID, product: EngineProductSlot(modelID: $0.modelID),
                               installedBytes: installedBytes)
                },
                skipped: skipped)
        case .product:
            var models: [EngineModelDescriptor] = []
            var chosen = Set<String>()
            for slot in EngineProductSlot.allCases where slot.engine == .mference {
                // Installs of the slot's checkpoint: discovery's family
                // identifier is the slot's identifier. With several, the one
                // named after the slot (`qwen36.gturbo`) wins, else the first
                // in discovery order; the rest are declined, not hidden.
                let candidates = index.entries.filter { $0.familyModelID == slot.modelID }
                guard let pick = candidates.first(where: { $0.basename == slot.rawValue })
                        ?? candidates.first else { continue }
                chosen.insert(pick.modelID)
                models.append(descriptor(for: pick, id: slot.modelID, product: slot,
                                         installedBytes: installedBytes))
            }
            for entry in index.entries where !chosen.contains(entry.modelID) {
                let isProduct = EngineProductSlot(modelID: entry.familyModelID) != nil
                skipped.append(EngineCatalogSkip(
                    location: entry.directory,
                    reason: isProduct
                        ? "another install of \(entry.familyModelID) is served; --catalog all lists both"
                        : "\(entry.modelID) is not one of the product models; --catalog all lists it",
                    // A second install of a product model keeps its own
                    // identifier free: the slot identifier names the served one.
                    modelID: entry.modelID == entry.familyModelID && isProduct ? nil : entry.modelID))
            }
            return EngineCatalog(models: models, skipped: skipped)
        }
    }

    static func descriptor(for entry: ServerLibraryEntry,
                           id: String,
                           product: EngineProductSlot?,
                           installedBytes: (ServerLibraryEntry) -> UInt64?) -> EngineModelDescriptor {
        EngineModelDescriptor(
            id: id,
            displayName: product?.displayName ?? entry.modelID,
            engine: .mference,
            format: .gturbo(family: entry.family,
                            checkpointID: try? ManifestReader.peekModelID(directoryURL: entry.directory)),
            location: entry.directory,
            sizeBytes: installedBytes(entry) ?? 0,
            contextLength: entry.family.maximumContext,
            product: product)
    }

    /// Writes what the catalog declined and what it serves to stderr in the
    /// `ServerLog` format.
    public static func logDiscovery(_ catalog: EngineCatalog) {
        for skip in catalog.skipped {
            ServerLog.librarySkipped(directory: skip.location.path, reason: skip.reason)
        }
        ServerLog.libraryReady(models: catalog.models.map(\.id))
    }
}
