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
///   `404`; so does the discovery identifier of a served install when it
///   differs from the stable one (`qwen3.6-35b-a3b@qwen36`).
/// - `.all` lists every runnable install under its discovery identifier.
///
/// Only `.gturbo` installs are discovered here; the GGUF product models join
/// the catalog with the engine's own discovery and llama.cpp backend.
public enum ServerEngineCatalog {
    /// Every reason the product catalog gives for declining a runnable install
    /// names the switch that lists it.
    static let allScopeHint = "--catalog all lists it"

    /// - Parameters:
    ///   - preferredDirectory: `--model` alongside `--library`. When it is an
    ///     install of a product model it is the one served for that model.
    ///   - rootOrder: the library roots in the order they were configured.
    ///     Among several installs of one product model (none preferred), the
    ///     one named after the model (`qwen36.gturbo`) wins, then the one under
    ///     the earliest root, then the first by path.
    public static func catalog(
        for index: ServerLibraryIndex,
        scope: EngineCatalogScope,
        preferredDirectory: URL? = nil,
        rootOrder: [URL] = [],
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
            let preferred = preferredDirectory?.standardizedFileURL.path
            func rank(_ entry: ServerLibraryEntry) -> (Int, Int, Int, String) {
                let path = entry.directory.standardizedFileURL.path
                let root = rootOrder.firstIndex {
                    let rootPath = $0.standardizedFileURL.path
                    return path == rootPath || path.hasPrefix(rootPath + "/")
                } ?? Int.max
                return (path == preferred ? 0 : 1, 0, root, path)
            }
            var models: [EngineModelDescriptor] = []
            var chosen: [String: String] = [:]  // discovery id -> slot id
            for slot in EngineProductSlot.allCases where slot.engine == .mference {
                // Installs of the slot's checkpoint: discovery's family
                // identifier is the slot's identifier.
                let candidates = index.entries.filter { $0.familyModelID == slot.modelID }
                guard let pick = candidates.min(by: {
                    var left = rank($0), right = rank($1)
                    left.1 = $0.basename == slot.rawValue ? 0 : 1
                    right.1 = $1.basename == slot.rawValue ? 0 : 1
                    return left < right
                }) else { continue }
                chosen[pick.modelID] = slot.modelID
                models.append(descriptor(for: pick, id: slot.modelID, product: slot,
                                         installedBytes: installedBytes))
            }
            for entry in index.entries {
                if let slotID = chosen[entry.modelID] {
                    // A client that pinned the discovery identifier of the
                    // served install learns its stable identifier.
                    if slotID != entry.modelID {
                        skipped.append(EngineCatalogSkip(
                            location: entry.directory,
                            reason: "\(entry.modelID) is served as \(slotID); request that identifier",
                            modelID: entry.modelID))
                    }
                    continue
                }
                let isProduct = EngineProductSlot(modelID: entry.familyModelID) != nil
                skipped.append(EngineCatalogSkip(
                    location: entry.directory,
                    reason: isProduct
                        ? "another install of \(entry.familyModelID) is served; \(allScopeHint)"
                        : "\(entry.modelID) is not one of the product models; \(allScopeHint)",
                    modelID: entry.modelID == entry.familyModelID && isProduct ? nil : entry.modelID))
            }
            return EngineCatalog(models: models, skipped: skipped)
        }
    }

    /// Whether the product catalog declined runnable installs that
    /// `--catalog all` would list.
    static func declinedRunnableInstalls(_ catalog: EngineCatalog) -> Int {
        catalog.skipped.filter { $0.reason.hasSuffix(allScopeHint) }.count
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
