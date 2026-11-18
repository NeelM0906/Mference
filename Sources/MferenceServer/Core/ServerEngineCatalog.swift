import Foundation
import Mference
import MferenceEngine

/// The catalog library mode serves, shaped for the server.
///
/// Production library mode takes its catalog from the engine's own discovery
/// (`EngineCatalogDiscovery`): the product models by default, every runnable
/// install with `--catalog all`. Two server-side adjustments follow:
///
/// - A model whose engine has no backend in this build (the GGUF models until
///   the llama.cpp backend is linked) is declined rather than listed, so
///   `/v1/models` never offers a model every request would fail on; every
///   GGUF then says so, whatever else discovery said about it.
/// - A declined install's reason names the server's switch, `--catalog all`,
///   where the engine says "the developer catalog scope".
public enum ServerEngineCatalog {
    /// Every reason for declining a runnable install that `--catalog all`
    /// would list ends with this.
    static let allScopeHint = "--catalog all lists it"
    /// The engine's wording for the same switch (`EngineCatalogDiscovery`).
    static let engineScopeHint = "the developer catalog scope lists it"

    /// Whether this build's llama.cpp backend runs Deep with MTP draft
    /// decoding. Deep is offered only with MTP (spec D1 and section 3), so
    /// until the backend has it Deep is declined with `deepNeedsMTP`. The PR
    /// that brings MTP to `LlamaBackend` flips this.
    public static let llamaBackendRunsDeepWithMTP = false
    static let deepNeedsMTP = "Deep is served only with MTP draft decoding, which this build's llama.cpp backend does not have yet"

    /// `catalog` restricted to the engines that have a backend.
    public static func servable(_ catalog: EngineCatalog,
                                engines: Set<EngineKind>,
                                deepWithMTP: Bool = llamaBackendRunsDeepWithMTP) -> EngineCatalog {
        var models: [EngineModelDescriptor] = []
        var skipped = catalog.skipped.map { skip -> EngineCatalogSkip in
            // A GGUF the product catalog declined would be declined again by
            // --catalog all without a llama.cpp backend: say what is missing.
            if !engines.contains(.llamaCpp), skip.location.pathExtension.lowercased() == "gguf" {
                return EngineCatalogSkip(location: skip.location,
                                         reason: "no llama.cpp backend in this build",
                                         modelID: skip.modelID)
            }
            return EngineCatalogSkip(location: skip.location,
                                     reason: skip.reason.replacingOccurrences(of: engineScopeHint,
                                                                              with: allScopeHint),
                                     modelID: skip.modelID)
        }
        for model in catalog.models {
            if model.product == .deep, engines.contains(.llamaCpp), !deepWithMTP {
                skipped.append(EngineCatalogSkip(location: model.location, reason: deepNeedsMTP,
                                                 modelID: model.id))
            } else if engines.contains(model.engine) {
                models.append(model)
            } else {
                skipped.append(EngineCatalogSkip(location: model.location,
                                                 reason: "no \(model.engine.rawValue) backend in this build",
                                                 modelID: model.id))
            }
        }
        return EngineCatalog(models: models, skipped: skipped)
    }

    /// Every entry of a server discovery index, under its discovery
    /// identifier: the catalog behind `ServerModelLibrary(index:loader:)`,
    /// which tests use to drive library mode with injected loaders.
    public static func catalog(
        for index: ServerLibraryIndex,
        installedBytes: (ServerLibraryEntry) -> UInt64? = ServerLibraryListing.receiptInstalledBytes
    ) -> EngineCatalog {
        EngineCatalog(
            models: index.entries.map { entry in
                let product = EngineProductSlot(modelID: entry.modelID)
                return EngineModelDescriptor(
                    id: entry.modelID,
                    displayName: product?.displayName ?? entry.modelID,
                    engine: .mference,
                    format: .gturbo(family: entry.family,
                                    checkpointID: try? ManifestReader.peekModelID(directoryURL: entry.directory)),
                    location: entry.directory,
                    sizeBytes: installedBytes(entry) ?? 0,
                    contextLength: entry.family.maximumContext,
                    product: product)
            },
            skipped: index.skipped.map {
                EngineCatalogSkip(location: $0.directory, reason: $0.reason, modelID: $0.modelID)
            })
    }

    /// How many runnable installs the product catalog declined that
    /// `--catalog all` would list.
    static func declinedRunnableInstalls(_ catalog: EngineCatalog) -> Int {
        catalog.skipped.filter { $0.reason.hasSuffix(allScopeHint) }.count
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
