import Foundation
import Mference
import MferenceEngine

/// A model the library has loaded, and whose chat dialect the request validator
/// needs before the request is rendered.
///
/// The dialect comes from the loaded tokenizer's special tokens rather than
/// from the family, which is why library mode cannot validate a request until
/// the model it names is resident.
public protocol ServerLoadedModel: ServerInferenceBackend {
    var chatDialect: ChatDialect { get }
}

/// Everything `GET /health`, `GET /v1/models`, and the unknown-model check need,
/// readable synchronously from a NIO channel callback while a load is in flight
/// on the engine host.
public final class ServerLibrarySnapshot: Sendable {
    public let catalog: EngineCatalog
    private let host: EngineHost

    init(catalog: EngineCatalog, host: EngineHost) {
        self.catalog = catalog
        self.host = host
    }

    public func descriptor(for modelID: String) -> EngineModelDescriptor? {
        catalog.descriptor(for: modelID)
    }

    public func unavailableReason(for modelID: String) -> String? {
        catalog.unavailableReason(for: modelID)
    }

    /// Each model's context is the one it loads with, resolved from the
    /// descriptor alone, so listing loads nothing.
    public var modelList: OpenAIModelList {
        OpenAIModelList(
            object: "list",
            data: catalog.models.map {
                .init(id: $0.id, object: "model", created: 0, ownedBy: "mference",
                      maxModelLen: $0.resolvedContext(options: host.loadOptions),
                      engine: $0.engine.rawValue)
            })
    }

    /// `("loading", target)` while a model is being loaded or swapped in,
    /// otherwise `("ok", loaded model or nil)`. Reported so a client can tell a
    /// minutes-long first-touch verification from a wedged server.
    public var health: (status: String, model: String?) {
        let health = host.status.health
        return (health.status, health.modelID)
    }

    var loadedModelID: String? {
        host.status.residentModelID
    }
}

/// Library mode's model owner: an `EngineHost` over an `EngineCatalog`.
///
/// Exactly one model is resident at a time and no second model process is
/// ever spawned: a request for a different model releases the resident one —
/// Metal buffers, expert cache, and KV with it — before the replacement is
/// loaded, so peak memory is one model, not two. The host serializes requests,
/// loads, and unloads through one gate; the HTTP layer additionally runs every
/// library request inside its generation coordinator's turn, which keeps the
/// server's queue limit and its committed-stream behavior unchanged.
public final class ServerModelLibrary: Sendable {
    /// Loads the install at a directory. Injected so tests can record load and
    /// release ordering without a real model.
    public typealias Loader = ServerLoaderBackend.Loader

    public let host: EngineHost
    public let snapshot: ServerLibrarySnapshot
    /// Which catalog the library serves; reported on the ready line.
    public let catalogScope: EngineCatalogScope

    /// Serves `catalog` with `backends`; the production entry point.
    public init(catalog: EngineCatalog,
                backends: [any EngineBackend],
                loadOptions: EngineLoadOptions,
                catalogScope: EngineCatalogScope = .all) {
        self.catalogScope = catalogScope
        let host = EngineHost(catalog: catalog,
                              backends: backends,
                              loadOptions: loadOptions,
                              observer: { ServerLog.hostEvent($0) })
        self.host = host
        self.snapshot = ServerLibrarySnapshot(catalog: catalog, host: host)
    }

    /// Serves every entry of `index` through `loader`. `maxContext` is the
    /// server's `--max-context`, nil for `max`; the loader applies the same
    /// value when it loads a model.
    public convenience init(index: ServerLibraryIndex,
                            maxContext: Int? = ServerArguments.defaultMaxContext,
                            loader: @escaping Loader) {
        self.init(catalog: ServerEngineCatalog.catalog(for: index),
                  backends: [ServerLoaderBackend(loader: loader)],
                  loadOptions: EngineLoadOptions(nativeMaxContext: maxContext))
    }

    /// Runs `body` with `modelID` resident and held, swapping it in first if
    /// another model is resident. Blocks for the whole load — tens of seconds
    /// to minutes with first-touch SHA-256 verification — rather than
    /// returning a `503` with `Retry-After`, because an OpenAI-compatible
    /// client already tolerates a long time to first token and would otherwise
    /// have to implement a retry loop the API does not describe.
    ///
    /// Validation, rendering, and generation all belong inside one body: the
    /// model must not be kept after it returns.
    public func withModel<T: Sendable>(
        _ modelID: String,
        _ body: @escaping @Sendable (_ modelID: String, _ model: any ServerLoadedModel) async throws -> T
    ) async throws -> T {
        do {
            return try await host.withModel(modelID) { model in
                try await body(model.descriptor.id, try serverLoadedModel(for: model))
            }
        } catch {
            throw serverError(error)
        }
    }

    /// Library mode as `MferenceServer --library` runs it: the engine's own
    /// discovery over `roots` (and the GGUF directories), the scope
    /// `--catalog` selects, and the engine's backends with the server's
    /// runtime flags as load options. Only engines with a backend in this
    /// build are served; see `ServerEngineCatalog.servable`.
    ///
    /// `explicitModel` is `--model` alongside `--library`: discovery considers
    /// it first, so it is the install served for its product model.
    public static func serving(
        roots: [URL],
        explicitModel: URL? = nil,
        arguments: ServerArguments,
        ggufDirectories: [URL]? = nil,
        fileSystem: EngineCatalogDiscovery.FileSystem = .live,
        backends: [any EngineBackend]? = nil
    ) -> ServerModelLibrary {
        // Production: the engine's native backend, its notices in the
        // server log. Tests pass scripted backends.
        let backends = backends ?? [NativeBackend(onNotice: { ServerLog.nativeBackendNotice($0) })]
        let discovered = EngineCatalogDiscovery.discover(
            roots: roots,
            ggufDirectories: ggufDirectories ?? EngineCatalogDiscovery.defaultGGUFDirectories(roots: roots),
            scope: arguments.catalogScope,
            explicitModel: explicitModel,
            fileSystem: fileSystem)
        let catalog = ServerEngineCatalog.servable(discovered, engines: Set(backends.map(\.engine)))
        return ServerModelLibrary(catalog: catalog,
                                  backends: backends,
                                  loadOptions: arguments.engineLoadOptions,
                                  catalogScope: arguments.catalogScope)
    }

    /// `--list-models` output.
    public var listing: String { ServerLibraryListing.text(for: snapshot.catalog) }

    /// What the catalog declined and what it serves, on stderr.
    public func logDiscovery() { ServerEngineCatalog.logDiscovery(snapshot.catalog) }

    /// The ready line's library fields.
    public var readyDetail: String {
        "catalog=\(catalogScope.rawValue) models=\(snapshot.catalog.models.count)"
    }

    /// Preloads the catalog model at `directory` (`--model` with `--library`).
    public func preload(directory: URL) async throws {
        // Compared by path: a directory URL may or may not carry a trailing
        // slash depending on how it was made.
        let path = directory.standardizedFileURL.path
        guard let model = snapshot.catalog.models.first(where: {
            $0.location.standardizedFileURL.path == path
        }) else {
            // Say why: a declined install's reason names --catalog all.
            let reason = snapshot.catalog.skipped.first {
                $0.location.standardizedFileURL.path == path
            }?.reason ?? "not a completed install"
            throw ServerArgumentError.invalid("--model \(path): \(reason)")
        }
        try await preload(modelID: model.id)
    }

    /// Makes `modelID` resident and returns it. For tests and diagnostics: a
    /// request path uses `withModel`, which holds the model for its duration.
    public func resolve(modelID: String) async throws
        -> (modelID: String, backend: any ServerLoadedModel) {
        try await withModel(modelID) { id, model in (modelID: id, backend: model) }
    }

    /// Startup preload for `--model` alongside `--library`. Without it the
    /// first request pays the load.
    public func preload(modelID: String) async throws {
        do {
            try await host.load(modelID)
        } catch {
            throw serverError(error)
        }
    }

    /// Releases the resident model and refuses further work.
    public func shutdown() async {
        await host.shutdown()
    }

    var loadedModelID: String? { snapshot.loadedModelID }
}
