import Foundation
import Synchronization
import Mference

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
/// on the library actor.
public final class ServerLibrarySnapshot: Sendable {
    private struct State {
        var index: ServerLibraryIndex
        var loaded: String?
        var loading: String?
    }

    private let state: Mutex<State>

    init(index: ServerLibraryIndex) {
        state = Mutex(State(index: index, loaded: nil, loading: nil))
    }

    public var index: ServerLibraryIndex {
        state.withLock { $0.index }
    }

    public func entry(for modelID: String) -> ServerLibraryEntry? {
        state.withLock { $0.index.entry(for: modelID) }
    }

    public var modelList: OpenAIModelList {
        state.withLock { $0.index.modelList }
    }

    /// `("loading", target)` while a model is being loaded or swapped in,
    /// otherwise `("ok", loaded model or nil)`. Reported so a client can tell a
    /// minutes-long first-touch verification from a wedged server.
    public var health: (status: String, model: String?) {
        state.withLock {
            if let loading = $0.loading { return ("loading", loading) }
            return ("ok", $0.loaded)
        }
    }

    var loadedModelID: String? {
        state.withLock { $0.loaded }
    }

    func beginLoading(_ modelID: String) {
        state.withLock {
            $0.loading = modelID
            $0.loaded = nil
        }
    }

    func finishLoading(_ modelID: String?) {
        state.withLock {
            $0.loading = nil
            $0.loaded = modelID
        }
    }
}

/// Holds the one model that is loaded and swaps it in place.
///
/// Exactly one `ServerModelSession` exists at a time and no second model
/// process is ever spawned: a request for a different model releases the
/// resident session — Metal buffers, expert cache, and KV with it — before the
/// replacement is loaded, so peak memory is one model, not two.
///
/// Serialization is the caller's: `resolve` runs inside the generation
/// coordinator's turn, so the generation that was in flight when the request
/// arrived has finished and nothing else can start until this request releases.
/// The `isLoading` gate below is a second belt for the startup preload and for
/// any future caller outside that turn.
public actor ServerModelLibrary {
    /// Loads the install at a directory. Injected so tests can record load and
    /// release ordering without a real model.
    public typealias Loader = @Sendable (URL) async throws -> any ServerLoadedModel

    public nonisolated let snapshot: ServerLibrarySnapshot

    private let index: ServerLibraryIndex
    private let loader: Loader
    private var current: (modelID: String, backend: any ServerLoadedModel)?
    private var isLoading = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(index: ServerLibraryIndex, loader: @escaping Loader) {
        self.index = index
        self.loader = loader
        self.snapshot = ServerLibrarySnapshot(index: index)
    }

    /// Returns the loaded backend for `modelID`, swapping it in if a different
    /// model is resident. Blocks for the whole load — tens of seconds to
    /// minutes with first-touch SHA-256 verification — rather than returning a
    /// `503` with `Retry-After`, because an OpenAI-compatible client already
    /// tolerates a long time to first token and would otherwise have to
    /// implement a retry loop the API does not describe.
    public func resolve(modelID: String) async throws
        -> (modelID: String, backend: any ServerLoadedModel) {
        guard let entry = index.entry(for: modelID) else {
            throw ServerRequestError.unknownModel
        }
        while isLoading {
            await withCheckedContinuation { waiters.append($0) }
        }
        if let current, current.modelID == entry.modelID { return current }

        isLoading = true
        defer {
            isLoading = false
            let resumable = waiters
            waiters.removeAll()
            for waiter in resumable { waiter.resume() }
        }
        snapshot.beginLoading(entry.modelID)
        // Only the identifier is kept: binding the session itself to a local
        // would hold a strong reference for the rest of this call and keep the
        // outgoing model alive through the load, which is exactly the two-models
        // -at-once peak this swap exists to avoid.
        let previousModelID = current?.modelID
        current = nil
        if let previousModelID {
            ServerLog.modelSwapStarted(from: previousModelID, to: entry.modelID)
        } else {
            ServerLog.modelLoadStarted(entry.modelID)
        }
        let started = ContinuousClock.now
        do {
            let backend = try await loader(entry.directory)
            current = (modelID: entry.modelID, backend: backend)
            snapshot.finishLoading(entry.modelID)
            ServerLog.modelSwapFinished(model: entry.modelID,
                                        duration: started.duration(to: .now))
            return (modelID: entry.modelID, backend: backend)
        } catch {
            snapshot.finishLoading(nil)
            ServerLog.modelSwapFailed(model: entry.modelID, error: error)
            throw error
        }
    }

    /// Startup preload for `--model` alongside `--library`. Without it the
    /// first request pays the load.
    public func preload(modelID: String) async throws {
        _ = try await resolve(modelID: modelID)
    }

    var loadedModelID: String? { current?.modelID }
}
