import Foundation

/// A loaded model: render a prompt, stream its completion.
///
/// The life of one request is `prepare` → `generate`. Everything that can
/// reject a request (template rendering, the context check) happens in
/// `prepare`, before anything is streamed, so a client can still answer with
/// an error status; once `generate` starts, the response is committed.
///
/// `EngineHost` gives a model to one request at a time, so an implementation
/// may assume `prepare` and `generate` are never called concurrently and that
/// neither runs after `unload` has begun. It must still be `Sendable`: an actor
/// or a class guarding its own state.
public protocol EngineModel: Sendable {
    /// The catalog entry this model was loaded from.
    var descriptor: EngineModelDescriptor { get }
    /// What the loaded model supports; fixed once loaded.
    var traits: EngineModelTraits { get }

    /// Renders `request` with the model's chat template and checks it against
    /// the context. Throws `EngineError.invalidRequest` for a request this
    /// model cannot run.
    func prepare(_ request: EngineRequest) async throws -> EnginePreparedPrompt

    /// Generates the completion of a prompt this model prepared, calling
    /// `onEvent` for each piece in order before returning the whole.
    ///
    /// Cancellation of the calling task stops generation at the next token
    /// boundary and throws `CancellationError`. After a failure or a
    /// cancellation the model stays usable for the next request; any reused
    /// prefix state the failed request touched is discarded.
    func generate(_ prepared: EnginePreparedPrompt,
                  onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws -> EngineCompletion

    /// Releases the model's memory (weights, KV, scratch) now rather than when
    /// the last reference goes. Called once by `EngineHost` before it drops the
    /// model; nothing is called on the model afterwards.
    func unload() async
}

/// Loads models of one engine kind.
public protocol EngineBackend: Sendable {
    var engine: EngineKind { get }

    /// Loads the model `descriptor` names. The host calls this only with no
    /// other model resident, so a backend never needs memory for two models.
    func load(_ descriptor: EngineModelDescriptor,
              options: EngineLoadOptions) async throws -> any EngineModel
}
