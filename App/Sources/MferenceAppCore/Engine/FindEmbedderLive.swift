import Foundation
import MferenceEngine
import Synchronization

/// Find's live embedder: EmbeddingGemma 2 through the engine's
/// `LlamaEmbedder`: the text model, and with `mediaModelPath` its image and
/// audio encoders. Only `Engine/` imports the engine, so Find reaches the
/// model through this type.
///
/// The model loads on first use and `unload()` frees it (Find calls it when
/// idle); the media encoders load only when a photo, frame or sound is
/// learned. It is a second, small llama.cpp model beside the one resident
/// chat model (App/CONTRACTS.md): about 310 MB of mapped weights, 865 MB with
/// the media encoders, never an `EngineHost` model, so it never evicts the
/// chat model. Work runs on one serial queue at utility priority.
public final class FindEmbedderLive: FindTextEmbedding {
    public let modelPath: URL
    public let mediaModelPath: URL?
    public var fingerprint: String { "embeddinggemma-2-q8_0@\(FindModel.revision.prefix(12))/\(FindVectors.dimensions)" }
    public var supportsMedia: Bool { mediaModelPath != nil }
    public var mediaFingerprint: String {
        guard mediaModelPath != nil else { return "none" }
        return "mmproj-q8_0@\(FindModel.revision.prefix(12))/" + FindMedia.fingerprint
    }

    private let queue = DispatchQueue(label: "ai.mference.find.embedder", qos: .utility)
    private let slot = Slot()
    /// User-initiated while Find works through a large backlog.
    private let hurry = Atomic<Bool>(false)

    public func setHurry(_ hurry: Bool) async {
        self.hurry.store(hurry, ordering: .relaxed)
    }

    /// The loaded embedder, if any (a class so closures can share the lock).
    private final class Slot: Sendable {
        let embedder = Mutex<LlamaEmbedder?>(nil)
    }

    public init(modelPath: URL, mediaModelPath: URL? = nil) {
        self.modelPath = modelPath
        self.mediaModelPath = mediaModelPath
    }

    public func embed(media: [FindMediaData]) async throws -> [[Float]?] {
        guard mediaModelPath != nil else { return media.map { _ in nil } }
        let inputs = media.map { input -> LlamaMediaInput in
            switch input {
            case .image(let width, let height, let rgb): .image(width: width, height: height, rgb: rgb)
            case .audio(let samples): .audio(samples: samples)
            }
        }
        // Same-size pictures share the encoder pass; all share the text pass.
        return try await withModel { try $0.embed(media: inputs) }
    }

    public func embed(documents: [FindEmbeddingInput]) async throws -> [[Float]] {
        try await run(documents.map(FindEmbeddingPrompts.document))
    }

    public func embed(query: String) async throws -> [Float] {
        try await run([query]).first ?? []
    }

    public func unload() async {
        let loaded = slot.embedder.withLock { embedder -> LlamaEmbedder? in
            defer { embedder = nil }
            return embedder
        }
        guard let loaded else { return }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async {
                loaded.unload()
                continuation.resume()
            }
        }
    }

    private func run(_ texts: [String]) async throws -> [[Float]] {
        try await withModel { try $0.embed(texts) }
    }

    /// Runs `body` on the serial queue with the model, loading it first.
    private func withModel<T: Sendable>(_ body: @escaping @Sendable (LlamaEmbedder) throws -> T) async throws -> T {
        let path = modelPath
        let mediaPath = mediaModelPath
        let qos: DispatchQoS = hurry.load(ordering: .relaxed) ? .userInitiated : .utility
        return try await withCheckedThrowingContinuation { continuation in
            queue.async(qos: qos, flags: .enforceQoS) { [slot] in
                do {
                    let model = try slot.embedder.withLock { embedder -> LlamaEmbedder in
                        if let embedder { return embedder }
                        let loaded = try LlamaEmbedder(modelPath: path, mediaModelPath: mediaPath)
                        embedder = loaded
                        return loaded
                    }
                    continuation.resume(returning: try body(model))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
