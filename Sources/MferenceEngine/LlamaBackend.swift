import Foundation
import LlamaCpp
import Mference

/// Failures of the llama.cpp backend itself. Request problems are
/// `EngineError.invalidRequest`; cancellation is `CancellationError`.
public enum LlamaBackendError: Error, Equatable, Sendable, CustomStringConvertible {
    /// The descriptor is not a `.gguf` model.
    case notGGUF(String)
    case modelLoadFailed(String)
    case contextCreationFailed(String)
    case vocabularyUnavailable
    /// The GGUF carries no `tokenizer.chat_template`.
    case missingChatTemplate(String)
    case invalidChatTemplate(String)
    /// The vocabulary lacks a special token the chat dialect needs.
    case unsupportedTokenizer(String)
    /// `llama_decode` failed with this status.
    case decodeFailed(Int32)
    /// llama.cpp could not create the MTP draft context.
    case draftContextFailed(String)

    public var description: String {
        switch self {
        case .notGGUF(let id): "model \(id) is not a GGUF file"
        case .modelLoadFailed(let path): "llama.cpp could not load \(path)"
        case .contextCreationFailed(let id): "llama.cpp could not create a context for \(id)"
        case .vocabularyUnavailable: "the GGUF model has no vocabulary"
        case .missingChatTemplate(let id): "model \(id) has no chat template in its GGUF metadata"
        case .invalidChatTemplate(let reason): "the GGUF chat template does not parse: \(reason)"
        case .unsupportedTokenizer(let reason): "unsupported GGUF tokenizer: \(reason)"
        case .decodeFailed(let status): "llama_decode failed with status \(status)"
        case .draftContextFailed(let id): "llama.cpp could not create the MTP draft context for \(id)"
        }
    }
}

/// Runs `.gguf` models through llama.cpp, linked in-process.
///
/// A load maps the file (`use_mmap`; the weights never pass through Swift
/// memory), offloads every layer to Metal, and creates one context of
/// `descriptor.resolvedContext(options:)` tokens with llama.cpp's default
/// batch sizes and KV cache types. `EngineHost` loads only with no other model
/// resident, and `LlamaModel.unload` frees the context and the model before the
/// host drops it.
public struct LlamaBackend: EngineBackend {
    public var engine: EngineKind { .llamaCpp }

    /// llama.cpp's default logical and physical batch sizes (`n_batch`,
    /// `n_ubatch`); prompts prefill in `logicalBatch`-token chunks exactly as
    /// the upstream tools do, so their logits match.
    public static let logicalBatch = 2048
    public static let physicalBatch = 512
    /// Tokens the MTP head drafts per step: llama.cpp's default
    /// (`--spec-draft-n-max`).
    public static let draftLimit = 3

    public init() {}

    public func load(_ descriptor: EngineModelDescriptor,
                     options: EngineLoadOptions) async throws -> any EngineModel {
        guard case .gguf(let metadata) = descriptor.format else {
            throw LlamaBackendError.notGGUF(descriptor.id)
        }
        LlamaRuntime.initialize()
        // MTP (Deep): load the nextn head too, then decode speculatively with
        // llama.cpp's draft-mtp unless draft decoding is off.
        let wantsDraft = metadata.hasMTP && options.draftDecoding != .off

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = -1
        // Map the file, never read it into process memory; no mlock.
        modelParams.load_mode = LLAMA_LOAD_MODE_MMAP
        modelParams.load_mtp = wantsDraft
        guard let modelPointer = llama_model_load_from_file(descriptor.location.path, modelParams) else {
            throw LlamaBackendError.modelLoadFailed(descriptor.location.path)
        }
        let handle = LlamaModelHandle(modelPointer)

        guard let templatePointer = llama_model_chat_template(modelPointer, nil) else {
            handle.free()
            throw LlamaBackendError.missingChatTemplate(descriptor.id)
        }
        let vocabulary: LlamaVocabulary
        let tokenizer: MFTokenizer
        let template: LlamaChatTemplate
        do {
            vocabulary = try LlamaVocabulary(model: handle)
            do {
                tokenizer = try MFTokenizer(tokenizer: vocabulary, family: nil)
            } catch {
                throw LlamaBackendError.unsupportedTokenizer("\(error)")
            }
            template = try LlamaChatTemplate(source: String(cString: templatePointer),
                                             bosToken: vocabulary.bosToken,
                                             eosToken: vocabulary.eosToken)
        } catch {
            handle.free()
            throw error
        }

        var contextParams = llama_context_default_params()
        contextParams.n_ctx = UInt32(descriptor.resolvedContext(options: options))
        contextParams.n_batch = UInt32(Self.logicalBatch)
        contextParams.n_ubatch = UInt32(Self.physicalBatch)
        contextParams.n_seq_max = 1
        contextParams.no_perf = true
        let useDraft = wantsDraft && llama_model_n_layer_nextn(modelPointer) > 0
        var draftParams = mference_mtp_default_params()
        draftParams.n_draft_max = Int32(Self.draftLimit)
        draftParams.n_batch = Int32(Self.logicalBatch)
        draftParams.n_ubatch = Int32(Self.physicalBatch)
        draftParams.n_threads = contextParams.n_threads
        draftParams.n_threads_batch = contextParams.n_threads_batch
        if useDraft {
            // Recurrent rollback and per-step outputs, as llama.cpp's tools
            // create the target context for draft-mtp.
            mference_mtp_target_params(&contextParams, draftParams)
        }
        guard let context = llama_init_from_model(modelPointer, contextParams) else {
            handle.free()
            throw LlamaBackendError.contextCreationFailed(descriptor.id)
        }
        let threadpool = Self.attachThreadpool(to: context, threads: contextParams.n_threads)
        var mtp: OpaquePointer?
        if useDraft {
            mtp = mference_mtp_init(modelPointer, context, draftParams)
            guard mtp != nil else {
                llama_free(context)
                if let threadpool { ggml_threadpool_free(threadpool) }
                handle.free()
                throw LlamaBackendError.draftContextFailed(descriptor.id)
            }
        }
        return LlamaModel(descriptor: descriptor, handle: handle, context: context,
                          vocabulary: vocabulary, tokenizer: tokenizer, template: template,
                          batchSize: Self.logicalBatch, mtp: mtp, draftLimit: Self.draftLimit,
                          threadpool: threadpool)
    }

    /// Attaches a ggml CPU threadpool to `context`, as llama.cpp's tools do
    /// (`common_init`, default polling level 50), so the CPU side of each
    /// graph reuses live threads instead of starting new ones. Nil, and the
    /// context computes without one, when `MFERENCE_LLAMA_THREADPOOL=off` (a
    /// diagnostic override, not a supported control) or creation fails.
    static func attachThreadpool(to context: OpaquePointer, threads: Int32) -> OpaquePointer? {
        guard ProcessInfo.processInfo.environment["MFERENCE_LLAMA_THREADPOOL"] != "off" else { return nil }
        var params = ggml_threadpool_params_default(threads)
        params.poll = 50
        guard let threadpool = ggml_threadpool_new(&params) else { return nil }
        llama_attach_threadpool(context, threadpool, nil)
        return threadpool
    }
}

/// Process-wide llama.cpp setup: backend registration once, and logging cut
/// down to warnings and errors (llama.cpp logs every tensor at load).
enum LlamaRuntime {
    private static let once: Void = {
        llama_backend_init()
        llama_log_set({ level, text, _ in
            guard let text, level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR else { return }
            fputs(String(cString: text), stderr)
        }, nil)
    }()

    static func initialize() {
        _ = once
    }
}
