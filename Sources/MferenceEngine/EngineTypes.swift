import Foundation
import Mference

// The value types every engine backend and every engine client share. They are
// deliberately built from the runtime's own chat, sampling, and tool-call types
// (`MFTokenizer.Message`, `GenerationConfig`, `ParsedToolCall`, ...) so a client
// that already speaks those, such as `MferenceServer`, maps onto the engine
// without a translation layer. See docs/ENGINE.md.

// MARK: - Identity

/// The inference engine underneath a model. Shown to people only in the model
/// picker, and exposed by the API as a plain field (spec D5).
public enum EngineKind: String, Sendable, Hashable, Codable, CaseIterable {
    /// Mference's own Swift/Metal runtime, reading a `.gturbo` install.
    case mference
    /// llama.cpp linked in-process, reading a `.gguf` file.
    case llamaCpp = "llama.cpp"
}

/// The models the product offers (spec D1), and the only ones the catalog lists
/// unless the developer scope `EngineCatalogScope.all` is chosen.
///
/// The raw value is a short internal name; `modelID` is the stable identifier
/// clients send. The native identifiers are the ones `MferenceServer` already
/// used for these families, so existing clients keep working.
public enum EngineProductSlot: String, Sendable, Hashable, Codable, CaseIterable {
    /// Quick: `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-UD-Q4_K_XL.gguf`.
    case quick
    /// Deep: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`,
    /// `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, decoded with MTP.
    case deep
    /// Native Qwen 3.6 35B-A3B install.
    case qwen36
    /// Native MiniCPM5-2B install.
    case minicpm5
    /// Native Maple Preview install.
    case maple

    public var modelID: String {
        switch self {
        case .quick: "qwen3.5-4b"
        case .deep: "qwen3.8-27b"
        case .qwen36: "qwen3.6-35b-a3b"
        case .minicpm5: "minicpm5-2b-int4g64"
        case .maple: "maple-preview-2bit-mlx"
        }
    }

    public var displayName: String {
        switch self {
        case .quick: "Quick (Qwen3.5 4B)"
        case .deep: "Deep (Qwen3.8 27B)"
        case .qwen36: "Qwen 3.6 35B-A3B"
        case .minicpm5: "MiniCPM5 2B"
        case .maple: "Maple Preview"
        }
    }

    public var engine: EngineKind {
        switch self {
        case .quick, .deep: .llamaCpp
        case .qwen36, .minicpm5, .maple: .mference
        }
    }

    public init?(modelID: String) {
        guard let slot = Self.allCases.first(where: { $0.modelID == modelID }) else { return nil }
        self = slot
    }
}

/// Header-only facts about a `.gguf` file, read without touching tensor data.
public struct EngineGGUFMetadata: Sendable, Hashable {
    /// `general.architecture`, for example `qwen35`.
    public let architecture: String
    /// `general.name`, when present.
    public let name: String?
    /// `<architecture>.context_length`, when present.
    public let contextLength: Int?
    public let tensorCount: Int
    /// Tensors of the next-token-prediction (MTP / nextn) head. Non-zero means
    /// the file can drive llama.cpp's MTP speculative decoding.
    public let nextNTensorCount: Int

    public var hasMTP: Bool { nextNTensorCount > 0 }

    public init(architecture: String,
                name: String? = nil,
                contextLength: Int?,
                tensorCount: Int,
                nextNTensorCount: Int) {
        self.architecture = architecture
        self.name = name
        self.contextLength = contextLength
        self.tensorCount = tensorCount
        self.nextNTensorCount = nextNTensorCount
    }
}

/// What is on disk behind a descriptor.
public enum EngineModelFormat: Sendable, Hashable {
    /// A `.gturbo` install directory. `checkpointID` is the manifest's model
    /// identifier, which tells checkpoint variants of one family apart.
    case gturbo(family: ModelFamily, checkpointID: String?)
    /// A single `.gguf` file.
    case gguf(EngineGGUFMetadata)
}

/// Everything known about a model before it is loaded: enough to list it,
/// show it in a picker, and choose the backend that runs it.
public struct EngineModelDescriptor: Sendable, Hashable, Identifiable {
    /// Stable identifier clients send (`model` in the OpenAI API).
    public let id: String
    public let displayName: String
    public let engine: EngineKind
    public let format: EngineModelFormat
    /// The `.gturbo` directory or the `.gguf` file.
    public let location: URL
    /// Bytes on disk: the whole install directory, or the GGUF file.
    public let sizeBytes: UInt64
    /// The model's native maximum context. The context a loaded model actually
    /// runs with is `EngineModelTraits.maxContext`.
    public let contextLength: Int
    /// The product slot this model fills, or nil for a model listed only by
    /// the developer scope `EngineCatalogScope.all`.
    public let product: EngineProductSlot?

    /// The context a GGUF model runs with when none is requested (spec
    /// section 3); a backend may still lower it where memory requires.
    public static let defaultGGUFContext = 32_768

    /// The context this model runs with for a requested context, as listed in
    /// `/v1/models` before anything is loaded. Nil asks for the default: the
    /// native maximum of a `.gturbo` install (the server's `--max-context
    /// max`), `defaultGGUFContext` for a GGUF. A GGUF never runs past its
    /// native context.
    public func resolvedContext(requested: Int?) -> Int {
        switch format {
        case .gturbo:
            return requested ?? contextLength
        case .gguf:
            return min(requested ?? Self.defaultGGUFContext, contextLength)
        }
    }

    /// The context this model runs with under `options`: the request for its
    /// own format, resolved as `resolvedContext(requested:)` does.
    public func resolvedContext(options: EngineLoadOptions) -> Int {
        switch format {
        case .gturbo: resolvedContext(requested: options.nativeMaxContext)
        case .gguf: resolvedContext(requested: options.ggufMaxContext)
        }
    }

    public init(id: String,
                displayName: String,
                engine: EngineKind,
                format: EngineModelFormat,
                location: URL,
                sizeBytes: UInt64,
                contextLength: Int,
                product: EngineProductSlot?) {
        self.id = id
        self.displayName = displayName
        self.engine = engine
        self.format = format
        self.location = location
        self.sizeBytes = sizeBytes
        self.contextLength = contextLength
        self.product = product
    }
}

// MARK: - Loading

/// How a native model reuses prompt prefixes across requests. Same cases and
/// spellings as the server's `--prompt-cache`.
public enum EnginePromptCacheMode: String, Sendable, Hashable {
    case off
    case singlePrefix = "single-prefix"
    case multiPrefix = "multi-prefix"
}

/// The ceiling on retained prompt-prefix state, as `--prompt-cache-bytes`.
public enum EnginePromptCacheBytes: Sendable, Hashable {
    case auto
    case bytes(Int)
}

/// Whether a backend may use draft (speculative) decoding, such as Deep's MTP
/// head. A backend uses it only where it is greedy-exact.
public enum EngineDraftDecoding: String, Sendable, Hashable {
    /// The backend's default for the model: MTP on for Deep.
    case automatic
    case off
}

/// Options applied when a model is loaded. One value serves every backend; a
/// backend ignores the fields that do not apply to it.
public struct EngineLoadOptions: Sendable, Equatable {
    /// The server's `--max-context` default, applied to native models.
    public static let defaultNativeMaxContext = 16_384

    /// Context for `.gturbo` models; nil is each model's native maximum.
    /// Defaults to the server's `--max-context` default.
    public var nativeMaxContext: Int?
    /// Context for GGUF models; nil is `defaultGGUFContext` (32,768), capped
    /// at the file's native context. A backend may run with less where memory
    /// requires it, and reports what it chose in
    /// `EngineModelTraits.maxContext`.
    public var ggufMaxContext: Int?
    public var draftDecoding: EngineDraftDecoding
    // Native (`.gturbo`) controls; see docs/RUNTIME_CONTROLS.md.
    public var promptCacheMode: EnginePromptCacheMode
    public var promptCacheBytes: EnginePromptCacheBytes
    public var integrityPolicy: ModelIntegrityPolicy
    public var shadowPrefetchBudget: Int?
    public var prefillChunkTokens: Int?
    public var reserveFullKV: Bool

    public init(nativeMaxContext: Int? = EngineLoadOptions.defaultNativeMaxContext,
                ggufMaxContext: Int? = nil,
                draftDecoding: EngineDraftDecoding = .automatic,
                promptCacheMode: EnginePromptCacheMode = .multiPrefix,
                promptCacheBytes: EnginePromptCacheBytes = .auto,
                integrityPolicy: ModelIntegrityPolicy = .trustedReceiptWhenValid,
                shadowPrefetchBudget: Int? = nil,
                prefillChunkTokens: Int? = nil,
                reserveFullKV: Bool = false) {
        self.nativeMaxContext = nativeMaxContext
        self.ggufMaxContext = ggufMaxContext
        self.draftDecoding = draftDecoding
        self.promptCacheMode = promptCacheMode
        self.promptCacheBytes = promptCacheBytes
        self.integrityPolicy = integrityPolicy
        self.shadowPrefetchBudget = shadowPrefetchBudget
        self.prefillChunkTokens = prefillChunkTokens
        self.reserveFullKV = reserveFullKV
    }
}

/// What a loaded model supports, known only once it is loaded (the chat
/// dialect, for example, comes from the loaded tokenizer, not the family).
/// Request validation reads these before anything is rendered.
public struct EngineModelTraits: Sendable {
    public var chatDialect: ChatDialect
    /// Sampling defaults for fields a request omits.
    public var generationDefaults: GenerationConfig
    /// The context the model runs with.
    public var maxContext: Int
    public var isGemmaQAT: Bool
    public var usesSwiftQwenTemplate: Bool
    public var acceptsReasoningEffort: Bool
    public var supportsQwenReasoningEffort: Bool
    /// Whether a later request that extends an earlier one reuses its KV.
    public var reusesPromptPrefixes: Bool
    /// Whether generation uses draft decoding (MTP) right now.
    public var usesDraftDecoding: Bool

    public init(chatDialect: ChatDialect,
                generationDefaults: GenerationConfig = .defaults,
                maxContext: Int,
                isGemmaQAT: Bool = false,
                usesSwiftQwenTemplate: Bool = false,
                acceptsReasoningEffort: Bool = false,
                supportsQwenReasoningEffort: Bool = false,
                reusesPromptPrefixes: Bool = false,
                usesDraftDecoding: Bool = false) {
        self.chatDialect = chatDialect
        self.generationDefaults = generationDefaults
        self.maxContext = maxContext
        self.isGemmaQAT = isGemmaQAT
        self.usesSwiftQwenTemplate = usesSwiftQwenTemplate
        self.acceptsReasoningEffort = acceptsReasoningEffort
        self.supportsQwenReasoningEffort = supportsQwenReasoningEffort
        self.reusesPromptPrefixes = reusesPromptPrefixes
        self.usesDraftDecoding = usesDraftDecoding
    }
}

// MARK: - Requests and results

/// One chat request, already validated by the client (the server validates the
/// OpenAI body against `EngineModelTraits` before building this).
public struct EngineRequest: Sendable {
    public var messages: [MFTokenizer.Message]
    public var tools: [MFTokenizer.FunctionDefinition]
    /// Sampling and stop strings. `maxNewTokens` is ignored; the cap is
    /// `maximumCompletionTokens`, further limited by the context left.
    public var generationConfig: GenerationConfig
    public var maximumCompletionTokens: Int
    public var reasoningEffort: QwenReasoningEffort?
    public var preserveThinking: Bool

    public init(messages: [MFTokenizer.Message],
                tools: [MFTokenizer.FunctionDefinition] = [],
                generationConfig: GenerationConfig = .defaults,
                maximumCompletionTokens: Int,
                reasoningEffort: QwenReasoningEffort? = nil,
                preserveThinking: Bool = false) {
        self.messages = messages
        self.tools = tools
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.reasoningEffort = reasoningEffort
        self.preserveThinking = preserveThinking
    }

    /// Whether the prompt needs the tool-aware template and decoder: tools are
    /// offered, or the history carries developer, tool, or tool-call turns.
    public var needsToolTemplate: Bool {
        !tools.isEmpty || messages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
    }
}

/// A request rendered by the model that will run it and measured against its
/// context. Produced by `EngineModel.prepare`, consumed by
/// `EngineModel.generate` on the same model.
public struct EnginePreparedPrompt: Sendable {
    public let request: EngineRequest
    /// The rendered prompt in the model's own token ids.
    public let promptTokens: [Int32]
    public let needsToolTemplate: Bool

    public init(request: EngineRequest, promptTokens: [Int32], needsToolTemplate: Bool) {
        self.request = request
        self.promptTokens = promptTokens
        self.needsToolTemplate = needsToolTemplate
    }
}

/// One streamed piece of a completion, in generation order.
public enum EngineEvent: Equatable, Sendable {
    /// Visible answer text, after stop strings are applied.
    case content(String)
    /// Thinking text, sent apart from the answer.
    case reasoning(String)
    case toolCall(ParsedToolCall)
}

/// Why generation ended. The raw values are the OpenAI `finish_reason`s.
public enum EngineFinishReason: String, Sendable, Hashable {
    case stop
    case length
    case toolCalls = "tool_calls"
}

public struct EngineUsage: Sendable, Hashable {
    public var promptTokens: Int
    public var completionTokens: Int
    /// Prompt tokens served from reused prefix state, when the backend knows.
    public var cachedPromptTokens: Int?
    /// Completion tokens spent thinking, when the backend counts them.
    public var reasoningTokens: Int?
    /// Completion tokens that reached the answer, when the backend counts
    /// them and no stop string cut the answer short.
    public var visibleTokens: Int?

    public var totalTokens: Int { promptTokens + completionTokens }

    public init(promptTokens: Int,
                completionTokens: Int,
                cachedPromptTokens: Int? = nil,
                reasoningTokens: Int? = nil,
                visibleTokens: Int? = nil) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.reasoningTokens = reasoningTokens
        self.visibleTokens = visibleTokens
    }
}

/// Wall-clock phases of one completion, and draft-decoding counts where used.
public struct EngineTimings: Sendable, Hashable {
    public var prefillSeconds: Double?
    public var decodeSeconds: Double?
    public var draftTokensProposed: Int?
    public var draftTokensAccepted: Int?

    public init(prefillSeconds: Double? = nil,
                decodeSeconds: Double? = nil,
                draftTokensProposed: Int? = nil,
                draftTokensAccepted: Int? = nil) {
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.draftTokensProposed = draftTokensProposed
        self.draftTokensAccepted = draftTokensAccepted
    }
}

/// The finished result. `content`, `reasoningContent`, and `toolCalls` equal
/// the concatenation of the events streamed for it.
public struct EngineCompletion: Sendable {
    public var content: String
    public var reasoningContent: String?
    public var toolCalls: [ParsedToolCall]
    public var finishReason: EngineFinishReason
    public var usage: EngineUsage
    public var timings: EngineTimings
    /// Operator-only runtime diagnostics from the native backend, never part
    /// of an API response body.
    public var diagnostics: RuntimeDiagnostics?

    public init(content: String,
                reasoningContent: String? = nil,
                toolCalls: [ParsedToolCall] = [],
                finishReason: EngineFinishReason,
                usage: EngineUsage,
                timings: EngineTimings = EngineTimings(),
                diagnostics: RuntimeDiagnostics? = nil) {
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
        self.timings = timings
        self.diagnostics = diagnostics
    }
}

// MARK: - Errors

/// Errors the engine itself raises. A backend's own load or runtime failure
/// propagates unchanged; cancellation is always `CancellationError`.
public enum EngineError: Error, Equatable, Sendable, CustomStringConvertible {
    /// No model with this identifier is in the catalog.
    case unknownModel(String)
    /// The catalog found the model but declined it, with the reason.
    case modelNotRunnable(id: String, reason: String)
    /// No backend for this engine is linked into this build.
    case backendUnavailable(EngineKind)
    /// The request cannot run on this model. Raised by `prepare` only, before
    /// anything is streamed, so a client can still answer with a 400. `code`
    /// uses the server's spellings (`context_length_exceeded`,
    /// `invalid_chat_template`, ...).
    case invalidRequest(message: String, param: String?, code: String)
    /// The host's waiting line is full.
    case queueFull
    /// The host is shutting down and takes no more work.
    case shuttingDown
    /// The model was unloaded; it cannot prepare or generate any more.
    case modelUnloaded(String)

    public var description: String {
        switch self {
        case .unknownModel(let id): "unknown model \(id)"
        case .modelNotRunnable(let id, let reason): "model \(id) is not runnable: \(reason)"
        case .backendUnavailable(let engine): "no \(engine.rawValue) backend in this build"
        case .invalidRequest(let message, _, _): message
        case .queueFull: "generation queue is full"
        case .shuttingDown: "engine is shutting down"
        case .modelUnloaded(let id): "model \(id) was unloaded"
        }
    }
}
