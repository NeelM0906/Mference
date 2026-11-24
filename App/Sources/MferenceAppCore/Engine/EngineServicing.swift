import Foundation

// SHARED CONTRACT (App Program spec sections 4.3 and 8). The app talks to the
// engine only through `EngineServicing`. Views and view models import
// MferenceAppCore, never MferenceEngine or Mference directly. Changes to the
// types in this file go through one PR the coordinator approves; U2 owns the
// implementation (`EngineService`).

/// Which picker group a model belongs to (decision L2).
public enum ModelGroup: String, Sendable, Hashable, Codable, CaseIterable {
    /// Quick, Deep, Auto.
    case recommended
    /// Inkling 2 Mini, Qwen 3.6, MiniCPM5, Maple.
    case alsoOnThisMac
    /// Apple's on-device model (FoundationModels), shown only while
    /// Settings > Models > Apple Intelligence model is on.
    case builtIntoMacOS
}

/// The product models the app offers (decision L1), in picker order.
public enum AppModel: String, Sendable, Hashable, Codable, CaseIterable, Identifiable {
    case quick, deep, inkling2Mini, qwen36, minicpm5, maple
    /// Apple's on-device model. Not an engine model: chat routes it to
    /// FoundationModels (`AppleChatRouting`), and the engine, the model
    /// library and downloads never list it (`engineModels`).
    case apple

    /// The engine's model identifier (`EngineProductSlot.modelID`).
    public var modelID: String {
        switch self {
        case .quick: "qwen3.5-4b"
        case .deep: "qwen3.8-27b"
        case .inkling2Mini: "inkling-2-mini-ud-q4_k_xl"
        case .qwen36: "qwen3.6-35b-a3b"
        case .minicpm5: "minicpm5-2b-int4g64"
        case .maple: "maple-preview-2bit-mlx"
        case .apple: "apple-foundation-model"
        }
    }

    public var id: String { modelID }

    /// Whether the in-process engine runs it (every model but `.apple`).
    public var isEngineModel: Bool { self != .apple }

    /// The models the engine, the model library and downloads list.
    public static let engineModels: [AppModel] = allCases.filter(\.isEngineModel)

    public init?(modelID: String) {
        guard let model = Self.allCases.first(where: { $0.modelID == modelID }) else { return nil }
        self = model
    }

    public var group: ModelGroup {
        switch self {
        case .quick, .deep: .recommended
        case .inkling2Mini, .qwen36, .minicpm5, .maple: .alsoOnThisMac
        case .apple: .builtIntoMacOS
        }
    }

    /// Picker name ("Quick"), mockup 02b.
    public var name: String {
        switch self {
        case .quick: "Quick"
        case .deep: "Deep"
        case .inkling2Mini: "Inkling 2 Mini"
        case .qwen36: "Qwen 3.6"
        case .minicpm5: "MiniCPM5"
        case .maple: "Maple"
        case .apple: "Apple Intelligence"
        }
    }

    /// Short name on the composer button and under answers ("Quick 4B").
    public var shortName: String {
        switch self {
        case .quick: "Quick 4B"
        case .deep: "Deep 27B"
        case .inkling2Mini: "Inkling 2"
        case .qwen36: "Qwen 3.6"
        case .minicpm5: "MiniCPM5"
        case .maple: "Maple"
        case .apple: AppleModelLabels.current.shortName
        }
    }

    /// The detail after the name in the picker ("Qwen3.5 4B · default").
    public var detail: String {
        switch self {
        case .quick: "Qwen3.5 4B · default"
        case .deep: "Qwen3.8 27B · MTP"
        case .inkling2Mini: "Mini"
        case .qwen36: "35B-A3B"
        case .minicpm5: "2B"
        case .maple: "Preview"
        case .apple: AppleModelLabels.current.detail
        }
    }

    /// One-line description in the picker.
    public var summary: String {
        switch self {
        case .quick: "Everyday chat, mail and web answers. Runs on every Mac and iPhone."
        case .deep: "Harder reasoning and long documents. Drafts several tokens at once."
        case .inkling2Mini: "Deep-tier mixture-of-experts model."
        case .qwen36: "Large mixture-of-experts. Experts stream from the SSD."
        case .minicpm5: "Very small and quick. Good for short replies."
        case .maple: "Experimental model."
        case .apple: "Apple’s on-device model. Short chats and quick rewrites. Nothing to download."
        }
    }

    /// Engine label shown small in the picker and in Settings > Details only.
    public var engineLabel: String {
        switch self {
        case .quick, .deep: "llama.cpp"
        case .inkling2Mini, .qwen36, .minicpm5, .maple: "Mference"
        case .apple: "Apple"
        }
    }

    /// Shown with a Preview badge.
    public var isPreview: Bool { self == .maple || self == .inkling2Mini }

    /// Whether the model reasons before answering by default (steps disclosure).
    public var supportsThinking: Bool {
        switch self {
        case .quick, .deep, .inkling2Mini, .qwen36: true
        case .minicpm5, .maple, .apple: false
        }
    }
}

/// What the picker shows for one model (spec 4.3: Loaded / Ready / Loading /
/// Not installed).
public enum ModelAvailability: Sendable, Hashable {
    /// Resident in memory now.
    case loaded
    /// Installed and runnable; loads on first use.
    case ready
    /// Being loaded. Loads cannot be interrupted.
    case loading
    /// Being released (idle unload, swap, or quit).
    case unloading
    /// Not on this Mac, or the install was declined; `reason` is user-facing.
    case notInstalled(reason: String)
}

/// One row of the picker and the model library.
public struct ModelOption: Sendable, Hashable, Identifiable {
    public let model: AppModel
    public var availability: ModelAvailability
    /// Size on disk, 0 when not installed.
    public var sizeBytes: UInt64
    /// Context the engine resolves for this model, 0 when not installed.
    public var contextLength: Int
    /// Install location, nil when not installed.
    public var location: URL?

    public var id: String { model.modelID }

    public init(model: AppModel, availability: ModelAvailability,
                sizeBytes: UInt64 = 0, contextLength: Int = 0, location: URL? = nil) {
        self.model = model
        self.availability = availability
        self.sizeBytes = sizeBytes
        self.contextLength = contextLength
        self.location = location
    }
}

/// The engine's state, for the picker, the toolbar and Settings.
public struct EngineServiceStatus: Sendable, Hashable {
    public enum Phase: String, Sendable, Hashable { case empty, loading, ready, unloading }

    public var phase: Phase
    /// The model loaded and usable; nil while loading or unloading.
    public var residentModel: AppModel?
    /// The model being loaded or released.
    public var targetModel: AppModel?
    /// A generation is running.
    public var isGenerating: Bool
    /// Requests waiting their turn.
    public var queuedRequests: Int
    /// User-facing message for the most recent failed load, if any.
    public var lastLoadFailure: String?
    /// The model that most recent failed load was for (cleared with the
    /// message by the next successful load), so a library row can offer Retry.
    public var lastLoadFailureModel: AppModel?
    /// Bumped whenever the catalog changes (a download finished, a refresh
    /// found new installs). Subscribers that list models re-read `models()`
    /// when it changes.
    public var catalogRevision: Int

    public init(phase: Phase = .empty, residentModel: AppModel? = nil, targetModel: AppModel? = nil,
                isGenerating: Bool = false, queuedRequests: Int = 0, lastLoadFailure: String? = nil,
                lastLoadFailureModel: AppModel? = nil, catalogRevision: Int = 0) {
        self.phase = phase
        self.residentModel = residentModel
        self.targetModel = targetModel
        self.isGenerating = isGenerating
        self.queuedRequests = queuedRequests
        self.lastLoadFailure = lastLoadFailure
        self.lastLoadFailureModel = lastLoadFailureModel
        self.catalogRevision = catalogRevision
    }

    public static let empty = EngineServiceStatus()
}

/// A JSON value for tool schemas and tool-call arguments, so callers need not
/// import the engine's own type.
public indirect enum ToolJSON: Sendable, Hashable, Codable {
    case null
    case bool(Bool)
    case integer(Int64)
    case number(Double)
    case string(String)
    case array([ToolJSON])
    case object([String: ToolJSON])
}

/// A function the model may call (`web_search`, `open_page`, ...).
public struct ToolDefinition: Sendable, Hashable {
    public var name: String
    public var description: String
    /// JSON Schema of the arguments.
    public var parameters: ToolJSON

    public init(name: String, description: String, parameters: ToolJSON) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// A tool call the model made, or one replayed from history.
public struct ToolCall: Sendable, Hashable, Codable, Identifiable {
    public var id: String
    public var name: String
    public var arguments: ToolJSON
    /// The arguments exactly as the model wrote them.
    public var argumentsJSON: String

    public init(id: String, name: String, arguments: ToolJSON, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.argumentsJSON = argumentsJSON
    }
}

/// One turn of history sent to the model.
public struct ChatTurn: Sendable, Hashable {
    public enum Role: String, Sendable, Hashable, Codable { case system, user, assistant, tool }

    public var role: Role
    public var content: String?
    /// Assistant thinking, replayed in history (engine contract).
    public var reasoning: String?
    /// Calls an assistant turn made.
    public var toolCalls: [ToolCall]
    /// For `role == .tool`: the call this result answers.
    public var toolCallID: String?
    /// For `role == .tool`: the tool's name.
    public var toolName: String?

    public init(role: Role, content: String?, reasoning: String? = nil, toolCalls: [ToolCall] = [],
                toolCallID: String? = nil, toolName: String? = nil) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.toolName = toolName
    }
}

/// Per-request options. Sampling uses the model's own defaults unless
/// `temperature` is set.
public struct GenerationOptions: Sendable, Hashable {
    /// Thinking on maps to reasoning effort `.xhigh`, off to `.off`; nil
    /// leaves the model's default.
    public var thinking: Bool?
    public var maximumCompletionTokens: Int
    public var tools: [ToolDefinition]
    public var temperature: Float?
    public var seed: UInt64?

    public init(thinking: Bool? = nil, maximumCompletionTokens: Int = 4_096,
                tools: [ToolDefinition] = [], temperature: Float? = nil, seed: UInt64? = nil) {
        self.thinking = thinking
        self.maximumCompletionTokens = maximumCompletionTokens
        self.tools = tools
        self.temperature = temperature
        self.seed = seed
    }
}

/// Why a generation ended.
public enum GenerationFinish: String, Sendable, Hashable, Codable {
    case stop, length, toolCalls
}

/// The summary that ends a successful generation.
public struct GenerationSummary: Sendable, Hashable {
    public var content: String
    public var reasoning: String?
    public var toolCalls: [ToolCall]
    public var finish: GenerationFinish
    public var promptTokens: Int
    public var completionTokens: Int
    public var prefillSeconds: Double?
    public var decodeSeconds: Double?
    /// The oldest turns were left out so the chat fits the model's context
    /// (Apple's model, 4K tokens on most Macs); the chat shows a notice.
    public var historyTrimmed: Bool

    public init(content: String, reasoning: String?, toolCalls: [ToolCall], finish: GenerationFinish,
                promptTokens: Int, completionTokens: Int,
                prefillSeconds: Double? = nil, decodeSeconds: Double? = nil, historyTrimmed: Bool = false) {
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.finish = finish
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.historyTrimmed = historyTrimmed
    }
}

/// One element of a generation stream, in order: `prepared`, then any mix of
/// `content` / `reasoning` / `toolCall`, then `completed`.
public enum GenerationUpdate: Sendable, Hashable {
    /// The prompt was accepted; generation starts next.
    case prepared(promptTokens: Int)
    case content(String)
    case reasoning(String)
    case toolCall(ToolCall)
    case completed(GenerationSummary)
}

/// One generation as an async sequence. Each iteration is one request,
/// started by `makeAsyncIterator()`; the iterator owns the request, so
/// releasing it (or cancelling the iterating task) cancels the request.
public struct GenerationStream: AsyncSequence, Sendable {
    public typealias Element = GenerationUpdate
    public typealias Continuation = AsyncThrowingStream<GenerationUpdate, any Error>.Continuation

    private let start: @Sendable (Continuation) -> Task<Void, Never>

    /// `start` launches the request, yields its updates into the
    /// continuation and finishes it; it must stop when its task is cancelled.
    public init(start: @escaping @Sendable (Continuation) -> Task<Void, Never>) {
        self.start = start
    }

    public func makeAsyncIterator() -> Iterator {
        let (elements, continuation) = AsyncThrowingStream<GenerationUpdate, any Error>.makeStream()
        let task = start(continuation)
        continuation.onTermination = { _ in task.cancel() }
        return Iterator(base: elements.makeAsyncIterator(), request: RequestHandle(task: task))
    }

    public struct Iterator: AsyncIteratorProtocol {
        var base: AsyncThrowingStream<GenerationUpdate, any Error>.Iterator
        let request: RequestHandle
        var completed = false

        /// Ends with `.completed`, or throws: a request that stops because
        /// the iterating task was cancelled throws `CancellationError`
        /// rather than ending quietly without a completion.
        public mutating func next() async throws -> GenerationUpdate? {
            let element = try await base.next()
            if case .completed? = element { completed = true }
            if element == nil, !completed { try Task.checkCancellation() }
            return element
        }
    }

    /// Cancels the request when the last iterator copy goes away.
    final class RequestHandle: Sendable {
        let task: Task<Void, Never>

        init(task: Task<Void, Never>) { self.task = task }

        deinit { task.cancel() }
    }
}

/// Errors the service throws, each with one user-facing message.
public enum EngineServiceError: Error, Sendable, Hashable {
    case unknownModel(String)
    case notInstalled(model: String, reason: String)
    case engineUnavailable
    case requestTooLarge(String)
    /// The model cannot run this request (template or option not supported).
    case invalidRequest(String)
    case busy
    case shuttingDown
    case modelUnloaded
    case loadFailed(String)
    /// Generation stopped on an engine error after it started.
    case generationFailed(String)

    /// The sentence the UI shows.
    public var userMessage: String {
        switch self {
        case .unknownModel: "That model is not available in this version of Mference."
        case .notInstalled(let model, _): "\(model) is not installed on this Mac."
        case .engineUnavailable: "This model's engine is not included in this build."
        case .requestTooLarge(let message): "This conversation is too long for the model. \(message)"
        case .invalidRequest(let message): "This model cannot answer that request: \(message)"
        case .busy: "Mference is busy with other requests. Try again in a moment."
        case .shuttingDown: "Mference is quitting."
        case .modelUnloaded: "The model was unloaded. Send the message again."
        case .loadFailed(let message): "The model could not be loaded: \(message)"
        case .generationFailed(let message): "The answer stopped because of an error: \(message)"
        }
    }
}

/// The app's one door to the engine (spec 4.3). One resident model, one
/// generation at a time (queued), swaps release the old model first.
public protocol EngineServicing: Sendable {
    /// Every product model, installed or not, in picker order.
    func models() async -> [ModelOption]
    /// Re-runs discovery (after a download finishes).
    func refreshCatalog() async
    /// The current status, readable without awaiting.
    var status: EngineServiceStatus { get }
    /// Status changes; yields the current status first.
    func statusUpdates() -> AsyncStream<EngineServiceStatus>
    /// Makes `model` resident (warm-up when a chat opens with it).
    func load(_ model: AppModel) async throws
    /// Releases the resident model after the running and queued requests.
    func unload() async throws
    /// Streams one generation, started when iteration starts. Cancelling the
    /// consuming task, or dropping the iterator (breaking out of the loop),
    /// stops it at the next token, even if the caller keeps the stream.
    func generate(model: AppModel, turns: [ChatTurn], options: GenerationOptions) -> GenerationStream
    /// Stops the running generation, if any.
    func cancelActiveGeneration() async
    /// Unloads and refuses further work; call on quit.
    func shutdown() async
}
