import Foundation
import Hub
import Tokenizers

public enum MFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case unsupportedForDialect(String)
    /// No `tokenizer.json` was found: not in the install's `tokenizer/`
    /// sidecar, nor in a folder `MFERENCE_TOKENIZER_DIR` names.
    case missingTokenizerSidecar(String)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        case .missingTokenizerSidecar(let path):
            return "no tokenizer.json at \(path), nor in a folder MFERENCE_TOKENIZER_DIR names; "
                + "the install is incomplete, reinstall the model"
        }
    }
}

/// Chat framing dialect, resolved from the loaded tokenizer's special tokens.
///
/// `.minicpm` is detected by the `<function` special token beside
/// `<|im_end|>`, `.chatml` by `<|im_end|>` alone (Qwen-style ChatML). A
/// tokenizer with neither is refused.
public enum ChatDialect: String, Sendable {
    case chatml
    /// MiniCPM5: ChatML framing with `<s>` first, XML `<function …>` tool
    /// calls, `<tool_response>` results inside user turns, two EOS ids.
    case minicpm
}

/// Tokenizer wrapper for the supported model families (ChatML/Qwen, Maple
/// and MiniCPM5).
///
/// Loads the tokenizer sidecars of a completed `.gturbo/tokenizer/`
/// directory (or `MFERENCE_TOKENIZER_DIR`). Exposes typed accessors for the
/// IDs the generator actually needs (BOS / EOS / pad / end-of-turn) and adapts
/// encode/decode to Int32 to match the buffer types kernels consume. Literal
/// control-token text in user content is a trusted-input research-runtime
/// limitation.
public struct MFTokenizer: @unchecked Sendable {
    public let dialect: ChatDialect
    public internal(set) var isBaseQwen38 = false
    var localTokenizerFolder: URL?
    /// Tool grammar is a family contract, independent of thinking policy.
    let usesJSONChatMLToolCalls: Bool
    /// Loaded checkpoints with a binary opt-in thinking control.
    let supportsOptInThinking: Bool
    /// Nominal BOS. For ChatML this is `<|endoftext|>` (the config's unused
    /// `bos_token_id`); it is never prepended — see `encode(_:addBOS:)`.
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    /// These alias the `<think>` / `</think>` markers.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// `<think>` / `</think>` special-token IDs.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int
    /// Omitted controls use the installed checkpoint's generation profile;
    /// explicit caller settings still win.
    public internal(set) var generationDefaults = GenerationConfig.defaults

    /// Maple's and Qwen 3.8's pinned prompts open a live `<think>` block. The
    /// initializer resolves this EOS relationship only when the caller
    /// supplies one of those manifest families; direct loads and other
    /// families retain their old behavior.
    public var generationPromptStartsInThinking: Bool {
        (dialect == .chatml && eosID == endOfTurnID)
            || (dialect == .minicpm && Self.miniCPMDefaultThinking == .enabled)
    }

    /// BOS actually prepended by `encode(_:addBOS:)`; nil for dialects that
    /// never use a BOS prefix (ChatML).
    private let bosPrefixID: Int32?

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load(from folder: URL) async throws -> MFTokenizer {
        try await load(from: folder, family: nil)
    }

    public static func load(from folder: URL,
                            family: ModelFamily?) async throws -> MFTokenizer {
        try await MFTokenizerLoadCoordinator.shared.load(
            .local(folder.standardizedFileURL.path, family))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> MFTokenizer {
        let family = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let checkpointID = try ManifestReader.peekModelID(directoryURL: modelDirectory)
        guard let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) else {
            throw MFTokenizerError.missingTokenizerSidecar(
                modelDirectory.appendingPathComponent("tokenizer", isDirectory: true).path)
        }
        let loaded = try await load(from: folder, family: family)
        return try loaded.forCheckpoint(checkpointID)
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["MFERENCE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(from folder: URL) async throws -> MFTokenizer {
        try await loadUncached(from: folder, family: nil)
    }

    static func loadUncached(from folder: URL,
                             family: ModelFamily?) async throws -> MFTokenizer {
        // What `AutoTokenizer.from(modelFolder:)` does.
        let configuration = LanguageModelConfigurationFromHub(modelFolder: folder)
        guard let tokenizerConfig = try await configuration.tokenizerConfig else {
            throw TokenizerError.missingConfig
        }
        let tokenizerData = try await configuration.tokenizerData
        let underlying = try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig,
                                                 tokenizerData: tokenizerData)
        var value = try MFTokenizer(tokenizer: underlying, family: family)
        value.localTokenizerFolder = folder.standardizedFileURL
        return value
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    public init(tokenizer: any Tokenizer) throws {
        try self.init(tokenizer: tokenizer, family: nil)
    }

    public init(tokenizer: any Tokenizer, family: ModelFamily?) throws {
        self.tokenizer = tokenizer
        self.usesJSONChatMLToolCalls = family == .maple
        self.supportsOptInThinking = family == .qwen36

        let dialect: ChatDialect? =
            if family == .minicpm5
                        || (Self.specialTokenID(tokenizer, Self.miniCPMFunctionOpen) != nil
                            && Self.specialTokenID(tokenizer, Self.imEndMark) != nil) {
                // Shares `<|im_end|>` with ChatML; the `<function` special
                // token is what no Qwen tokenizer carries.
                .minicpm
            } else if Self.specialTokenID(tokenizer, Self.imEndMark) != nil {
                .chatml
            } else {
                nil
            }
        // Every supported family frames its chat with `<|im_end|>`; a
        // tokenizer without it belongs to a retired family's install.
        guard let dialect else {
            throw MFTokenizerError.missingSpecialToken(Self.imEndMark)
        }
        guard family != .maple || dialect == .chatml else {
            throw MFTokenizerError.unsupportedForDialect("Maple requires ChatML framing")
        }
        let resolved = switch dialect {
        case .chatml: try Self.resolveChatMLTokens(tokenizer, family: family)
        case .minicpm: try Self.resolveMiniCPMTokens(tokenizer)
        }

        self.dialect = dialect
        self.bosID = resolved.bosID
        self.bosPrefixID = resolved.bosPrefixID
        self.eosID = resolved.eosID
        self.padID = resolved.padID
        self.endOfTurnID = resolved.endOfTurnID
        self.toolCallStartID = resolved.toolCallStartID
        self.toolCallEndID = resolved.toolCallEndID
        self.toolResponseID = resolved.toolResponseID
        self.toolResponseEndID = resolved.toolResponseEndID
        self.channelStartID = resolved.channelStartID
        self.channelEndID = resolved.channelEndID
        self.thinkStartID = resolved.thinkStartID
        self.thinkEndID = resolved.thinkEndID
        self.stopTokenIDs = resolved.stopTokenIDs
        self.vocabSize = resolved.vocabSize
    }

    private struct ResolvedSpecialTokens {
        let bosID: Int32
        let bosPrefixID: Int32?
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32
        let toolCallEndID: Int32
        let toolResponseID: Int32
        let toolResponseEndID: Int32
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }

    private static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer,
        family: ModelFamily?
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw MFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // `<|im_start|>` is required even though no stored property holds it;
        // template rendering relies on the tokenizer recognizing its text.
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        // Maple and Qwen 3.8 chat templates end the generation prompt inside
        // a live `<think>` block and stop on `<|im_end|>`; the eos ==
        // end-of-turn identity is what `generationPromptStartsInThinking`
        // keys off. A nil family keeps the Qwen 3.6 (non-thinking) behavior.
        let startsInThinking = family == .maple || family == .qwen38
        return ResolvedSpecialTokens(
            bosID: endOfText,
            bosPrefixID: nil,
            eosID: startsInThinking ? imEnd : endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [imEnd, endOfText],
            vocabSize: family == .maple ? 151_936 : 248_320)
    }

    /// Sentinel for token roles a dialect frames as plain text rather than a
    /// single special token. Never a valid token ID, so comparisons against
    /// generated tokens can never match.
    private static let noSuchTokenID: Int32 = -1

    static let miniCPMFunctionOpen = "<function"
    static let miniCPMFunctionClose = "</function>"

    /// MiniCPM5 (`openbmb/MiniCPM5-2B`, tokenizer.json `3e065a55…`): `<s>` 0,
    /// `</s>` 1, `<|im_start|>` 130072, `<|im_end|>` 130073, `<think>` 8 and
    /// `</think>` 9 (added tokens with `special: false`, still single ids),
    /// `<function` 18 / `</function>` 19 / `<param` 20, `<tool_response>` 10 /
    /// `</tool_response>` 11. `config.json` lists two EOS ids, `</s>` and
    /// `<|im_end|>`; both stop generation. The template prepends `<s>` itself
    /// (`add_bos_token` is false), so `bosPrefixID` serves raw prompts only.
    private static func resolveMiniCPMTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw MFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        let bos = try id("<s>")
        let eos = try id("</s>")
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let functionOpen = try id(Self.miniCPMFunctionOpen)
        let functionClose = try id(Self.miniCPMFunctionClose)
        _ = try id("<param")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: bos,
            bosPrefixID: bos,
            eosID: imEnd,
            padID: eos,
            endOfTurnID: imEnd,
            toolCallStartID: functionOpen,
            toolCallEndID: functionClose,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [eos, imEnd],
            vocabSize: 130_560)
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends the dialect's
    /// BOS prefix where it has one (MiniCPM's `<s>`); ChatML has no BOS, so
    /// `addBOS` is a no-op for that dialect.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        guard addBOS, let bosPrefixID else { return base }
        return [bosPrefixID] + base
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat template

    public enum Role: String, Sendable { case system, developer, user, assistant, tool }
    public struct HistoricalToolCall: Sendable, Equatable {
        public let id: String
        public let name: String
        public let arguments: JSONValue

        public init(id: String, name: String, arguments: JSONValue) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?
        public let reasoningContent: String?

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
            self.reasoningContent = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil,
                    reasoningContent: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
            self.reasoningContent = reasoningContent
        }
    }

    static let imStartMark = "<|im_start|>"
    static let imEndMark   = "<|im_end|>"
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch.
    private static let chatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    private static let thinkingChatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n"
    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        switch dialect {
        case .chatml: return try chatMLChatTemplate(messages)
        case .minicpm:
            return try miniCPMRender(messages: messages, tools: [],
                                     thinking: Self.miniCPMDefaultThinking)
        }
    }

    private func chatMLChatTemplate(_ messages: [Message]) throws -> String {
        var s = ""
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw MFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = generationPromptStartsInThinking
                ? rawContent
                : rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw MFTokenizerError.invalidChatTemplate("system message must be first")
            }
            s += Self.imStartMark + message.role.rawValue + "\n" + content + Self.imEndMark + "\n"
        }
        s += generationPromptStartsInThinking
            ? Self.thinkingChatMLGenerationSuffix
            : Self.chatMLGenerationSuffix
        return s
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition],
                               reasoningEffort: QwenReasoningEffort? = nil,
                               preserveThinking: Bool = false,
                               addGenerationPrompt: Bool = true) throws -> [Int32] {
        guard acceptsReasoningEffort || reasoningEffort == nil else {
            throw MFTokenizerError.unsupportedForDialect("reasoning_effort is not supported by this model")
        }
        // MiniCPM's template is hand-ported too (its Python-side semantics —
        // loop-scoped `set`, an undefined `has_tool_sep`, Python `repr` of
        // parameter values — are exactly what a second Jinja engine would get
        // subtly wrong); the render is byte-checked against HF fixtures.
        if dialect == .minicpm {
            return encode(try miniCPMRender(messages: messages, tools: tools,
                                            thinking: Self.miniCPMDefaultThinking),
                          addBOS: false)
        }
        guard tokenizer.hasChatTemplate else {
            throw MFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            if let reasoning = message.reasoningContent { value["reasoning_content"] = reasoning }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            // ChatML emits `tool | tojson` and carries schemas through unchanged.
            let parameters = tool.parameters
            return [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: usesSourceQwenTemplate(reasoningEffort: reasoningEffort)
                ? ["enable_thinking": reasoningEffort != .off,
                   "reasoning_effort": (reasoningEffort ?? .xhigh).rawValue,
                   "preserve_thinking": true]
                : usesSourceTemplate(reasoningEffort: reasoningEffort)
                ? ["enable_thinking": reasoningEffort != .off,
                   "preserve_thinking": true]
                : ["enable_thinking": false]
        ).map(Int32.init)
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        let trimmedContent = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        switch dialect {
        case .chatml:
            let content = generationPromptStartsInThinking ? userContent : trimmedContent
            let suffix = generationPromptStartsInThinking
                ? Self.thinkingChatMLGenerationSuffix
                : Self.chatMLGenerationSuffix
            return [endOfTurnID] + encode(
                "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                    + suffix,
                addBOS: false)
        case .minicpm:
            // Same shape as the ChatML thinking families: close the cached
            // assistant turn, open the next user turn, and the generation
            // prompt. Content untrimmed — the template never trims.
            let suffix: String
            switch Self.miniCPMDefaultThinking {
            case .enabled: suffix = "<think>\n"
            case .disabled: suffix = "<think>\n\n</think>\n\n"
            case .unspecified: suffix = ""
            }
            return [endOfTurnID] + encode(
                "\n\(Self.imStartMark)user\n\(userContent)\(Self.imEndMark)\n"
                    + "\(Self.imStartMark)assistant\n" + suffix,
                addBOS: false)
        }
    }

    /// Tool-result KV continuation: no supported dialect can bridge a cached
    /// tool-call turn by token boundary. The ChatML template's `<think>`
    /// stripping depends on each assistant turn's position relative to the
    /// last user query, so a re-rendered prefix is not guaranteed to be a
    /// token prefix of the full render. Callers (PromptPrefixCache) fall back
    /// to prefix matching.
    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition],
        reasoningEffort: QwenReasoningEffort? = nil,
        preserveThinking: Bool = false
    ) throws -> [Int32] {
        throw MFTokenizerError.unsupportedForDialect("tool-result KV continuation")
    }
}

private enum MFTokenizerLoadSource: Hashable {
    case local(String, ModelFamily?)
}

private actor MFTokenizerLoadCoordinator {
    static let shared = MFTokenizerLoadCoordinator()

    private var tasks: [MFTokenizerLoadSource: Task<MFTokenizer, Error>] = [:]

    func load(_ source: MFTokenizerLoadSource) async throws -> MFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> MFTokenizer in
            switch source {
            case .local(let path, let family):
                return try await MFTokenizer.loadUncached(
                    from: URL(fileURLWithPath: path), family: family)
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
