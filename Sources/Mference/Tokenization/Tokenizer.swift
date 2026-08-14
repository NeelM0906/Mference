import Foundation
import Tokenizers

public enum MFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case unsupportedForDialect(String)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        }
    }
}

/// Chat framing dialect, resolved from the loaded tokenizer's special tokens.
///
/// `.deepseek` is detected by the presence of the `<｜User｜>` special token
/// (DeepSeek-V4), `.chatml` by `<|im_end|>` (Qwen-style ChatML); everything
/// else uses the Gemma 4 contract.
public enum ChatDialect: String, Sendable {
    case gemma
    case chatml
    case deepseek
    case inkling
}

/// Tokenizer wrapper for the supported model families (Gemma 4, ChatML/Qwen,
/// and DeepSeek-V4).
///
/// Prefers tokenizer sidecars in a completed `.gturbo/tokenizer/` directory,
/// then falls back to the IT variant's Hugging Face Hub tokenizer cache. Exposes
/// typed accessors for the IDs the generator actually needs (BOS / EOS / pad /
/// end-of-turn) and adapts encode/decode to Int32 to match the buffer types
/// kernels consume.
///
/// Mference owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
public struct MFTokenizer: @unchecked Sendable {
    public static let modelID = "google/gemma-4-26B-A4B-it"
    public static let chatTemplateIdentity = "gemma4-it-text-no-tools-v1"
    public static let toolChatTemplateIdentity = "gemma4-it-tools-jinja-v1"

    public let dialect: ChatDialect
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
    /// For ChatML these alias the `<think>` / `</think>` markers, the dialect's
    /// closest analog of Gemma's hidden-channel delimiters.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// ChatML `<think>` / `</think>` special-token IDs; nil for Gemma.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int

    /// Maple's and Qwen 3.8's pinned prompts open a live `<think>` block. The
    /// initializer resolves this EOS relationship only when the caller
    /// supplies one of those manifest families; direct loads and other
    /// families retain their old behavior.
    public var generationPromptStartsInThinking: Bool {
        dialect == .chatml && eosID == endOfTurnID
    }

    /// BOS actually prepended by `encode(_:addBOS:)`; nil for dialects that
    /// never use a BOS prefix (ChatML).
    private let bosPrefixID: Int32?

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> MFTokenizer {
        try await MFTokenizerLoadCoordinator.shared.load(.pretrained(modelID))
    }

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
        if let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) {
            return try await load(from: folder, family: family)
        }
        if family == .maple { throw MFTokenizerError.missingToolTemplate }
        return try await load()
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

    static func loadUncached(pretrained modelID: String = Self.modelID) async throws -> MFTokenizer {
        let underlying = try await AutoTokenizer.from(pretrained: modelID)
        return try MFTokenizer(tokenizer: underlying)
    }

    static func loadUncached(from folder: URL) async throws -> MFTokenizer {
        try await loadUncached(from: folder, family: nil)
    }

    static func loadUncached(from folder: URL,
                             family: ModelFamily?) async throws -> MFTokenizer {
        let underlying = try await AutoTokenizer.from(modelFolder: folder)
        return try MFTokenizer(tokenizer: underlying, family: family)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    public init(tokenizer: any Tokenizer) throws {
        try self.init(tokenizer: tokenizer, family: nil)
    }

    public init(tokenizer: any Tokenizer, family: ModelFamily?) throws {
        self.tokenizer = tokenizer

        let dialect: ChatDialect =
            if Self.specialTokenID(tokenizer, Self.inklingUserMark) != nil {
                .inkling
            } else if Self.specialTokenID(tokenizer, Self.deepseekUserMark) != nil {
                .deepseek
            } else if Self.specialTokenID(tokenizer, Self.imEndMark) != nil {
                .chatml
            } else {
                .gemma
            }
        guard family != .maple || dialect == .chatml else {
            throw MFTokenizerError.unsupportedForDialect("Maple requires ChatML framing")
        }
        let resolved = switch dialect {
        case .gemma: try Self.resolveGemmaTokens(tokenizer)
        case .chatml: try Self.resolveChatMLTokens(tokenizer, family: family)
        case .deepseek: try Self.resolveDeepseekTokens(tokenizer)
        case .inkling: try Self.resolveInklingTokens(tokenizer)
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

    private static func resolveGemmaTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        guard let bos = tokenizer.bosTokenId else {
            throw MFTokenizerError.missingSpecialToken("<bos>")
        }
        guard let eos = tokenizer.eosTokenId else {
            throw MFTokenizerError.missingSpecialToken("<eos>")
        }
        guard let pad = tokenizer.convertTokenToId("<pad>") else {
            throw MFTokenizerError.missingSpecialToken("<pad>")
        }
        guard let eot = tokenizer.convertTokenToId("<turn|>") else {
            throw MFTokenizerError.missingSpecialToken("<turn|>")
        }
        guard let toolResponse = tokenizer.convertTokenToId("<|tool_response>") else {
            throw MFTokenizerError.missingSpecialToken("<|tool_response>")
        }
        guard let toolCallStart = tokenizer.convertTokenToId("<|tool_call>"),
              let toolCallEnd = tokenizer.convertTokenToId("<tool_call|>"),
              let toolResponseEnd = tokenizer.convertTokenToId("<tool_response|>"),
              let channelStart = tokenizer.convertTokenToId("<|channel>"),
              let channelEnd = tokenizer.convertTokenToId("<channel|>") else {
            throw MFTokenizerError.missingSpecialToken("Gemma tool/channel markers")
        }
        return ResolvedSpecialTokens(
            bosID: Int32(bos),
            bosPrefixID: Int32(bos),
            eosID: Int32(eos),
            padID: Int32(pad),
            endOfTurnID: Int32(eot),
            toolCallStartID: Int32(toolCallStart),
            toolCallEndID: Int32(toolCallEnd),
            toolResponseID: Int32(toolResponse),
            toolResponseEndID: Int32(toolResponseEnd),
            channelStartID: Int32(channelStart),
            channelEndID: Int32(channelEnd),
            thinkStartID: nil,
            thinkEndID: nil,
            stopTokenIDs: [Int32(eos), Int32(eot), Int32(toolResponse)],
            vocabSize: 262_144)
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

    private static func resolveDeepseekTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw MFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        let bos = try id(Self.deepseekBOSMark)
        let eos = try id(Self.deepseekEOSMark)
        // The turn markers are required even though no stored property holds
        // them; template rendering relies on the tokenizer recognizing their text.
        _ = try id(Self.deepseekUserMark)
        _ = try id(Self.deepseekAssistantMark)
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: bos,
            bosPrefixID: bos,
            eosID: eos,
            padID: eos,
            endOfTurnID: eos,
            // DSML tool-call framing and `<tool_result>` wrappers are plain
            // text in this dialect, not special tokens; the streaming decoder
            // scans the delta text instead of matching these IDs.
            toolCallStartID: noSuchTokenID,
            toolCallEndID: noSuchTokenID,
            toolResponseID: noSuchTokenID,
            toolResponseEndID: noSuchTokenID,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [eos],
            // The model's padded embedding/lm_head row count — logits buffers
            // use this, mirroring the other dialects.
            vocabSize: 129_280)
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends `<bos>`.
    ///
    /// The library's `addSpecialTokens: true` flag is a no-op for the Gemma 4 IT
    /// tokenizer (its config has `add_bos_token = false`; BOS is expected to come
    /// from the chat template). We prepend manually so the kernel-facing API stays
    /// the same regardless of upstream defaults. ChatML has no BOS, so `addBOS`
    /// is a no-op for that dialect.
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

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }
    }

    /// Text-only, no-tool rendering of the pinned checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    private static let turnOpen    = "<|turn>"
    private static let turnClose   = "<turn|>"
    private static let bosMark     = "<bos>"
    private static let imStartMark = "<|im_start|>"
    private static let imEndMark   = "<|im_end|>"
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch.
    private static let chatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    private static let thinkingChatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n"
    /// DeepSeek-V4 special-token text; note the fullwidth vertical bars
    /// (U+FF5C) and the U+2581 fillers in the sentence markers.
    private static let deepseekBOSMark       = "<｜begin▁of▁sentence｜>"
    private static let deepseekEOSMark       = "<｜end▁of▁sentence｜>"
    private static let deepseekUserMark      = "<｜User｜>"
    private static let deepseekAssistantMark = "<｜Assistant｜>"
    /// Generation prompt with thinking disabled: chat mode closes the think
    /// block immediately, so decoding starts after `</think>`.
    private static let deepseekGenerationSuffix = "<｜Assistant｜></think>"
    /// The assistant branch's own think-close (the shipped Jinja emits one
    /// from the user branch AND one from the assistant branch).
    private static let deepseekThinkCloseMark = "</think>"

    // Inkling message framing (see the checkpoint's chat_template.jinja):
    // each turn is `<role token><|content_text|>CONTENT<|end_message|>`, an
    // assistant turn additionally closes with `<|content_model_end_sampling|>`
    // (the sampling stop), a thinking-effort system line precedes the first
    // non-system message, and the generation prompt is a bare
    // `<|message_model|>`. No BOS anywhere (the reference encodes prompts
    // with add_special_tokens=False).
    private static let inklingUserMark    = "<|message_user|>"
    private static let inklingModelMark   = "<|message_model|>"
    private static let inklingSystemMark  = "<|message_system|>"
    private static let inklingContentText = "<|content_text|>"
    private static let inklingEndMessage  = "<|end_message|>"
    private static let inklingEndSampling = "<|content_model_end_sampling|>"
    /// v1 pins reasoning effort to 0 ("none"): the decode path has no
    /// thinking-block post-processing for this dialect yet, and 0 is a
    /// first-class template value.
    private static let inklingEffortLine =
        inklingSystemMark + inklingContentText
        + "Thinking effort level: 0" + inklingEndMessage

    private static func resolveInklingTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw MFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        let bos = try id("<|begin_of_text|>")
        let eos = try id(Self.inklingEndSampling)
        let thinkStart = try id("<|content_thinking|>")
        let endMessage = try id(Self.inklingEndMessage)
        _ = try id(Self.inklingUserMark)
        _ = try id(Self.inklingModelMark)
        return ResolvedSpecialTokens(
            bosID: bos,
            // The reference never prepends BOS; encode() stays bare.
            bosPrefixID: nil,
            eosID: eos,
            padID: eos,
            endOfTurnID: eos,
            toolCallStartID: noSuchTokenID,
            toolCallEndID: noSuchTokenID,
            toolResponseID: noSuchTokenID,
            toolResponseEndID: noSuchTokenID,
            channelStartID: thinkStart,
            channelEndID: endMessage,
            thinkStartID: thinkStart,
            thinkEndID: endMessage,
            stopTokenIDs: [eos],
            vocabSize: 201_024)
    }

    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        switch dialect {
        case .gemma: return try gemmaChatTemplate(messages)
        case .chatml: return try chatMLChatTemplate(messages)
        case .deepseek: return try deepseekChatTemplate(messages)
        case .inkling: return try inklingChatTemplate(messages)
        }
    }

    /// Text-only, no-tool rendering byte-matched to the shipped Jinja:
    /// content is never trimmed, the effort line precedes the first
    /// non-system message (or closes the render when every message is
    /// system), and the generation prompt is always appended.
    private func inklingChatTemplate(_ messages: [Message]) throws -> String {
        var s = ""
        var effortEmitted = false
        for (index, message) in messages.enumerated() {
            guard let content = message.content else {
                throw MFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            if message.role == .system && index != 0 {
                throw MFTokenizerError.invalidChatTemplate("system message must be first")
            }
            if !effortEmitted && message.role != .system {
                s += Self.inklingEffortLine
                effortEmitted = true
            }
            switch message.role {
            case .system:
                s += Self.inklingSystemMark + Self.inklingContentText
                    + content + Self.inklingEndMessage
            case .user, .developer:
                s += Self.inklingUserMark + Self.inklingContentText
                    + content + Self.inklingEndMessage
            case .assistant:
                s += Self.inklingModelMark + Self.inklingContentText
                    + content + Self.inklingEndMessage + Self.inklingEndSampling
            case .tool:
                throw MFTokenizerError.invalidChatTemplate(
                    "inkling tool turns are not supported by the text-only encoder")
            }
        }
        if !effortEmitted { s += Self.inklingEffortLine }
        s += Self.inklingModelMark
        return s
    }

    private func gemmaChatTemplate(_ messages: [Message]) throws -> String {
        var s = Self.bosMark
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw MFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw MFTokenizerError.invalidChatTemplate("system message must be first")
            }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            s += Self.turnOpen + role + "\n" + content + Self.turnClose + "\n"
        }
        s += Self.turnOpen + "model\n<|channel>thought\n<channel|>"
        return s
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

    /// Text-only, no-tool rendering of the DeepSeek-V4 non-thinking ("chat"
    /// mode) encoding, byte-matched to the checkpoint's shipped
    /// `chat_template.jinja`: a system message renders bare, EVERY user turn
    /// is followed by `<｜Assistant｜></think>`, every assistant turn opens
    /// with its own `</think>` (so a user→assistant pair carries
    /// `</think></think>` between marker and content, exactly as the Jinja
    /// renders it) and closes with EOS, content is never trimmed, and the
    /// generation prompt is appended only when the last message is not a
    /// user turn (the user branch already ends in the prompt).
    private func deepseekChatTemplate(_ messages: [Message]) throws -> String {
        var s = Self.deepseekBOSMark
        for (index, message) in messages.enumerated() {
            guard let content = message.content else {
                throw MFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            if message.role == .system && index != 0 {
                throw MFTokenizerError.invalidChatTemplate("system message must be first")
            }
            switch message.role {
            case .system:
                s += content
            case .user, .developer:
                // The reference encoder frames `developer` guidance with the
                // same User marker it uses for user turns.
                s += Self.deepseekUserMark + content + Self.deepseekGenerationSuffix
            case .assistant:
                s += Self.deepseekThinkCloseMark + content + Self.deepseekEOSMark
            case .tool:
                throw MFTokenizerError.invalidChatTemplate(
                    "deepseek merges tool results into user turns; use the tool chat encoder")
            }
        }
        if let last = messages.last,
           !(last.role == .user || last.role == .developer) {
            s += Self.deepseekGenerationSuffix
        }
        return s
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition]) throws -> [Int32] {
        // DeepSeek ships no chat_template.jinja; its tool framing is native.
        if dialect == .deepseek {
            return try encodeDeepseekToolChat(messages: messages, tools: tools)
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
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            // Gemma's template reads `value['type'] | upper` for every property,
            // so union / array / type-less schemas must be flattened first.
            // ChatML emits `tool | tojson` and carries them through unchanged.
            let parameters = dialect == .gemma
                ? tool.parameters.gemmaSchemaNormalized()
                : tool.parameters
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
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: ["enable_thinking": false]
        ).map(Int32.init)
    }

    // MARK: - DeepSeek native tool chat

    /// Full DeepSeek-V4 tool-chat render, mirroring the reference encoder's
    /// chat-mode output: tool schemas join the system message as a `## Tools`
    /// section, `tool` results merge into `<｜User｜>` turns as
    /// `<tool_result>` blocks, and historical tool calls render as DSML
    /// `<｜DSML｜tool_calls>` blocks.
    private func encodeDeepseekToolChat(messages: [Message],
                                        tools: [FunctionDefinition]) throws -> [Int32] {
        var s = Self.deepseekBOSMark
        var remaining = messages[...]
        // Tool schemas ride in the system message — always after a blank
        // line, even onto empty content, matching the reference render; a
        // conversation that opens without one synthesizes the empty message.
        var systemText: String?
        if let first = remaining.first, first.role == .system {
            systemText = first.content ?? ""
            remaining = remaining.dropFirst()
        }
        if !tools.isEmpty {
            let section = try Self.deepseekToolsSection(tools)
            systemText = (systemText ?? "") + "\n\n" + section
        }
        if let systemText { s += systemText }

        // The dialect has no standalone tool role: a run of user text and
        // tool results collapses into one `<｜User｜>` turn, its parts joined
        // by blank lines, exactly as the reference merge step does.
        var pendingUserParts: [String] = []
        var lastTurnWasUser = false
        func flushUserTurn() {
            guard !pendingUserParts.isEmpty else { return }
            s += Self.deepseekUserMark + pendingUserParts.joined(separator: "\n\n")
            pendingUserParts = []
            lastTurnWasUser = true
        }
        for message in remaining {
            switch message.role {
            case .system:
                throw MFTokenizerError.invalidChatTemplate("system message must be first")
            case .user:
                pendingUserParts.append(message.content ?? "")
            case .tool:
                pendingUserParts.append("<tool_result>\(message.content ?? "")</tool_result>")
            case .developer:
                // Developer guidance keeps its own User-framed turn upstream;
                // it never merges with adjacent tool results.
                flushUserTurn()
                s += Self.deepseekUserMark + (message.content ?? "")
                lastTurnWasUser = true
            case .assistant:
                flushUserTurn()
                // The reference closes the preceding user turn with the
                // assistant transition before the reply's content.
                if lastTurnWasUser { s += Self.deepseekGenerationSuffix }
                var turn = message.content ?? ""
                if !message.toolCalls.isEmpty {
                    let invokes = try message.toolCalls
                        .map(Self.deepseekInvoke)
                        .joined(separator: "\n")
                    turn += "\n\n" + DeepseekToolCallParser.toolCallsOpenMark + "\n"
                        + invokes + "\n" + DeepseekToolCallParser.toolCallsCloseMark
                }
                s += turn + Self.deepseekEOSMark
                lastTurnWasUser = false
            }
        }
        flushUserTurn()
        s += Self.deepseekGenerationSuffix
        return encode(s, addBOS: false)
    }

    /// One historical tool call as a DSML invoke block. String arguments pass
    /// through raw with `string="true"`; everything else serializes to JSON
    /// with `string="false"`, matching the reference encoder. Keys render in
    /// sorted order to keep the prompt deterministic.
    ///
    /// DSML has no escape syntax, so a name or value containing the
    /// `｜DSML｜` mark cannot be framed unambiguously (the parser reads a
    /// value up to the first close tag) — such calls are rejected rather
    /// than silently corrupting the next turn's prompt framing.
    private static func deepseekInvoke(_ call: HistoricalToolCall) throws -> String {
        guard case .object(let arguments) = call.arguments else {
            throw MFTokenizerError.invalidChatTemplate(
                "historical tool arguments must be a JSON object")
        }
        let dsml = DeepseekToolCallParser.dsmlMark
        func guardFramable(_ text: String, what: String) throws {
            guard !text.contains(dsml) else {
                throw MFTokenizerError.invalidChatTemplate(
                    "historical tool call \(what) contains the DSML marker and cannot be re-rendered unambiguously")
            }
        }
        try guardFramable(call.name, what: "name")
        let parameters = try arguments.keys.sorted().map { key -> String in
            try guardFramable(key, what: "parameter name")
            let value = arguments[key]!
            if case .string(let raw) = value {
                try guardFramable(raw, what: "argument \"\(key)\"")
                return "<\(dsml)parameter name=\"\(key)\" string=\"true\">\(raw)</\(dsml)parameter>"
            }
            let encoded = try value.encoded()
            try guardFramable(encoded, what: "argument \"\(key)\"")
            return "<\(dsml)parameter name=\"\(key)\" string=\"false\">\(encoded)</\(dsml)parameter>"
        }.joined(separator: "\n")
        return "<\(dsml)invoke name=\"\(call.name)\">\n\(parameters)\n</\(dsml)invoke>"
    }

    /// The `## Tools` system-prompt section from the reference encoder,
    /// carrying the DSML invoke syntax and the JSON tool schemas.
    private static func deepseekToolsSection(_ tools: [FunctionDefinition]) throws -> String {
        let dsml = DeepseekToolCallParser.dsmlMark
        // Fixed key order mirrors the OpenAI-format function objects the
        // reference serializes; schema keys sort for determinism.
        let schemas = try tools.map { tool -> String in
            let name = try JSONValue.string(tool.name).encoded(sortedKeys: false)
            let description = try JSONValue.string(tool.description).encoded(sortedKeys: false)
            let parameters = try tool.parameters.encoded()
            return "{\"name\":\(name),\"description\":\(description),\"parameters\":\(parameters)}"
        }.joined(separator: "\n")
        return """
        ## Tools

        You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<\(dsml)tool_calls>" block like the following:

        <\(dsml)tool_calls>
        <\(dsml)invoke name="$TOOL_NAME">
        <\(dsml)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(dsml)parameter>
        ...
        </\(dsml)invoke>
        <\(dsml)invoke name="$TOOL_NAME2">
        ...
        </\(dsml)invoke>
        </\(dsml)tool_calls>

        String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

        If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

        Otherwise, output directly after </think> with tool calls or final response.

        ### Available Tool Schemas

        \(schemas)

        You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.

        """
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        let trimmedContent = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        switch dialect {
        case .gemma:
            return [endOfTurnID] + encode(
                "\n\(Self.turnOpen)user\n\(trimmedContent)\(Self.turnClose)\n"
                    + "\(Self.turnOpen)model\n<|channel>thought\n<channel|>",
                addBOS: false)
        case .chatml:
            let content = generationPromptStartsInThinking ? userContent : trimmedContent
            let suffix = generationPromptStartsInThinking
                ? Self.thinkingChatMLGenerationSuffix
                : Self.chatMLGenerationSuffix
            return [endOfTurnID] + encode(
                "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                    + suffix,
                addBOS: false)
        case .deepseek:
            // The cached assistant turn stopped just before its EOS; the
            // bridge supplies it, then opens the next user turn. Content is
            // NOT trimmed — the full-render template never trims, and the
            // continuation must produce the same bytes a fresh render would.
            return [endOfTurnID] + encode(
                Self.deepseekUserMark + userContent + Self.deepseekGenerationSuffix,
                addBOS: false)
        case .inkling:
            // The cached assistant turn stopped just before
            // `<|content_model_end_sampling|>` (== endOfTurnID); supply it,
            // then the next user turn and the generation prompt. Untrimmed,
            // matching the full render.
            return [endOfTurnID] + encode(
                Self.inklingUserMark + Self.inklingContentText + userContent
                    + Self.inklingEndMessage + Self.inklingModelMark,
                addBOS: false)
        }
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
        // The ChatML template's `<think>` stripping depends on each assistant
        // turn's position relative to the last user query, so a re-rendered
        // prefix is not guaranteed to be a token prefix of the full render.
        // DeepSeek merges tool results into user turns rather than keying on
        // special tokens, so the boundary search below has nothing to anchor
        // on. Callers (ServerPromptCache) fall back to prefix matching.
        guard dialect == .gemma else {
            throw MFTokenizerError.unsupportedForDialect("tool-result KV continuation")
        }
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant],
            tools: tools)
        let full = try encodeToolChat(messages: incomingMessages, tools: tools)
        let callCount = assistant.toolCalls.count
        let starts = prefix.indices.filter { prefix[$0] == toolCallStartID }
        guard callCount > 0, starts.count >= callCount,
              let callEnd = prefix.lastIndex(of: toolCallEndID) else {
            throw MFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is missing")
        }
        let callStart = starts[starts.count - callCount]
        let callSequence = Array(prefix[callStart...callEnd])
        let matches = full.subsequenceStartIndices(matching: callSequence)
        guard matches.count == 1 else {
            throw MFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is ambiguous")
        }
        let suffixStart = matches[0] + callSequence.count
        let suffix = Array(full[suffixStart...])
        guard suffix.first == toolResponseID else {
            throw MFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return suffix
    }
}

private extension Array where Element: Equatable {
    func subsequenceStartIndices(matching needle: [Element]) -> [Int] {
        guard !needle.isEmpty, needle.count <= count else { return [] }
        return indices.dropLast(needle.count - 1).filter { start in
            self[start..<(start + needle.count)].elementsEqual(needle)
        }
    }
}

private enum MFTokenizerLoadSource: Hashable {
    case pretrained(String)
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
            case .pretrained(let modelID):
                return try await MFTokenizer.loadUncached(pretrained: modelID)
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
