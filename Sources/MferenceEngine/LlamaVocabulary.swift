import Foundation
import LlamaCpp
import Tokenizers

// The tokenizer of a GGUF model, as llama.cpp reads it from the file, behind
// the swift-transformers `Tokenizer` protocol. `MFTokenizer` is built on that
// protocol, so the runtime's chat dialect detection, special-token resolution,
// structured decoder (reasoning split, tool calls) and stop handling work on a
// GGUF model unchanged, while every token id comes from llama.cpp itself: the
// same tokenizer the upstream reference tools use.

/// Owns a loaded `llama_model`. Freed once, by `free()`; nothing may use the
/// vocabulary afterwards (`EngineHost` calls nothing on a model after
/// `unload`).
final class LlamaModelHandle: @unchecked Sendable {
    private(set) var pointer: OpaquePointer?

    init(_ pointer: OpaquePointer) {
        self.pointer = pointer
    }

    func free() {
        if let pointer { llama_model_free(pointer) }
        pointer = nil
    }

    deinit { free() }
}

/// Owns a `llama_context`. It holds the model it was created from, so the
/// context is always freed first, whether by `free()` (`unload`) or when the
/// last reference goes.
final class LlamaContextHandle: @unchecked Sendable {
    private(set) var pointer: OpaquePointer?
    private let model: LlamaModelHandle
    /// The ggml CPU threadpool attached to the context, freed after it.
    private var threadpool: OpaquePointer?

    init(_ pointer: OpaquePointer, model: LlamaModelHandle, threadpool: OpaquePointer? = nil) {
        self.pointer = pointer
        self.model = model
        self.threadpool = threadpool
    }

    func free() {
        if let pointer { llama_free(pointer) }
        pointer = nil
        if let threadpool { ggml_threadpool_free(threadpool) }
        threadpool = nil
    }

    deinit { free() }
}

/// Owns an `mference_mtp` (the MTP draft context and llama.cpp's speculative
/// state). It holds the target context, so it is always freed before it.
final class LlamaDrafter: @unchecked Sendable {
    private(set) var pointer: OpaquePointer?
    /// Tokens one step can return: the draft plus the target's own token.
    let stepCapacity: Int
    private let context: LlamaContextHandle

    init(_ pointer: OpaquePointer, draftLimit: Int, context: LlamaContextHandle) {
        self.pointer = pointer
        self.stepCapacity = draftLimit + 1
        self.context = context
    }

    func free() {
        if let pointer { mference_mtp_free(pointer) }
        pointer = nil
    }

    deinit { free() }
}

/// A GGUF vocabulary as a swift-transformers `Tokenizer`.
///
/// `encode` parses special-token text (`<|im_start|>` and the like) into its
/// single id, as llama.cpp's own tools do for a rendered chat prompt, and
/// `decode` concatenates each token's bytes before converting to text, so a
/// character split across tokens comes out whole. Chat templates are not
/// applied here: `LlamaChatTemplate` renders them.
final class LlamaVocabulary: Tokenizer, @unchecked Sendable {
    private let model: LlamaModelHandle
    let vocab: OpaquePointer
    let tokenCount: Int
    private let idsByText: [String: Int]
    private let bosID: Int32?
    private let eosID: Int32?

    init(model: LlamaModelHandle) throws {
        guard let modelPointer = model.pointer,
              let vocab = llama_model_get_vocab(modelPointer) else {
            throw LlamaBackendError.vocabularyUnavailable
        }
        self.model = model
        self.vocab = vocab
        tokenCount = Int(llama_vocab_n_tokens(vocab))
        var idsByText: [String: Int] = [:]
        idsByText.reserveCapacity(tokenCount)
        for id in 0..<tokenCount {
            guard let text = llama_vocab_get_text(vocab, llama_token(id)) else { continue }
            // First id wins, as in a tokenizer.json vocabulary.
            let key = String(cString: text)
            if idsByText[key] == nil { idsByText[key] = id }
        }
        self.idsByText = idsByText
        let bos = llama_vocab_bos(vocab)
        let eos = llama_vocab_eos(vocab)
        bosID = bos >= 0 ? bos : nil
        eosID = eos >= 0 ? eos : nil
    }

    // MARK: Token ids

    /// Tokenizes `text` as llama.cpp's tools tokenize a prompt
    /// (`common_tokenize(..., add_special, parse_special = true)`).
    func tokenIDs(_ text: String, addSpecial: Bool) -> [Int32] {
        let utf8 = Array(text.utf8)
        guard !utf8.isEmpty || addSpecial else { return [] }
        var capacity = utf8.count + 2
        while true {
            var tokens = [llama_token](repeating: 0, count: capacity)
            let count = utf8.withUnsafeBufferPointer { bytes in
                bytes.withMemoryRebound(to: CChar.self) { chars in
                    llama_tokenize(vocab, chars.baseAddress, Int32(chars.count),
                                   &tokens, Int32(capacity), addSpecial, true)
                }
            }
            if count >= 0 { return Array(tokens.prefix(Int(count))) }
            capacity = Int(-count)
        }
    }

    /// The bytes a token contributes to text. Control tokens yield nothing
    /// unless `special` is set; user-defined tokens always yield their text.
    func pieceBytes(_ token: Int32, special: Bool) -> [UInt8] {
        var buffer = [CChar](repeating: 0, count: 64)
        var count = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, special)
        if count < 0 {
            buffer = [CChar](repeating: 0, count: Int(-count))
            count = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, special)
        }
        guard count > 0 else { return [] }
        return buffer.prefix(Int(count)).map { UInt8(bitPattern: $0) }
    }

    /// Whether tokenization prepends BOS / appends EOS (`add_bos_token`,
    /// `add_eos_token` in the GGUF).
    var addsBOS: Bool { llama_vocab_get_add_bos(vocab) }
    var addsEOS: Bool { llama_vocab_get_add_eos(vocab) }

    /// Tokens the GGUF says never to sample (`tokenizer.ggml.suppress_tokens`).
    var suppressedTokens: [Int32] {
        var count: Int32 = 0
        guard let tokens = llama_vocab_get_suppress_tokens(vocab, &count), count > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: tokens, count: Int(count)))
    }

    /// End-of-generation per llama.cpp (`<|im_end|>`, `<|endoftext|>`, ...).
    func isEndOfGeneration(_ token: Int32) -> Bool {
        llama_vocab_is_eog(vocab, token)
    }

    // MARK: Tokenizer

    func tokenize(text: String) -> [String] {
        tokenIDs(text, addSpecial: false).map { convertIdToToken(Int($0)) ?? "" }
    }

    func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        tokenIDs(text, addSpecial: addSpecialTokens).map(Int.init)
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        var bytes: [UInt8] = []
        for token in tokens {
            bytes += pieceBytes(Int32(truncatingIfNeeded: token), special: !skipSpecialTokens)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        idsByText[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0, id < tokenCount, let text = llama_vocab_get_text(vocab, llama_token(id)) else {
            return nil
        }
        return String(cString: text)
    }

    var bosToken: String? { bosID.flatMap { convertIdToToken(Int($0)) } }
    var bosTokenId: Int? { bosID.map(Int.init) }
    var eosToken: String? { eosID.flatMap { convertIdToToken(Int($0)) } }
    var eosTokenId: Int? { eosID.map(Int.init) }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    // `MFTokenizer` renders through `LlamaChatTemplate`, never through these.
    var hasChatTemplate: Bool { false }

    func applyChatTemplate(messages: [Message]) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?,
                           addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?,
                           tools: [ToolSpec]?) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?,
                           addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?,
                           tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}

/// Turns a stream of token bytes into text without splitting a UTF-8
/// character: an incomplete trailing sequence is held until its remaining
/// bytes arrive (or until `finish`).
struct LlamaUTF8Stream {
    private var pending: [UInt8] = []

    mutating func push(_ bytes: [UInt8]) -> String {
        pending += bytes
        let complete = Self.completePrefixLength(pending)
        guard complete > 0 else { return "" }
        let text = String(decoding: pending[..<complete], as: UTF8.self)
        pending.removeFirst(complete)
        return text
    }

    mutating func finish() -> String {
        defer { pending.removeAll() }
        return String(decoding: pending, as: UTF8.self)
    }

    /// Length of the longest prefix that does not end inside a multi-byte
    /// sequence. Invalid bytes are not held; `String(decoding:)` replaces
    /// them.
    static func completePrefixLength(_ bytes: [UInt8]) -> Int {
        let count = bytes.count
        // A sequence is at most 4 bytes, so only the last 3 can be an open one.
        var index = count - 1
        let lowest = max(0, count - 3)
        while index >= lowest {
            let byte = bytes[index]
            if byte & 0xC0 == 0x80 {
                index -= 1
                continue
            }
            let needed: Int
            switch byte {
            case 0xC2...0xDF: needed = 2
            case 0xE0...0xEF: needed = 3
            case 0xF0...0xF4: needed = 4
            default: needed = 1
            }
            return count - index < needed ? index : count
        }
        return count
    }
}
