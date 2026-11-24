import Foundation
import Testing
@testable import Mference

/// A turn the token cap stopped is kept for prefix reuse with its last
/// sampled token uncommitted. For Inkling 2 the cap bridge (that token, then
/// the text continuation, which begins with `<|content_model_end_sampling|>`)
/// would leave the cut text or thinking block without the `<|end_message|>`
/// a fresh render writes, so the next prompt would depend on whether the
/// cache hit. An Inkling 2 capped turn therefore continues only through the
/// exact rendered prefix. Every other dialect keeps its cap bridge as it was.
@Suite("Capped-turn prefix reuse")
struct CappedTurnReuseTests {
    typealias Message = MFTokenizer.Message

    static func fixture(_ name: String) throws -> URL {
        try #require(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"))
    }

    static let domain = PromptPrefixCacheDomain(
        modelID: "test", sourceSnapshotHash: nil, runtimeProfileHash: "test",
        maximumContext: 4_096, kvStorage: "test", fp16RingEnabled: false, templateSHA256: "")

    static let question = Message(role: .user, content: "Weather in Paris?")
    static let followUp = "And tomorrow?"

    /// The entry a turn leaves when it stops after `generated`: every
    /// generated token but the last is in the KV; the last was sampled and
    /// never fed back.
    private static func entry(prompt: [Int32], generated: [Int32], assistant: Message,
                              reason: StopReason) throws -> PromptPrefixCacheEntry {
        let last: Int32 = try #require(generated.last)
        var kvBacked: [Int32] = prompt
        kvBacked.append(contentsOf: generated.dropLast())
        return PromptPrefixCacheEntry(
            domain: domain,
            inputMessages: [question],
            tools: [],
            assistantTurn: CachedAssistantTurn(message: assistant, rawStopReason: reason),
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [last],
            kvPosition: kvBacked.count)
    }

    /// Each piece encoded on its own: a split the model may sample, which the
    /// tokenizer's encoding of the joined text does not produce, so the
    /// re-rendered history is not an exact extension of the KV.
    private static func pieces(_ tok: MFTokenizer, _ pieces: [String]) -> [Int32] {
        pieces.flatMap { tok.encode($0, addBOS: false) }
    }

    /// Cuts inside "th" and "er", which both fixtures merge when encoding
    /// the joined text ("Need the weather in Paris").
    static let split = ["Need t", "he weat", "her in Par", "is"]

    private static func match(_ entry: PromptPrefixCacheEntry, assistant: Message,
                              tokenizer: MFTokenizer) throws -> (PromptPrefixCacheMatch, rendered: [Int32]) {
        let messages = [question, assistant, Message(role: .user, content: followUp)]
        let rendered = try tokenizer.encodeChat(messages: messages)
        let match = PromptPrefixCache.match(
            entry: entry, domain: domain, turn: PromptPrefixCacheTurn(messages: messages),
            renderedPromptIDs: rendered, tokenizer: tokenizer)
        return (match, rendered)
    }

    private static func inkling2() async throws -> MFTokenizer {
        let tok = try await MFTokenizer.load(from: fixture("Inkling2Tokenizer"))
        #expect(tok.dialect == .inkling2)
        return tok
    }

    static let mdl = "<|message_model|>", txt = "<|content_text|>", thk = "<|content_thinking|>"
    static let end = "<|end_message|>", eos = "<|content_model_end_sampling|>"

    /// Inkling 2, cut by the cap inside a text block and inside a thinking
    /// block: no hit through the bridge. The caller then prefills the fresh
    /// render, the prompt it uses with the cache off.
    @Test(arguments: [false, true])
    func anInkling2TurnCutByTheCapIsNotBridged(inThinking: Bool) async throws {
        let tok = try await Self.inkling2()
        let prompt = try tok.encodeChat(messages: [Self.question])
        let split = Self.split
        let assistant = inThinking
            ? Message(role: .assistant, content: "", reasoningContent: split.joined())
            : Message(role: .assistant, content: split.joined())
        var framed: [String] = [Self.mdl, inThinking ? Self.thk : Self.txt]
        framed.append(contentsOf: split)
        let generated: [Int32] = Self.pieces(tok, framed)
        let entry = try Self.entry(prompt: prompt, generated: generated, assistant: assistant,
                                   reason: .maxTokens)
        let (match, rendered) = try Self.match(entry, assistant: assistant, tokenizer: tok)
        #expect(!rendered.starts(with: entry.kvBackedTokenIDs), "the case must miss the exact prefix")
        #expect(match == .miss)
    }

    /// Inkling 2, cut by the cap where the KV is an exact prefix of the fresh
    /// render: still a hit, and its effective prompt is that render.
    @Test(arguments: [false, true])
    func anInkling2CappedTurnStillContinuesThroughTheExactPrefix(inThinking: Bool) async throws {
        let tok = try await Self.inkling2()
        let prompt = try tok.encodeChat(messages: [Self.question])
        let text = "It is sunny in Paris"
        let assistant = inThinking
            ? Message(role: .assistant, content: "", reasoningContent: text)
            : Message(role: .assistant, content: text)
        let opening: String = inThinking ? Self.thk : Self.txt
        let generated: [Int32] = tok.encode(Self.mdl + opening + text, addBOS: false)
        let entry = try Self.entry(prompt: prompt, generated: generated, assistant: assistant,
                                   reason: .maxTokens)
        let (match, rendered) = try Self.match(entry, assistant: assistant, tokenizer: tok)
        #expect(rendered.starts(with: entry.kvBackedTokenIDs))
        #expect(match == .hit(effectivePromptIDs: rendered, cachedPromptTokens: entry.kvPosition))
    }

    /// Inkling 2, a turn that ended on its own: the text bridge is unchanged.
    @Test func anInkling2FinishedTurnKeepsItsBridge() async throws {
        let tok = try await Self.inkling2()
        let prompt = try tok.encodeChat(messages: [Self.question])
        let split = Self.split
        let assistant = Message(role: .assistant, content: split.joined())
        var framed: [String] = [Self.mdl, Self.txt]
        framed.append(contentsOf: split)
        framed.append(contentsOf: [Self.end, Self.eos])
        let generated: [Int32] = Self.pieces(tok, framed)
        let entry = try Self.entry(prompt: prompt, generated: generated, assistant: assistant,
                                   reason: .endOfTurn)
        let (match, rendered) = try Self.match(entry, assistant: assistant, tokenizer: tok)
        #expect(!rendered.starts(with: entry.kvBackedTokenIDs))
        var expected: [Int32] = entry.kvBackedTokenIDs
        expected.append(contentsOf: tok.encodeTextContinuation(userContent: Self.followUp))
        #expect(match == .hit(effectivePromptIDs: expected, cachedPromptTokens: entry.kvPosition))
    }

    /// Another dialect (ChatML) keeps its cap bridge exactly: the sampled
    /// token, then the text continuation.
    @Test func otherDialectsKeepTheirCapBridge() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixture("ChatMLTokenizer"))
        #expect(tok.dialect == .chatml)
        let prompt = try tok.encodeChat(messages: [Self.question])
        let split = Self.split
        let assistant = Message(role: .assistant, content: split.joined())
        let entry = try Self.entry(prompt: prompt, generated: Self.pieces(tok, split), assistant: assistant,
                                   reason: .maxTokens)
        let (match, rendered) = try Self.match(entry, assistant: assistant, tokenizer: tok)
        #expect(!rendered.starts(with: entry.kvBackedTokenIDs))
        var expected: [Int32] = entry.kvBackedTokenIDs
        expected.append(contentsOf: entry.uncommittedBoundaryTokenIDs)
        expected.append(contentsOf: tok.encodeTextContinuation(userContent: Self.followUp))
        #expect(match == .hit(effectivePromptIDs: expected, cachedPromptTokens: entry.kvPosition))
    }
}
