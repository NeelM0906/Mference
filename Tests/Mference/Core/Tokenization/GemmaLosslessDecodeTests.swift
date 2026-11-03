import Foundation
import Hub
import Testing
@testable import Mference

/// The lossless Gemma decode (`GemmaDecoding`) on the real Gemma 4 tokenizer.
/// `MFTokenizer.load()` reads the Hub copy of the same `tokenizer.json` the
/// installers pin; the first run downloads it (~32 MB) to
/// `~/.cache/huggingface/`.
@Suite("Gemma lossless decode")
struct GemmaLosslessDecodeTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load()
    }

    @Test("The Gemma tokenizer takes the lossless path")
    func losslessPathTaken() {
        #expect(tok.dialect == .gemma)
        #expect(tok.losslessGemmaSpecialTokenIDs != nil)
    }

    @Test("Special-token set matches the library's skip filter")
    func specialTokenSetMatchesLibrary() throws {
        let specials = try #require(tok.losslessGemmaSpecialTokenIDs)
        var probes = randomIDs(count: 64, seed: 0xDEAD_BEEF_CAFE_F00D)
        probes += [tok.bosID, tok.eosID, tok.padID, tok.endOfTurnID,
                   tok.toolCallStartID, tok.toolCallEndID, tok.toolResponseID,
                   tok.toolResponseEndID, tok.channelStartID, tok.channelEndID]
        probes += Array(specials)
        for id in probes {
            // A special token decodes to nothing when skipped and to its
            // literal text when kept.
            let skipped = tok.tokenizer.decode(tokens: [Int(id)], skipSpecialTokens: true)
            let kept = tok.tokenizer.decode(tokens: [Int(id)], skipSpecialTokens: false)
            let librarySpecial = skipped.isEmpty && !kept.isEmpty
            #expect(specials.contains(id) == librarySpecial,
                    "id \(id) special-set membership diverged from library semantics")
        }
    }

    @Test("Library fuses a split codepoint across byte-fallback tokens")
    func libraryFusesSplitCodepoint() throws {
        // The library stays the oracle for the pinned chain's ByteFallback and
        // Fuse steps. The run needs a trailing regular token: the library drops
        // byte-fallback tokens that end a sequence.
        let ids = try ["<0xC3>", "<0xA9>", "the"].map { Int(try tokenID($0)) }
        #expect(tok.tokenizer.decode(tokens: ids, skipSpecialTokens: false) == "éthe")
    }

    // MARK: - Round trips

    @Test("Decode and streaming keep the text the cleanup pass rewrote", arguments: [
        "he said ' ok ' now",
        "step 1 . done",
        "Bonjour !",
        "x , y",
        "a    . b",
        "wait ! really ?",
        "it 's ok",
        "do n't stop",
        "we 've seen",
        "they 're here",
        "I 'm sure",
        "x = ' ' # a space",
    ])
    func cleanUpPatternsRoundTrip(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
        #expect(stream(ids) == text)
        // The library's decode is what rewrote them.
        #expect(tok.tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: true) != text)
    }

    @Test("Whitespace run before punctuation survives streaming")
    func streamingWhitespaceThenPunctuation() throws {
        // Encoder-unreachable but generator-reachable: the normalizer maps " "
        // to "▁", so no encode() call produces a bare whitespace run followed
        // by punctuation, but a generating model emits it in indented text and
        // code. The cleanup pass rewrote " ." to "." across that boundary.
        for (run, mark) in [("▁▁▁▁", "."), ("▁▁", ","), ("▁▁▁▁", "?"), ("▁▁", "!")] {
            let ids = [try tokenID(run), try tokenID(mark)]
            let expected = String(repeating: " ", count: run.count) + mark
            #expect(stream(ids) == expected)
            #expect(tok.decode(ids) == expected)
        }
    }

    @Test("Streaming reassembles graphemes extended by later scalars", arguments: [
        "ห้าม",
        "e\u{301}",
        "ا\u{64E}",
        "ש\u{5B8}",
        "क्ष",
        "\u{1100}\u{1161}",
        "\u{263A}\u{FE0F}",
        "\u{1F44D}\u{1F3FD}",
        "1\u{FE0F}\u{20E3}",
        "\u{1F1EC}\u{1F1E7}",
        "\u{1F469}\u{200D}\u{1F4BB}",
    ])
    func streamingExtendedGraphemes(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(stream(ids) == text)
        #expect(tok.decode(ids) == text)
    }

    // MARK: - Agreement with the library and with streaming

    @Test("Decode differs from the library only by its cleanup pass")
    func decodeDiffersFromLibraryOnlyByCleanup() throws {
        // swift-transformers as an independent oracle: our decode, run through
        // the cleanup pass we dropped, must reproduce the library byte for
        // byte. That covers special-token filtering, byte-fallback grouping
        // and the "▁" mapping against code we do not own. The fixed seed puts
        // no two byte-fallback tokens next to each other, so the deliberate
        // divergences on multi-byte runs (reference semantics, trailing runs
        // kept) do not reach this comparison; the byte tests below cover them.
        var ids = randomIDs(count: 256, seed: 0x2545_F491_4F6C_DD1D)
        ids += [tok.bosID, tok.eosID, tok.endOfTurnID, tok.channelStartID, tok.padID]
        // The library drops byte-fallback tokens that end a sequence.
        ids.append(try tokenID("the"))
        for skipSpecialTokens in [true, false] {
            let mine = tok.decode(ids, skipSpecialTokens: skipSpecialTokens)
            let library = tok.tokenizer.decode(tokens: ids.map(Int.init),
                                               skipSpecialTokens: skipSpecialTokens)
            #expect(Self.huggingFaceCleanUp(mine) == library,
                    "diverged beyond the cleanup pass (skipSpecialTokens: \(skipSpecialTokens))")
        }
    }

    @Test("Streaming matches batch decode for arbitrary token streams")
    func streamingMatchesBatchForArbitraryIDs() {
        // Encode() only produces sequences the encoder emits; a generating
        // model is under no such constraint, so drive it with raw IDs.
        for round in 0..<16 {
            let ids = randomIDs(count: 256, seed: 0x9E37_79B9_7F4A_7C15 &+ UInt64(round))
            #expect(stream(ids) == tok.decode(ids), "round \(round) diverged from batch decode")
        }
    }

    // MARK: - Byte-fallback runs

    @Test("Byte run fuses across a skipped special token")
    func byteRunFusesAcrossSkippedSpecial() throws {
        // The library filters skipped specials before its decoder chain, so a
        // run fuses across one. Kept, the special closes the run instead and
        // leaves two single-byte runs that are invalid on their own.
        let ids: [Int32] = [try tokenID("<0xC3>"), tok.eosID, try tokenID("<0xA9>"), try tokenID("the")]
        #expect(tok.decode(ids, skipSpecialTokens: true) == "éthe")
        #expect(stream(ids) == "éthe")
        #expect(tok.decode(ids, skipSpecialTokens: false) == "\u{FFFD}<eos>\u{FFFD}the")
    }

    @Test("Invalid byte runs stream replacement characters immediately")
    func invalidByteRunsStreamImmediately() throws {
        var detok = MFDetokenizer(tokenizer: tok)
        let stray = try tokenID("<0x80>")
        for _ in 0..<64 {
            #expect(detok.push(stray) == "\u{FFFD}")
        }
        #expect(detok.flush() == "")

        detok = MFDetokenizer(tokenizer: tok)
        #expect(detok.push(try tokenID("<0xC3>")) == "")
        #expect(detok.push(try tokenID("<0xFF>")) == "\u{FFFD}\u{FFFD}")
    }

    @Test("A valid byte run commits with the token that closes it")
    func validByteRunCommitsAtBoundary() throws {
        var detok = MFDetokenizer(tokenizer: tok)
        for token in ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"] {
            #expect(detok.push(try tokenID(token)) == "")
        }
        #expect(detok.push(try tokenID("the")) == "😀the")
        #expect(detok.flush() == "")
    }

    @Test("Byte-token soup matches batch decode and reference run semantics")
    func byteSoupMatchesReference() throws {
        // Mostly byte tokens with occasional regular tokens, checked against a
        // direct reimplementation of the reference run semantics (valid UTF-8,
        // else one U+FFFD per byte).
        let byteIDs: [Int32] = try (0...255).map { try tokenID(String(format: "<0x%02X>", $0)) }
        let theID = try tokenID("the")
        var state: UInt64 = 0xA076_1D64_78BD_642F
        func next() -> UInt64 {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }
        for round in 0..<16 {
            var ids: [Int32] = []
            var expected = ""
            var run: [UInt8] = []
            func closeRun() {
                guard !run.isEmpty else { return }
                expected += String(bytes: run, encoding: .utf8)
                    ?? String(repeating: "\u{FFFD}", count: run.count)
                run.removeAll()
            }
            for _ in 0..<256 {
                let r = next()
                if r % 8 == 0 {
                    closeRun()
                    ids.append(theID)
                    expected += "the"
                } else {
                    let byte = UInt8(truncatingIfNeeded: r >> 8)
                    ids.append(byteIDs[Int(byte)])
                    run.append(byte)
                }
            }
            closeRun()
            #expect(tok.decode(ids) == expected, "round \(round) batch diverged")
            #expect(stream(ids) == expected, "round \(round) streaming diverged")
        }
    }

    @Test("An unknown ID contributes nothing and leaves the run open")
    func unknownIDIsDropped() throws {
        // The library's decode drops IDs it cannot resolve.
        let unknown = Int32(tok.vocabSize + 1_000)
        #expect(tok.tokenizer.convertIdToToken(Int(unknown)) == nil)
        let ids: [Int32] = [try tokenID("<0xC3>"), unknown, try tokenID("<0xA9>"), try tokenID("the")]
        #expect(stream(ids) == "éthe")
        #expect(tok.decode(ids) == "éthe")
    }

    // MARK: - Which loads take the lossless path

    @Test("A foreign decoder declaration falls back to the library decode")
    func foreignDecoderFallsBack() throws {
        let data: Config = [
            "decoder": ["type": "Metaspace", "replacement": "▁", "prependScheme": "first"] as Config,
        ]
        let fallback = try MFTokenizer(tokenizer: tok.tokenizer, family: .gemma4, tokenizerData: data)
        #expect(fallback.dialect == .gemma)
        #expect(fallback.losslessGemmaSpecialTokenIDs == nil)
        let ids = tok.encode("he said ' ok ' now", addBOS: false)
        #expect(fallback.decode(ids) == "he said'ok'now")
    }

    @Test("The public initializer keeps the library decode")
    func publicInitializerKeepsLibraryDecode() throws {
        // No tokenizer.json to check the decoder against.
        let wrapped = try MFTokenizer(tokenizer: tok.tokenizer, family: .gemma4)
        #expect(wrapped.losslessGemmaSpecialTokenIDs == nil)
    }

    // MARK: - Tool calls

    @Test("A tool-call payload decodes with the model's spacing")
    func toolCallPayloadKeepsSpacing() throws {
        let payload = #"call:get_weather{location:<|"|>Paris , France<|"|>,note:<|"|>it 's ok<|"|>}"#
        let ids = tok.encode(payload, addBOS: false)
        #expect(tok.decode(ids, skipSpecialTokens: false) == payload)

        let decoder = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["get_weather"],
                                                 idGenerator: { "call_0" })
        var events: [StructuredAssistantEvent] = []
        events += try decoder.consume(tokenID: tok.toolCallStartID, delta: "")
        for id in ids { events += try decoder.consume(tokenID: id, delta: "") }
        events += try decoder.consume(tokenID: tok.toolCallEndID, delta: "")
        guard events.count == 1, case .toolCall(let call) = events[0] else {
            Issue.record("expected one tool call, got \(events)")
            return
        }
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object([
            "location": .string("Paris , France"),
            "note": .string("it 's ok"),
        ]))
    }

    // MARK: - Helpers

    private func stream(_ ids: [Int32]) -> String {
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids { assembled += detok.push(id) }
        return assembled + detok.flush()
    }

    /// Vocabulary ID of `token`, rejecting the unknown-token substitution.
    private func tokenID(_ token: String) throws -> Int32 {
        let id = try #require(tok.tokenizer.convertTokenToId(token))
        try #require(tok.tokenizer.convertIdToToken(id) == token, "vocabulary is missing \(token)")
        return Int32(id)
    }

    /// Deterministic xorshift64 IDs: token streams the encoder cannot produce
    /// but a generating model can.
    private func randomIDs(count: Int, seed: UInt64) -> [Int32] {
        var state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        return (0..<count).map { _ in
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Int32(state % UInt64(tok.vocabSize))
        }
    }

    /// swift-transformers' `clean_up_tokenization_spaces` pass, reproduced so
    /// the differential test pins our only intended difference to it.
    private static func huggingFaceCleanUp(_ text: String) -> String {
        let replacements: [(String, String)] = [
            (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
            (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"),
        ]
        return replacements.reduce(text) { $0.replacingOccurrences(of: $1.0, with: $1.1) }
    }
}
