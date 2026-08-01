import Foundation
import Testing
@testable import Mference

/// Focused coverage for `MFDetokenizer.commitDelta`, which is the only place the
/// streaming loop decides what text to hand to the caller.
///
/// The interesting failure mode is a token boundary that lands *inside* a
/// grapheme cluster: a Thai base consonant in one token and its tone/vowel mark
/// in the next. Grapheme-level prefix checks reject that, so the delta has to be
/// computed over UTF-8 bytes.
@Suite("Detokenizer")
struct DetokenizerTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load()
    }

    // MARK: - Grapheme clusters split across token boundaries

    @Test("Delta survives a token boundary inside a grapheme cluster", arguments: [
        // Thai: base consonant, then mai tho + sara aa.
        ["ห", "ห้าม"],
        // Thai: the cluster boundary lands mid-word in a longer phrase.
        ["การค", "การคัดกรอง"],
        // Devanagari: bare consonant, then the matra that combines onto it.
        ["ह", "हिन्दी"],
        // Emoji ZWJ sequence: one cluster, four scalars plus joiners.
        ["👨", "👨\u{200D}👩\u{200D}👧\u{200D}👦"],
    ])
    func deltaSurvivesSplitCluster(_ decodes: [String]) {
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for decoded in decodes {
            assembled += detok.commitDelta(decoded)
        }
        #expect(assembled == decodes[decodes.count - 1],
                "reassembly mismatch: got '\(assembled)' want '\(decodes[decodes.count - 1])'")
    }

    @Test("Combining mark arriving alone is emitted, not dropped")
    func combiningMarkAloneIsEmitted() {
        var detok = MFDetokenizer(tokenizer: tok)
        #expect(detok.commitDelta("ห") == "ห")
        // The next decode extends the same cluster; the delta is the raw bytes
        // of the combining mark even though it is not a grapheme of its own.
        #expect(detok.commitDelta("ห้") == "\u{0E49}")
    }

    // MARK: - Existing behavior must not change

    @Test("ASCII deltas are unchanged", arguments: [
        ["He", "Hello", "Hello, ", "Hello, world."],
        ["a", "ab", "abc"],
    ])
    func asciiDeltasUnchanged(_ decodes: [String]) {
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for decoded in decodes {
            assembled += detok.commitDelta(decoded)
        }
        #expect(assembled == decodes[decodes.count - 1])
    }

    @Test("Empty delta when the decode did not grow")
    func emptyDeltaOnNoGrowth() {
        var detok = MFDetokenizer(tokenizer: tok)
        #expect(detok.commitDelta("abc") == "abc")
        #expect(detok.commitDelta("abc") == "")
    }

    // MARK: - Genuine resync

    @Test("Genuine prefix rewrite resyncs and emits the divergent tail")
    func genuineResyncEmitsDivergentTail() {
        var detok = MFDetokenizer(tokenizer: tok)
        #expect(detok.commitDelta("abc") == "abc")
        // The decoder rewrote an already-emitted character: not a prefix. The
        // emitted "c" cannot be retracted, but "d" must not be dropped too.
        #expect(detok.commitDelta("abd") == "d")
        #expect(detok.emitted == "abd")
        // Streaming resumes from the resynced state.
        #expect(detok.commitDelta("abde") == "e")
    }

    @Test("Shorter decode resyncs rather than under-flowing")
    func shorterDecodeResyncs() {
        var detok = MFDetokenizer(tokenizer: tok)
        #expect(detok.commitDelta("abcdef") == "abcdef")
        #expect(detok.commitDelta("abc") == "")
        #expect(detok.emitted == "abc")
    }

    // MARK: - Clean-up rewrites break append-only decoding

    /// swift-transformers applies `cleanUpTokenizationSpaces` inside `decode`,
    /// and it defaults to true when the key is absent — which it is for Gemma.
    /// The rewrites delete a space that an earlier decode already emitted
    /// (" 's" -> "'s"), so the decode is not append-only and the resync path is
    /// reachable in ordinary text. The stray space cannot be retracted, but no
    /// character may be lost: clean-up only ever deletes whitespace, so the
    /// non-whitespace characters must match `decode` exactly.
    @Test("Clean-up rewrites never drop characters from the stream", arguments: [
        "it 's ok",
        "x = ' ' # a space",
        "do n't stop",
        "we 've seen",
        "they 're here",
        "I 'm sure",
    ])
    func cleanUpRewritesDoNotDropCharacters(_ target: String) {
        let ids = tok.encode(target, addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        let reference = tok.decode(ids)
        #expect(assembled.filter { !$0.isWhitespace } == reference.filter { !$0.isWhitespace },
                "characters lost: stream '\(assembled)' vs decode '\(reference)'")
    }

    // MARK: - Byte-fallback flush

    /// Gemma's SentencePiece splits an emoji into `<0xF0><0x9F><0xA6><0x99>`
    /// byte-fallback tokens, which `push` holds back in full. A stream that ends
    /// after a whole emoji plus part of the next one leaves `flush` assembling a
    /// buffer that is *partly* valid UTF-8 — the complete codepoint must still
    /// reach the caller.
    @Test("Flush keeps the complete codepoints in a partly-invalid byte tail")
    func flushKeepsCompleteCodepointsBeforeAPartialOne() {
        let ids = tok.encode("🦙🦙", addBOS: false)
        #expect(ids.count == 8, "expected eight byte-fallback tokens, got \(ids.count)")
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        // Truncate mid-way through the second emoji.
        for id in ids.dropLast() {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == "🦙\u{FFFD}",
                "byte-fallback tail mismatch: got '\(assembled)'")
    }

    // MARK: - End-to-end streaming

    @Test("Streaming reassembles combining-mark scripts", arguments: [
        "ห้าม",
        "การคัดกรอง",
        "ยินดีต้อนรับ",
        "हिन्दी में लिखा",
        "ជំរាបសួរ",
        "မင်္ဂလာပါ",
        "مرحبًا بالعالم",
        "👨\u{200D}👩\u{200D}👧\u{200D}👦 family",
        "🏳️\u{200D}🌈 flag",
    ])
    func streamingCombiningScripts(_ target: String) {
        let ids = tok.encode(target, addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == target,
                "stream reassembly mismatch: got '\(assembled)' want '\(target)'")
    }
}

/// The Gemma suite above cannot reach the byte-level failure mode: SentencePiece
/// pieces are whole scalars, and the bytes of a split codepoint arrive as
/// `<0xNN>` byte-fallback tokens that `push` already holds back.
///
/// Qwen's GPT-2-style byte-level BPE has no byte-fallback tokens. One scalar's
/// UTF-8 bytes simply span several ordinary tokens, and `ByteLevelDecoder`
/// finishes with `String(decoding: utfCodepoints, as: UTF8.self)`, so a decode
/// that stops mid-scalar materializes U+FFFD immediately. These tests run the
/// real `ByteLevel` fixture tokenizer, not a simulation.
@Suite("Detokenizer byte-level")
struct ByteLevelDetokenizerTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    }

    @Test("Scalars split across tokens stream without corruption", arguments: [
        "🦙",
        "🧿",
        "𓀀",
        "ᨆ",
        "🦙 llama",
        "mixed 漢 and 🦝 text",
        "👨\u{200D}👩\u{200D}👧\u{200D}👦 family",
    ])
    func splitScalarsStreamIntact(_ target: String) {
        let ids = tok.encode(target, addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            let delta = detok.push(id)
            #expect(!delta.unicodeScalars.contains("\u{FFFD}"),
                    "partial scalar leaked to the caller: '\(delta)'")
            assembled += delta
        }
        assembled += detok.flush()
        #expect(assembled == target,
                "stream reassembly mismatch: got '\(assembled)' want '\(target)'")
    }

    @Test("A replacement char the model really produced survives the stream", arguments: [
        "\u{FFFD}",
        "a\u{FFFD}",
        "\u{FFFD}b",
    ])
    func genuineReplacementCharIsNotSwallowed(_ target: String) {
        let ids = tok.encode(target, addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == target,
                "stream reassembly mismatch: got '\(assembled)' want '\(target)'")
    }

    /// The ChatML fixture sets `clean_up_tokenization_spaces: false`, matching
    /// the shipped Qwen 3.6 `tokenizer_config.json`. With clean-up off the decode
    /// is append-only in bytes, so the resync path is unreachable and the stream
    /// must equal `decode` exactly — including for the strings whose rewrites
    /// break Gemma. This is what pins "the resync change cannot move Qwen".
    @Test("Clean-up patterns stream byte-identically to decode", arguments: [
        "it 's ok",
        "x = ' ' # a space",
        "do n't stop",
        "we 've seen",
        "they 're here",
        "I 'm sure",
        "hi .",
        "Wait ! Really ?",
        "a , b",
        "🦙 do n't . 漢",
    ])
    func cleanUpPatternsAreByteIdentical(_ target: String) {
        let ids = tok.encode(target, addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == tok.decode(ids), "stream diverged from decode")
        #expect(assembled == target, "round trip diverged: got '\(assembled)'")
    }

    /// The trailing run can be longer than one scalar, so the scan has to be a
    /// loop. A single `if` passes every streaming test above, because a run only
    /// appears when the decoder emits two adjacent unresolvable subparts.
    @Test("Trailing replacement run is measured to its full length", arguments: [
        ("", 0),
        ("ab", 0),
        ("\u{FFFD}", 3),
        ("a\u{FFFD}", 3),
        ("a\u{FFFD}\u{FFFD}", 6),
        ("\u{FFFD}\u{FFFD}\u{FFFD}", 9),
        ("\u{FFFD}a", 0),
        ("\u{FFFD}a\u{FFFD}\u{FFFD}", 6),
        ("🦙\u{FFFD}", 3),
    ])
    func trailingRunIsMeasuredFully(_ probe: (text: String, expected: Int)) {
        #expect(MFDetokenizer.trailingReplacementByteCount(probe.text) == probe.expected,
                "for \(probe.text.debugDescription)")
    }

    /// An unpaired UTF-16 surrogate encoded as UTF-8 (`ED A0 80`) is three
    /// separate unresolvable subparts, so the decoder renders three adjacent
    /// U+FFFD. Byte-level BPE can emit those bytes as ordinary tokens.
    @Test("A multi-scalar replacement run is withheld as a whole")
    func multiScalarRunIsWithheldWhole() {
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for byte in [0xED, 0xA0] {
            let delta = detok.push(Int32(byte))
            #expect(delta.isEmpty, "withheld run leaked: '\(delta)'")
            assembled += delta
        }
        assembled += detok.flush()
        #expect(assembled == "\u{FFFD}\u{FFFD}",
                "run mismatch: got '\(assembled)'")
    }

    /// `flush()` releases a withheld run, so afterwards `emitted` can end with
    /// U+FFFD. Nothing in the type forbids a later `push`, and if one happens the
    /// committed prefix must not rewind — a shrinking `emitted` silently re-emits
    /// text the caller already received.
    @Test("Pushing after a flush never rewinds the committed prefix")
    func pushAfterFlushNeverRewinds() {
        let ids = tok.encode("a🦙", addBOS: false)
        #expect(ids.count == 5, "expected one ASCII plus four emoji bytes")
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids.prefix(2) {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == "a\u{FFFD}", "flush mismatch: got '\(assembled)'")

        var high = detok.emitted.utf8.count
        for id in ids.dropFirst(2) {
            assembled += detok.push(id)
            #expect(detok.emitted.utf8.count >= high,
                    "emitted rewound to '\(detok.emitted)' (\(detok.emitted.utf8.count) < \(high) bytes)")
            high = max(high, detok.emitted.utf8.count)
        }
    }

    @Test("A stream truncated mid-scalar flushes what the decoder saw")
    func truncatedScalarFlushesReplacementChar() {
        let ids = tok.encode("🦙", addBOS: false)
        #expect(ids.count > 1, "fixture must split the emoji across tokens")
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids.dropLast() {
            assembled += detok.push(id)
        }
        // The generation loop always flushes; a truncated codepoint has to
        // surface as the decoder's own U+FFFD rather than vanish.
        assembled += detok.flush()
        #expect(assembled == "\u{FFFD}",
                "truncated scalar mismatch: got '\(assembled)'")
    }
}
