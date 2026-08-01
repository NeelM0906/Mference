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

    @Test("Genuine prefix rewrite resyncs and emits nothing")
    func genuineResyncEmitsNothing() {
        var detok = MFDetokenizer(tokenizer: tok)
        #expect(detok.commitDelta("abc") == "abc")
        // The decoder rewrote an already-emitted character: not a prefix.
        #expect(detok.commitDelta("abd") == "")
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
