import Foundation
import Testing
@testable import Mference

/// Encode, decode and streaming detokenization on the committed ChatML
/// (Qwen 3.6) tokenizer fixture; no network or Hub cache.
@Suite("Tokenizer")
struct TokenizerTests {
    let tok: MFTokenizer

    init() async throws {
        self.tok = try await MFTokenizer.chatMLFixture()
    }

    // MARK: - Special tokens

    @Test("Special token IDs are within vocab")
    func specialTokensWithinVocab() {
        let ids: [Int32] = [tok.bosID, tok.eosID, tok.padID, tok.endOfTurnID]
        #expect(tok.eosID != tok.endOfTurnID)
        for id in ids {
            #expect(id >= 0)
            #expect(id < Int32(tok.vocabSize))
        }
    }

    @Test("Stop-token set covers EOS and end-of-turn")
    func stopTokens() {
        #expect(tok.stopTokenIDs.contains(tok.eosID))
        #expect(tok.stopTokenIDs.contains(tok.endOfTurnID))
        #expect(tok.stopTokenIDs.count == 2)
    }

    // MARK: - Encode / decode

    @Test("Round-trip ASCII", arguments: [
        "Hello, world.",
        "The quick brown fox jumps over the lazy dog.",
        "code:  let x = 42;  // comment",
        "numbers 0 1 2 3 4 5 6 7 8 9",
    ])
    func roundTripASCII(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
    }

    @Test("Round-trip multi-byte UTF-8", arguments: [
        "你好，世界。",
        "漢字",
        "🦝 raccoon emoji",
        "mixed 漢 and 🦝 and a",
        "Здравствуй",
    ])
    func roundTripMultibyte(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
    }

    @Test("ChatML never prepends a BOS")
    func addBOSIsANoOp() {
        #expect(tok.encode("x", addBOS: true) == tok.encode("x", addBOS: false))
        #expect(tok.encode("", addBOS: true).isEmpty)
    }

    @Test("Decoding strips special tokens when requested")
    func decodeStripsSpecial() {
        let ids = [tok.endOfTurnID] + tok.encode("hi", addBOS: false)
        let withoutSpecial = tok.decode(ids, skipSpecialTokens: true)
        #expect(withoutSpecial == "hi")
    }

    // MARK: - Streaming detokenizer

    @Test("Streaming detokenizer reassembles ASCII")
    func streamingASCII() {
        let target = "Hello, world."
        assertStreams(target)
    }

    @Test("Streaming detokenizer reassembles multi-byte UTF-8", arguments: [
        "漢字",
        "你好",
        "🦝🦝🦝",
        "mixed 漢 and 🦝",
        "Здравствуй мир",
        "ends with emoji 🦝",
        "🦝 starts with emoji",
        "🦝 middle 漢 end",
    ])
    func streamingMultibyte(_ target: String) {
        assertStreams(target)
    }

    @Test("Streaming detokenizer never emits replacement chars")
    func streamingNoMojibake() {
        let ids = tok.encode("mixed 漢 and 🦝 text", addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        for id in ids {
            let delta = detok.push(id)
            #expect(!delta.unicodeScalars.contains("\u{FFFD}"),
                    "delta contained replacement char: '\(delta)'")
        }
        let tail = detok.flush()
        #expect(!tail.unicodeScalars.contains("\u{FFFD}"))
    }

    @Test("Streaming detokenizer preserves long mixed output", arguments: [
        "The quick brown fox jumps over the lazy dog. 0123456789\n",
        "\u{E000}\u{E001}\u{E002}\u{E003}\u{E004}\u{E005}\u{E006}\u{E007}",
        "Mference 漢字 Здравствуй 🦝 café Ελληνικά \u{E000}\n",
    ])
    func streamingLongOutput(_ seed: String) {
        var text = seed
        var ids = tok.encode(text, addBOS: false)
        while ids.count < 256 {
            text += seed
            ids = tok.encode(text, addBOS: false)
        }

        let prefix = Array(ids.prefix(256))
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in prefix {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == tok.decode(prefix))
        // A byte-level vocab can cut a scalar at the 256-token boundary; the
        // flushed tail then ends in U+FFFD, exactly as `decode` renders it.
        // No replacement character may appear before that tail.
        let interior = String(String.UnicodeScalarView(
            assembled.unicodeScalars.reversed().drop(while: { $0 == "\u{FFFD}" }).reversed()))
        #expect(!interior.unicodeScalars.contains("\u{FFFD}"))
    }

    @Test("Flush with no tokens yields empty string")
    func streamingEmpty() {
        var detok = MFDetokenizer(tokenizer: tok)
        #expect(detok.flush() == "")
    }

    @Test("Single-token push works for in-vocab characters")
    func streamingSingleToken() {
        // "漢字" is a single in-vocab token (verified empirically).
        var detok = MFDetokenizer(tokenizer: tok)
        let ids = tok.encode("漢字", addBOS: false)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == "漢字")
    }

    // MARK: - Helpers

    private func assertStreams(_ target: String) {
        let ids = tok.encode(target, addBOS: false)
        var detok = MFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == target, "stream reassembly mismatch: got '\(assembled)' want '\(target)'")
    }
}
