import Foundation
import Testing
@testable import Mference

/// The reference tokenizer on a small synthetic byte-level vocabulary: the
/// qwen35 pre-tokenizer split, BPE merge order, special-token partition and
/// pieces. The check against llama.cpp's token ids on the real vocabulary is
/// env-gated in `ReferenceQuickPackTests`.
@Suite("ReferenceTokenizerTests")
struct ReferenceTokenizerTests {
    static func split(_ s: String) -> [String] {
        let cpts = s.unicodeScalars.map { $0.value }
        var out: [String] = []
        var start = 0
        for n in Native.ReferenceUnicode.splitQwen35(cpts) {
            out.append(String(String.UnicodeScalarView(cpts[start ..< start + n].map { Unicode.Scalar($0)! })))
            start += n
        }
        return out
    }

    @Test func qwen35SplitFollowsTheRegex() {
        #expect(Self.split("Hello world") == ["Hello", " world"])
        #expect(Self.split("I'm here, they'LL go") == ["I", "'m", " here", ",", " they", "'LL", " go"])
        #expect(Self.split("x = 12345;") == ["x", " =", " ", "1", "2", "3", "4", "5", ";"])
        #expect(Self.split("a\n\nb  c   ") == ["a", "\n\n", "b", " ", " c", "   "])
        #expect(Self.split("  \n  x") == ["  \n", " ", " x"])
        #expect(Self.split("e\u{301}t\u{e9}") == ["e\u{301}t\u{e9}"])
        #expect(Self.split("...!?\r\nnext") == ["...!?\r\n", "next"])
        #expect(Self.split("中文 字") == ["中文", " 字"])
    }

    /// A byte-level vocabulary: all 256 byte tokens, some merges, specials.
    static func vocabulary() throws -> Native.ReferenceTokenizer {
        var tokens: [String] = []
        for b in 0..<256 {
            tokens.append(String(decoding: Native.ReferenceUnicode.byteToUTF8[b], as: UTF8.self))
        }
        let merges = ["h e", "l l", "he ll", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d"]
        for m in merges { tokens.append(m.replacingOccurrences(of: " ", with: "")) }
        let specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|fim_pad|>", "<tool_call>"]
        tokens += specials
        var types = [Int32](repeating: 1, count: tokens.count)
        for i in (tokens.count - 5)..<(tokens.count - 1) { types[i] = 3 }
        types[tokens.count - 1] = 4
        let data = Native.TokenizerData(model: "gpt2", pre: "qwen35", tokens: tokens, tokenTypes: types,
                                        merges: merges, bosTokenID: nil, eosTokenID: Int32(tokens.count - 3),
                                        paddingTokenID: nil, addBOS: nil, addEOS: nil)
        return try Native.ReferenceTokenizer(data: data)
    }

    @Test func bpeMergesAndPieces() throws {
        let t = try Self.vocabulary()
        let ids = try t.encode("hello world", addSpecial: true, parseSpecial: true)
        #expect(ids.count == 2)
        #expect(t.piece(ids[0], special: false) == Array("hello".utf8))
        #expect(t.piece(ids[1], special: false) == Array(" world".utf8))
        // Unmerged text falls back to byte tokens; multi-byte UTF-8 round trips.
        let text = "héllo wo ✓"
        let tokens = try t.encode(text, addSpecial: false, parseSpecial: false)
        #expect(tokens.flatMap { t.piece($0, special: false) } == Array(text.utf8))
    }

    @Test func specialTokensPartitionAndEOG() throws {
        let t = try Self.vocabulary()
        let n = Int32(t.vocabSize)
        let imStart = n - 5, imEnd = n - 4, eot = n - 3, fimPad = n - 2, toolCall = n - 1
        let ids = try t.encode("<|im_start|>hello<|im_end|><tool_call>", addSpecial: false, parseSpecial: true)
        #expect(ids.first == imStart)
        #expect(ids[ids.count - 2] == imEnd && ids.last == toolCall)
        // Without parse_special, control tokens are plain text; user-defined ones still split.
        let plain = try t.encode("<|im_end|><tool_call>", addSpecial: false, parseSpecial: false)
        #expect(!plain.contains(imEnd))
        #expect(plain.last == toolCall)
        #expect(t.endOfGenerationTokenIDs == [imEnd, eot, fimPad])
        #expect(t.piece(imEnd, special: false).isEmpty)
        #expect(t.piece(imEnd, special: true) == Array("<|im_end|>".utf8))
        #expect(t.piece(toolCall, special: false) == Array("<tool_call>".utf8))
        #expect(t.specialTokens.first.map { t.piece($0, special: true).count } == 13)
    }

    /// Divergences from the pin found in review, each pinned here:
    /// `llama_decode_text`'s unknown-byte text, BPE `token_to_byte`,
    /// first-wins merge ranks, and gpt2's default BOS/EOS of 11.
    @Test func llamaCppEdgeCases() throws {
        // An unmapped code point: "[UNK_BYTE_0x" + its UTF-8 hex + the whole text + "]".
        let text = Array("a€".utf8)
        #expect(Native.ReferenceUnicode.decodeByteText(text)
                == Array("a[UNK_BYTE_0xe282aca€]".utf8))

        var tokens = (0..<20).map { "t\($0)" }
        tokens[12] = "<0x41>"
        tokens[13] = "<0x"
        var types = [Int32](repeating: 1, count: tokens.count)
        types[12] = 6
        types[13] = 6
        let data = Native.TokenizerData(model: "gpt2", pre: "qwen35", tokens: tokens, tokenTypes: types,
                                        merges: ["t 1", "t 2", "t 1"], bosTokenID: nil, eosTokenID: nil,
                                        paddingTokenID: nil, addBOS: nil, addEOS: nil)
        let t = try Native.ReferenceTokenizer(data: data)
        #expect(t.piece(12, special: false) == [0x41])
        #expect(t.piece(13, special: false) == [0])
        // strtol(_, 16) on text[3..<5]: whitespace, sign and a bare 0x as C does.
        for (text, byte) in [("ab<0", 0), ("<0x 7", 7), ("<0x-1", 0xFF), ("<0x+f", 0x0F), ("<0x0x", 0), ("<0xzz", 0)] {
            #expect(Native.ReferenceTokenizer.strtol16(Array(Array(text.utf8)[3...])) & 0xFF == byte, "\(text)")
        }
        #expect(t.ranks[.init(left: Array("t".utf8), right: Array("1".utf8))] == 0)
        #expect(t.bosTokenID == 11 && t.eosTokenID == 11)
        #expect(t.endOfGenerationTokenIDs.contains(11))
        let template = Native.ReferenceChatTemplateSource(template: "", data: data)
        #expect(template.bosToken == "t11" && template.eosToken == "t11")
        // An id from the GGUF replaces the default; an out-of-range id does not.
        var withIDs = data
        withIDs.eosTokenID = 5
        withIDs.bosTokenID = 99
        let ids = Native.ReferenceTokenizer.specialIDs(withIDs)
        #expect(ids.eos == 5 && ids.bos == 11)
        // GGUF special-token keys replace text detection and join EOG as
        // llama.cpp's fim_pad/fim_rep/fim_sep/eot/eom do; out-of-range ids are ignored.
        var keyed = data
        keyed.specialTokenIDs = ["eot": 7, "fim_pad": 8, "fim_rep": 999, "unknown": 3]
        let k = try Native.ReferenceTokenizer(data: keyed)
        #expect(k.endOfGenerationTokenIDs == [7, 8, 11])
        #expect(k.attr(7) == .normal)
    }

    @Test func refusesOtherTokenizers() {
        let data = Native.TokenizerData(model: "llama", pre: "default", tokens: ["a"], tokenTypes: [1], merges: [],
                                        bosTokenID: nil, eosTokenID: nil, paddingTokenID: nil, addBOS: nil,
                                        addEOS: nil)
        #expect(throws: Native.NativeError.self) { _ = try Native.ReferenceTokenizer(data: data) }
    }
}
