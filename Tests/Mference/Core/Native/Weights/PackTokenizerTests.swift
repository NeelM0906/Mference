import Foundation
import Testing
@testable import Mference

/// The pack tokenizer against the frozen reference tokenizer (the oracle,
/// itself checked against llama.cpp's ids on the golden prompts): Unicode
/// flags over every code point, the qwen35 split, BPE merges on trained and
/// hand-made vocabularies, special-token partition, EOG and pieces. The real
/// Quick vocabulary and the golden token ids are in `PackQuickTests`.
@Suite("PackTokenizerTests")
struct PackTokenizerTests {
    /// GPT-2 byte-encoded text of `s`.
    static func encoded(_ s: String) -> String {
        String(decoding: Array(s.utf8).flatMap { Native.PackUnicode.byteToUTF8[Int($0)] }, as: UTF8.self)
    }

    /// A small BPE vocabulary trained on `corpus` (most frequent pair first,
    /// ties to the lexicographically smallest), plus specials.
    static func trained(corpus: [String], merges count: Int) -> Native.TokenizerData {
        var tokens = (0..<256).map { String(decoding: Native.PackUnicode.byteToUTF8[$0], as: UTF8.self) }
        var words: [[String]] = []
        for text in corpus {
            let cpts = Array(text.unicodeScalars.map(\.value))
            var start = 0
            for n in Native.ReferenceUnicode.splitQwen35(cpts) {
                let word = String(String.UnicodeScalarView(cpts[start ..< start + n].map { Unicode.Scalar($0)! }))
                words.append(Array(Self.encoded(word)).map(String.init))
                start += n
            }
        }
        var merges: [String] = []
        for _ in 0..<count {
            var counts: [String: Int] = [:]
            for w in words where w.count > 1 {
                for i in 1..<w.count { counts[w[i - 1] + " " + w[i], default: 0] += 1 }
            }
            guard let best = counts.max(by: { $0.value != $1.value ? $0.value < $1.value : $0.key > $1.key }) else {
                break
            }
            merges.append(best.key)
            let parts = best.key.split(separator: " ", maxSplits: 1).map(String.init)
            tokens.append(parts[0] + parts[1])
            words = words.map { w in
                var out: [String] = []
                var i = 0
                while i < w.count {
                    if i + 1 < w.count && w[i] == parts[0] && w[i + 1] == parts[1] {
                        out.append(parts[0] + parts[1])
                        i += 2
                    } else {
                        out.append(w[i])
                        i += 1
                    }
                }
                return out
            }
        }
        // Merge-only operands that are not tokens (exercise the byte fallback),
        // a merge without a space, a repeated pair (first rank wins) and a
        // duplicate token text (last id wins).
        merges += ["Ġ q", "z z", "nospace", "e r"]
        tokens.append("Ġqz")
        tokens.append(tokens[Int(Array("e".utf8)[0])])
        let specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|fim_pad|>", "<tool_call>", "</tool_call>",
                        "<think>", "<|unused_0|>", ""]
        var types = [Int32](repeating: 1, count: tokens.count)
        tokens += specials
        types += [3, 3, 3, 3, 4, 4, 4, 3, 3]
        tokens += ["<0x41>", "<0x"]
        types += [6, 6]
        return Native.TokenizerData(model: "gpt2", pre: "qwen35", tokens: tokens, tokenTypes: types, merges: merges,
                                    bosTokenID: nil, eosTokenID: Int32(tokens.count - 9), paddingTokenID: nil,
                                    addBOS: nil, addEOS: nil)
    }

    static let corpus = [
        "Hello world, hello there! The weather in Lisbon is light rain; it's 18 degrees.",
        "In two sentences, explain what a hash table is. A hash table maps keys to values.",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "Return only a JSON object with the keys \"city\" and \"population_millions\".",
        "Coastal wetlands reduce flood damage: they're buffers, they'll slow water, we've seen it.",
        "中文文本，日本語のテキスト、한국어 텍스트. Ünïcödé café naïve façade.",
    ]

    static let fragments = [
        "h", "e", "l", "o", " ", "  ", "\n", "\n\n", "\r\n", "\t", "'s", "'LL", "'re", "'ve", "'m", "'d", "'x",
        "123", "4", "中", "文", "é", "e\u{301}", "🙂", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
        "<think>", "<|unused_0|>", "...", "!?", "Ġ", "\u{a0}", "\u{3000}", "ﬁ", "Ж", "x", "y", "€", "hello",
        " world", "the", " weather", "fibonacci", "(", ")", "{", "\"", "\\", "\u{0}", "\u{7f}", "\u{85}",
        "\u{2028}", "ء", "ـ", "a\u{300}\u{301}", "🇵🇹", "Q", "z", "zz", "q", " q", "<|im_end", "|>", "<0x41>",
    ]

    static func fuzz(_ count: Int, seed: UInt64) -> [String] {
        var rng = RefTest.SplitMix64(seed)
        var out = corpus + ["", " ", "\n", "  \n  x", "a\n\nb  c   ", "<|im_start|>user\nhi<|im_end|>\n",
                            "<|im_end|><|im_end|>", "<<|im_end|>>", "' 's 'S"]
        for _ in 0..<count {
            let n = Int(rng.next() % 40)
            out.append((0..<n).map { _ in fragments[Int(rng.next() % UInt64(fragments.count))] }.joined())
        }
        return out
    }

    /// Encode with every flag combination and every piece, against the reference.
    static func compare(_ data: Native.TokenizerData, texts: [String]) throws -> Int {
        let ours = try Native.PackTokenizer(data: data)
        let ref = try Native.ReferenceTokenizer(data: data)
        #expect(ours.vocabSize == ref.vocabSize)
        #expect(ours.eosTokenID == ref.eosTokenID && ours.bosTokenID == ref.bosTokenID)
        #expect(ours.endOfGenerationTokenIDs == ref.endOfGenerationTokenIDs)
        #expect(ours.specialTokens == ref.specialTokens)
        for t in 0..<Int32(ref.vocabSize) {
            #expect(ours.attr(t).rawValue == ref.attr(t).rawValue, "attr \(t)")
            #expect(ours.piece(t, special: false) == ref.piece(t, special: false), "piece \(t)")
            #expect(ours.piece(t, special: true) == ref.piece(t, special: true), "special piece \(t)")
        }
        #expect(ours.piece(-1, special: true).isEmpty && ours.piece(Int32(ref.vocabSize), special: true).isEmpty)
        var checked = 0
        for text in texts {
            for (addSpecial, parseSpecial) in [(false, false), (false, true), (true, false), (true, true)] {
                let a = try ours.encode(text, addSpecial: addSpecial, parseSpecial: parseSpecial)
                let b = try ref.encode(text, addSpecial: addSpecial, parseSpecial: parseSpecial)
                #expect(a == b, "\(text.debugDescription) add \(addSpecial) parse \(parseSpecial)")
                checked += 1
            }
        }
        return checked
    }

    @Test func unicodeFlagsMatchTheReferenceForEveryCodePoint() {
        var mismatches: [UInt32] = []
        for cp in UInt32(0) ..< 0x110000 {
            let r = Native.ReferenceUnicode.flags(cp)
            var want: UInt8 = Native.PackUnicode.inRange
            if r.letter { want |= Native.PackUnicode.letter }
            if r.number { want |= Native.PackUnicode.number }
            if r.mark { want |= Native.PackUnicode.mark }
            if r.whitespace { want |= Native.PackUnicode.whitespace }
            if Native.PackUnicode.flags(cp) != want { mismatches.append(cp) }
        }
        #expect(mismatches.isEmpty, "\(mismatches.prefix(10))")
        for b in 0..<256 {
            #expect(Native.PackUnicode.byteToUTF8[b] == Native.ReferenceUnicode.byteToUTF8[b])
        }
    }

    @Test func qwen35SplitMatchesTheReference() {
        for text in Self.fuzz(1500, seed: 11) {
            let cps = text.unicodeScalars.map(\.value)
            let flags = cps.map(Native.PackUnicode.flags)
            var ends: [Int32] = []
            cps.withUnsafeBufferPointer { c in
                flags.withUnsafeBufferPointer { f in Native.PackUnicode.splitQwen35(c, f, into: &ends) }
            }
            var lengths: [Int] = []
            var prev = 0
            for e in ends {
                lengths.append(Int(e) - prev)
                prev = Int(e)
            }
            #expect(lengths == Native.ReferenceUnicode.splitQwen35(cps), "\(text.debugDescription)")
        }
    }

    @Test func referenceSyntheticVocabularyMatches() throws {
        let data = try ReferenceTokenizerTests.vocabulary().data
        #expect(try Self.compare(data, texts: Self.fuzz(800, seed: 1)) > 3000)
        let tiny = Native.TokenizerData(model: "gpt2", pre: "qwen35", tokens: RefSynthetic.vocabulary.tokens,
                                        tokenTypes: RefSynthetic.vocabulary.types, merges: RefSynthetic.vocabulary.merges,
                                        bosTokenID: nil, eosTokenID: 263, paddingTokenID: nil, addBOS: nil, addEOS: nil)
        _ = try Self.compare(tiny, texts: Self.fuzz(300, seed: 2))
    }

    @Test func trainedVocabularyMatches() throws {
        let data = Self.trained(corpus: Self.corpus + Self.corpus.map { $0.lowercased() }, merges: 400)
        #expect(data.merges.count > 150, "\(data.merges.count) merges")
        #expect(try Self.compare(data, texts: Self.fuzz(1500, seed: 3)) > 6000)
        // add_bos / add_eos and keyed special ids.
        var flagged = data
        flagged.addBOS = true
        flagged.addEOS = true
        flagged.bosTokenID = Int32(data.tokens.count - 11)
        flagged.specialTokenIDs = ["eot": Int32(data.tokens.count - 10), "fim_pad": 3, "fim_rep": 99_999]
        _ = try Self.compare(flagged, texts: Self.fuzz(200, seed: 4))
    }

    @Test func llamaCppEdgeCasesMatch() throws {
        #expect(Native.PackTokenizer.decodeByteText(ArraySlice(Array("a€".utf8)))
                == Native.ReferenceUnicode.decodeByteText(Array("a€".utf8)))
        for (text, _) in [("ab<0", 0), ("<0x 7", 7), ("<0x-1", 0xFF), ("<0x+f", 0x0F), ("<0x0x", 0), ("<0xzz", 0),
                          ("<0x4G", 4), ("<0X1f", 0x1F)] {
            let digits = Array(Array(text.utf8)[3...])
            #expect(Native.PackTokenizer.strtol16(digits) == Native.ReferenceTokenizer.strtol16(digits), "\(text)")
        }
        // gpt2 default BOS/EOS of 11, an out-of-range GGUF id.
        var tokens = (0..<20).map { "t\($0)" }
        tokens[12] = "<0x41>"
        var types = [Int32](repeating: 1, count: tokens.count)
        types[12] = 6
        var data = Native.TokenizerData(model: "gpt2", pre: "qwen35", tokens: tokens, tokenTypes: types,
                                        merges: ["t 1", "t 2", "t 1"], bosTokenID: 99, eosTokenID: nil,
                                        paddingTokenID: nil, addBOS: true, addEOS: nil)
        _ = try Self.compare(data, texts: ["t1t2", "t11", ""])
        let template = Native.PackChatTemplateSource(template: "x", data: data)
        let refTemplate = Native.ReferenceChatTemplateSource(template: "x", data: data)
        #expect(template.bosToken == refTemplate.bosToken && template.eosToken == refTemplate.eosToken)
        data.tokenTypes = []
        _ = try Self.compare(data, texts: ["t1t2"])
        // Refuses other tokenizers and malformed type arrays.
        var other = data
        other.model = "llama"
        #expect(throws: Native.NativeError.self) { _ = try Native.PackTokenizer(data: other) }
        other = data
        other.pre = "default"
        #expect(throws: Native.NativeError.self) { _ = try Native.PackTokenizer(data: other) }
        other = data
        other.tokenTypes = [1]
        #expect(throws: Native.NativeError.self) { _ = try Native.PackTokenizer(data: other) }
    }

    @Test func tokenizerSourceBuildsOnce() throws {
        let source = Native.PackTokenizerSource(data: try ReferenceTokenizerTests.vocabulary().data)
        let a = try source.makeTokenizer(), b = try source.makeTokenizer()
        #expect(a === b)
    }

    @Test func byteStringTableInternsAndGrows() {
        var table = Native.PackByteStrings(expected: 1)
        var ids: [Int32] = []
        for i in 0..<5000 {
            ids.append(Array("s\(i)".utf8).withUnsafeBytes { table.intern($0) })
        }
        #expect(ids == (0..<5000).map(Int32.init))
        #expect(Array("s4999".utf8).withUnsafeBytes { table.find($0) } == 4999)
        #expect(Array("s5000".utf8).withUnsafeBytes { table.find($0) } == -1)
        #expect([UInt8]().withUnsafeBytes { table.intern($0) } == 5000)
        #expect(Array("s7".utf8).withUnsafeBytes { table.intern($0) } == 7)
        #expect(Array(table.bytes(42)) == Array("s42".utf8) && table.length(42) == 3)
    }
}
