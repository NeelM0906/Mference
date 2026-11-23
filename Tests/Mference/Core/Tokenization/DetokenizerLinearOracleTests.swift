import Foundation
import Testing
@testable import Mference

/// `MFDetokenizer` decodes incrementally (a window of recent tokens per push);
/// `QuadraticDetokenizerOracle` is the implementation it replaced, which
/// decodes every stable token on each push. For each tokenizer `MFDetokenizer`
/// serves, long adversarial streams go through both, and every emitted chunk
/// must be equal: each push's, the flush's, and pushes after the flush. The
/// stop-string matcher downstream must stop at the same push with the same
/// text. Also checked at every push: the incremental decode equals the full
/// decode of the same IDs.
///
/// Tokenizers:
/// - the four in the repository: ChatML (Qwen 3.6 and 3.8), Maple's ChatML,
///   MiniCPM5, and Inkling 2's, which the installer synthesizes from the GGUF.
///   All are byte-level BPE with clean-up off. Their tiny vocabularies make
///   almost every token one byte, so multi-byte scalars split across tokens
///   constantly.
/// - a SentencePiece-style tokenizer written here: byte fallback, `▁`
///   leading-space rules with the leading-space strip, and
///   `cleanUpTokenizationSpaces` on, the paths none of the four reaches.
/// - with `MFERENCE_DETOK_REAL_TOKENIZERS` (colon-separated tokenizer folders),
///   the installs' full tokenizers. These are files only, no model.
@Suite("Incremental detokenizer against the full-decode oracle", .serialized)
struct DetokenizerLinearOracleTests {
    // MARK: Streams

    /// Text that exercises every rule: multi-byte scripts, grapheme
    /// clusters, emoji sequences, combining marks, clean-up patterns (" .",
    /// " 's", " n't" and the rest), whitespace runs, and a literal U+FFFD.
    static let adversarialText = """
        Hello , world ! It 's a test . Do n't stop ; you 're right , I 'm sure we 've seen it ' s end ? \
        中文字符和标点。日本語のテキスト、カタカナ。한국어 문장입니다. Ελληνικά κείμενα. Русский текст. \
        العربية نص. עברית. हिन्दी पाठ। ภาษาไทย ห้าม การคัดกรอง ✓ Emoji: 👨‍👩‍👧‍👦 🙂 👍🏽 🇯🇵 ❤️ \
        Combining: e\u{301} a\u{308} Z\u{335}\u{336} Math: ∑∫√∞ ≠ ≤ ½ Code: `let x = [1, 2, 3].map { $0 * 2 }`
        \ttabs  double  spaces\r\n replacement \u{FFFD} done .
        """

    /// Seeded xorshift, so a failure names a reproducible stream.
    struct Generator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return state
        }
        mutating func below(_ n: Int) -> Int { Int(next() % UInt64(n)) }
    }

    /// The streams for `tok`: its encoding of the adversarial text (repeated),
    /// uniform random IDs over the whole vocabulary (special and added tokens
    /// included), random IDs from its one-character tokens only (byte-level
    /// bytes, so arbitrary and invalid UTF-8), and a mix of all three.
    static func streams(_ tok: MFTokenizer, randomLength: Int, seed: UInt64) -> [(String, [Int32])] {
        let text = tok.encode(String(repeating: adversarialText + "\n", count: 3), addBOS: false)
        let vocab = tok.vocabSize
        var g = Generator(state: seed)
        let uniform = (0..<randomLength).map { _ in Int32(g.below(vocab)) }
        let oneCharacter = (0..<vocab).filter {
            (tok.tokenizer.convertIdToToken($0) ?? "").count == 1
        }
        let bytes = oneCharacter.isEmpty ? [] : (0..<randomLength).map { _ in
            Int32(oneCharacter[g.below(oneCharacter.count)])
        }
        var mixed: [Int32] = []
        while mixed.count < randomLength {
            switch g.below(3) {
            case 0: mixed += text.dropFirst(g.below(max(1, text.count - 40))).prefix(1 + g.below(40))
            case 1: mixed += (0..<(1 + g.below(12))).map { _ in Int32(g.below(vocab)) }
            default:
                if !oneCharacter.isEmpty {
                    mixed += (0..<(1 + g.below(12))).map { _ in Int32(oneCharacter[g.below(oneCharacter.count)]) }
                }
            }
        }
        return [("text", text), ("uniform", uniform), ("one-character", bytes), ("mixed", mixed)]
    }

    // MARK: The comparison

    /// Byte equality: `String ==` compares by canonical equivalence, which
    /// would accept different bytes for the same text.
    static func same(_ a: String, _ b: String) -> Bool { a.utf8.elementsEqual(b.utf8) }

    /// Pushes `ids` through both implementations, flushes both, then pushes a
    /// few more IDs after the flush. Every chunk must match byte for byte, and
    /// the incremental decode must equal the full decode after every push.
    /// Returns the oracle's chunks.
    @discardableResult
    static func compare(_ tok: MFTokenizer, _ ids: [Int32], label: String) -> [String] {
        var oracle = QuadraticDetokenizerOracle(tokenizer: tok)
        var linear = MFDetokenizer(tokenizer: tok)
        var chunks: [String] = []
        for (i, id) in ids.enumerated() {
            let a = oracle.push(id)
            let b = linear.push(id)
            if !same(a, b) {
                Issue.record("\(label): push \(i) (id \(id)) oracle \(a.debugDescription) incremental \(b.debugDescription)")
                return chunks
            }
            let full = tok.tokenizer.decode(tokens: linear.stableIDs, skipSpecialTokens: true)
            if !same(linear.decoded, full) {
                Issue.record("\(label): push \(i) incremental decode differs from the full decode")
                return chunks
            }
            chunks.append(a)
        }
        let a = oracle.flush(), b = linear.flush()
        if !same(a, b) { Issue.record("\(label): flush oracle \(a.debugDescription) incremental \(b.debugDescription)") }
        chunks.append(a)
        for (i, id) in ids.prefix(24).enumerated() {
            let a = oracle.push(id), b = linear.push(id)
            if !same(a, b) {
                Issue.record("\(label): push \(i) after flush oracle \(a.debugDescription) incremental \(b.debugDescription)")
                break
            }
        }
        return chunks
    }

    /// Stop strings that span pushes: the end of one chunk and the start of a
    /// later one. Both chunk streams go through `StreamingStopMatcher` the way
    /// `runRawCompletion` uses it, and must stop at the same push with the
    /// same text.
    static func compareStops(_ tok: MFTokenizer, _ ids: [Int32], chunks: [String], label: String) {
        // Stops of two scalars from each side of a chunk boundary that occur in
        // the whole text. Both implementations must agree on every one. Only
        // the printable-ASCII ones must also match: the matcher's
        // `range(of:)` compares grapheme clusters, so it cannot find a stop
        // that ends inside one ("❤️" without its variation selector, half of
        // "\r\n"), whichever implementation fed it.
        let joined = chunks.joined()
        let joinable = chunks.indices.dropLast().filter { !chunks[$0].isEmpty && !chunks[$0 + 1].isEmpty }
        let candidates = joinable.filter { $0 % 37 == 3 }.map { k in
            String(String.UnicodeScalarView(chunks[k].unicodeScalars.suffix(2)))
                + String(String.UnicodeScalarView(chunks[k + 1].unicodeScalars.prefix(2)))
        }
        for stop in candidates.filter({ joined.range(of: $0) != nil }).prefix(6) {
            var oracle = QuadraticDetokenizerOracle(tokenizer: tok)
            var linear = MFDetokenizer(tokenizer: tok)
            var oracleStop = StreamingStopMatcher(stops: [stop])
            var linearStop = StreamingStopMatcher(stops: [stop])
            var oracleOut = "", linearOut = ""
            var oracleAt = -1, linearAt = -1
            for (i, id) in ids.enumerated() {
                if oracleAt < 0 {
                    oracleOut += oracleStop.push(oracle.push(id))
                    if oracleStop.isStopped { oracleAt = i }
                }
                if linearAt < 0 {
                    linearOut += linearStop.push(linear.push(id))
                    if linearStop.isStopped { linearAt = i }
                }
                if oracleAt >= 0 && linearAt >= 0 { break }
            }
            oracleOut += oracleStop.finish()
            linearOut += linearStop.finish()
            #expect(oracleAt == linearAt && same(oracleOut, linearOut), "\(label): stop \(stop.debugDescription)")
            if stop.unicodeScalars.allSatisfy({ (0x20...0x7E).contains($0.value) }) {
                #expect(oracleAt >= 0, "\(label): stop \(stop.debugDescription) never matched")
            }
        }
    }

    static func run(_ tok: MFTokenizer, name: String, randomLength: Int, seed: UInt64) {
        for (kind, ids) in streams(tok, randomLength: randomLength, seed: seed) {
            let label = "\(name)/\(kind) (\(ids.count) tokens, seed \(seed))"
            let chunks = compare(tok, ids, label: label)
            compareStops(tok, ids, chunks: chunks, label: label)
        }
    }

    // MARK: The tokenizers in the repository

    static func fixture(_ name: String) throws -> URL {
        try #require(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"))
    }

    @Test func chatMLQwen() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixture("ChatMLTokenizer"), family: .qwen36)
        Self.run(tok, name: "ChatML (Qwen 3.6/3.8)", randomLength: 3_000, seed: 0x51)
    }

    @Test func mapleChatML() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixture("MapleChatMLTokenizer"), family: .maple)
        Self.run(tok, name: "Maple ChatML", randomLength: 3_000, seed: 0x52)
    }

    @Test func miniCPM5() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixture("MiniCPM5Tokenizer"), family: .minicpm5)
        Self.run(tok, name: "MiniCPM5", randomLength: 3_000, seed: 0x53)
    }

    @Test func inkling2GGUFDialect() async throws {
        let tok = try await MFTokenizer.load(from: Self.fixture("Inkling2Tokenizer"), family: .inkling2Loop)
        #expect(tok.dialect == .inkling2)
        Self.run(tok, name: "Inkling 2 (GGUF)", randomLength: 3_000, seed: 0x54)
    }

    // MARK: Byte fallback, leading-space rules and clean-up

    /// A Llama-style SentencePiece tokenizer: `<0xNN>` byte-fallback tokens,
    /// `▁`-prefixed pieces, the decoder chain Replace(▁ → space), ByteFallback,
    /// Fuse, Strip(one leading space), and `clean_up_tokenization_spaces`
    /// true. ChatML specials make it a supported dialect.
    static func sentencePieceFixture() throws -> URL {
        let folder = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mference-detok-sp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var vocab: [String: Int] = ["<unk>": 0, "<s>": 1, "</s>": 2]
        for b in 0..<256 { vocab[String(format: "<0x%02X>", b)] = vocab.count }
        let pieces = ["▁", "▁▁", "▁the", "▁do", "n't", "▁n't", "'s", "▁'s", ".", "▁.", ",", "▁,", "!", "▁!",
                      "?", "▁?", "'", "▁'", "m", "▁'m", "'m", "ve", "'ve", "▁'ve", "re", "'re", "▁'re",
                      "▁it", "▁is", "▁a", "▁test", "▁don", "t", "▁hello", "▁world", "e", "a", "s", "▁x",
                      "▁中", "文", "字", "▁日本", "語", "é", "▁café", "▁naïve", "ห", "้", "า", "ม", "▁ห้าม",
                      "\n", "▁\n", "\t", "▁end", "▁ok", "ok", "▁'", "▁▁▁"]
        for p in pieces where vocab[p] == nil { vocab[p] = vocab.count }
        let specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>",
                        "<tool_response>", "</tool_response>", "<think>", "</think>"]
        let added: [[String: Any]] = specials.enumerated().map { i, content in
            ["id": vocab.count + i, "content": content, "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true]
        }
        let tokenizer: [String: Any] = [
            "version": "1.0", "truncation": NSNull(), "padding": NSNull(), "added_tokens": added,
            "normalizer": NSNull(),
            "pre_tokenizer": ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false],
            "post_processor": NSNull(),
            "decoder": ["type": "Sequence", "decoders": [
                ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                ["type": "ByteFallback"],
                ["type": "Fuse"],
                ["type": "Strip", "content": " ", "start": 1, "stop": 0],
            ]],
            "model": ["type": "BPE", "dropout": NSNull(), "unk_token": "<unk>", "continuing_subword_prefix": "",
                      "end_of_word_suffix": "", "fuse_unk": true, "byte_fallback": true,
                      "ignore_merges": false, "vocab": vocab, "merges": [] as [String]],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer, options: [.sortedKeys])
            .write(to: folder.appendingPathComponent("tokenizer.json"))
        let config: [String: Any] = ["tokenizer_class": "LlamaTokenizer", "bos_token": "<s>",
                                     "eos_token": "<|im_end|>", "unk_token": "<unk>",
                                     "clean_up_tokenization_spaces": true, "model_max_length": 4096]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: folder.appendingPathComponent("tokenizer_config.json"))
        return folder
    }

    @Test func sentencePieceByteFallbackAndCleanUp() async throws {
        let folder = try Self.sentencePieceFixture()
        defer { try? FileManager.default.removeItem(at: folder) }
        let tok = try await MFTokenizer.load(from: folder, family: nil)
        let t = tok.tokenizer
        func id(_ piece: String) throws -> Int32 { Int32(try #require(t.convertTokenToId(piece), "\(piece)")) }
        // The rules this fixture exists for, checked on the full decode first.
        #expect(t.decode(tokens: [try id("▁hello"), try id("▁.")].map(Int.init), skipSpecialTokens: true)
                == "hello.")
        #expect(t.decode(tokens: [try id("▁do"), try id("▁n't")].map(Int.init), skipSpecialTokens: true)
                == "don't")
        // A byte-fallback run decodes once a piece follows it (the library
        // drops a trailing run; `MFDetokenizer` holds such runs back).
        #expect(t.decode(tokens: [0xE4, 0xB8, 0xAD].map { 3 + $0 } + [Int(try id("▁ok"))], skipSpecialTokens: true)
                == "中 ok")
        #expect(MFDetokenizer.isByteFallback("<0xE4>"))

        // Crafted runs: clean-up across pushes, byte-fallback scalars split
        // and invalid, leading spaces, specials in between, at every offset
        // from the window's edge.
        let bait: [[Int32]] = [
            [try id("▁hello"), try id("▁.")], [try id("▁do"), try id("▁n't")], [try id("▁it"), try id("▁'s")],
            [try id("▁'"), try id("▁"), try id("s")], [try id("▁a"), try id("▁,"), try id("▁!"), try id("▁?")],
            [try id("▁it"), try id("▁'ve")], [try id("▁it"), try id("▁'re")], [try id("▁it"), try id("▁'m")],
            [0xE4, 0xB8, 0xAD].map { Int32(3 + $0) }, [0xF0, 0x9F, 0x99, 0x82].map { Int32(3 + $0) },
            [0xE4, 0xB8].map { Int32(3 + $0) }, [0xFF, 0xC0, 0x80].map { Int32(3 + $0) },
            [try id("▁")], [try id("▁▁▁")], [Int32(t.convertTokenToId("<|im_end|>") ?? 0)],
            [try id("ห"), try id("้"), try id("า"), try id("ม")], [try id("▁café"), try id("é")],
        ]
        var g = Generator(state: 0x5E)
        for offset in 0..<20 {
            var ids: [Int32] = (0..<(offset + 1)).map { _ in try! id("▁ok") }
            for _ in 0..<400 { ids += bait[g.below(bait.count)] }
            Self.compare(tok, ids, label: "SentencePiece bait, offset \(offset)")
        }
        Self.run(tok, name: "SentencePiece (byte fallback, clean-up)", randomLength: 3_000, seed: 0x55)
    }

    // MARK: The installs' tokenizers (opt-in)

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_DETOK_REAL_TOKENIZERS"] != nil,
                   "needs MFERENCE_DETOK_REAL_TOKENIZERS"))
    func installedTokenizers() async throws {
        let folders = (ProcessInfo.processInfo.environment["MFERENCE_DETOK_REAL_TOKENIZERS"] ?? "")
            .split(separator: ":").map { URL(fileURLWithPath: String($0)) }
        for (i, folder) in folders.enumerated() {
            let tok = try await MFTokenizer.load(from: folder, family: nil)
            Self.run(tok, name: folder.path.components(separatedBy: "/").suffix(2).joined(separator: "/"),
                     randomLength: 2_500, seed: 0x60 + UInt64(i))
        }
    }
}
