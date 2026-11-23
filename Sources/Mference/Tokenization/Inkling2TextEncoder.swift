import Foundation
import Tokenizers

/// Text to token ids the way the reference tokenizes Inkling 2.
///
/// The reference (llama.cpp b12641, `src/llama-vocab.cpp`, pre-tokenizer
/// `inkling2`) splits text with the GPT-4o pattern and runs byte-level BPE
/// with `ignore_merges`: a pre-tokenized word that is itself a vocabulary
/// entry becomes that one token before any merge is tried (#32799). The
/// vendor's `tokenizer.json` sets the same flag. swift-transformers 1.3
/// ignores it, so this encoder applies it: added (special) tokens are matched
/// first, the rest is split with the same pattern, and each word is looked
/// up whole in the vocabulary, falling back to swift-transformers' BPE for
/// that word alone.
struct Inkling2TextEncoder: @unchecked Sendable {
    /// GPT-4o's split pattern, as llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GPT4O`
    /// quotes it from `tokenizer.json`.
    static let pattern =
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private let words: NSRegularExpression
    private let specialPattern: NSRegularExpression
    /// Added tokens by text (every `<|...|>` the vocabulary marks special).
    private let specials: [String: Int]

    init(specials: [String: Int]) throws {
        self.words = try NSRegularExpression(pattern: Self.pattern)
        self.specials = specials
        // Longest first, so a token never matches a prefix of a longer one.
        let alternatives = specials.keys.sorted { $0.count != $1.count ? $0.count > $1.count : $0 < $1 }
            .map(NSRegularExpression.escapedPattern(for:))
        self.specialPattern = try NSRegularExpression(pattern: alternatives.joined(separator: "|"))
    }

    /// GPT-2's byte-to-character table of byte-level BPE: printable bytes
    /// map to themselves, the others to U+0100 onward in byte order.
    static let byteToCharacter: [Character] = {
        let printable = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var table: [Character] = []
        var next: UInt32 = 0
        for byte in 0..<256 {
            if printable.contains(byte) {
                table.append(Character(Unicode.Scalar(UInt32(byte))!))
            } else {
                table.append(Character(Unicode.Scalar(256 + next)!))
                next += 1
            }
        }
        return table
    }()

    static func byteLevel(_ word: String) -> String {
        String(word.utf8.map { byteToCharacter[Int($0)] })
    }

    func encode(_ text: String, tokenizer: any Tokenizer) -> [Int] {
        var ids: [Int] = []
        let ns = text as NSString
        var cursor = 0
        for match in specialPattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            if match.range.location > cursor {
                encodeText(ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor)),
                           tokenizer: tokenizer, into: &ids)
            }
            ids.append(specials[ns.substring(with: match.range)]!)
            cursor = match.range.location + match.range.length
        }
        if cursor < ns.length {
            encodeText(ns.substring(from: cursor), tokenizer: tokenizer, into: &ids)
        }
        return ids
    }

    private func encodeText(_ text: String, tokenizer: any Tokenizer, into ids: inout [Int]) {
        let ns = text as NSString
        for match in words.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let word = ns.substring(with: match.range)
            let mapped = Self.byteLevel(word)
            if let id = tokenizer.convertTokenToId(mapped), tokenizer.convertIdToToken(id) == mapped {
                ids.append(id)
            } else {
                ids += tokenizer.encode(text: word, addSpecialTokens: false)
            }
        }
    }
}
