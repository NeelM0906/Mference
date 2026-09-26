import Foundation
import Hub
import Tokenizers

/// The pinned Gemma 4 detokenization pipeline, without HF's cleanup pass.
/// Ported from TurboFieldfare (`GemmaDecoding`, drumih/turbo-fieldfare#118).
///
/// `tokenizer.json` declares `decoder: Sequence[Replace("▁" -> " "),
/// ByteFallback, Fuse]`. swift-transformers runs that sequence and then applies
/// `clean_up_tokenization_spaces`, which defaults to **true** when the key is
/// absent — and Gemma's `tokenizer_config.json` omits it.
///
/// That cleanup pass is a legacy heuristic for whitespace tokenizers and is
/// wrong for a metaspace + byte-fallback BPE, which is lossless by
/// construction. It rewrites text the model deliberately produced:
///
/// ```text
/// "he said ' ok ' now"  ->  "he said'ok'now"
/// "step 1 . done"       ->  "step 1. done"
/// "Bonjour !"           ->  "Bonjour!"
/// ```
///
/// So `decode(encode(x)) != x`, and because the pass runs over the whole string
/// on every call, appending a token can rewrite text already streamed. This
/// type reproduces the declared decoder sequence and stops there. Batch decode
/// (`MFTokenizer.decode`) and streaming (`MFDetokenizer`) both go through it,
/// so they agree by construction, and streaming costs O(1) per token.
enum GemmaDecoding {
    /// `<0xXX>` byte-fallback token, e.g. `<0xE2>`.
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }

    static func byteValue(_ token: String) -> UInt8? {
        guard isByteFallback(token) else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }

    /// One token's contribution to the output: the `Replace` decoder's
    /// `"▁" -> " "` substitution. Byte-fallback tokens are handled by
    /// `ByteFallbackRun` instead — they only decode as a complete run.
    static func fragment(_ token: String) -> String {
        // Runs once per generated token; most tokens contain no "▁", and the
        // guard returns them without Foundation bridging or an allocation.
        guard token.unicodeScalars.contains(sentencePieceUnderline) else { return token }
        var scalars = String.UnicodeScalarView()
        for scalar in token.unicodeScalars {
            scalars.append(scalar == sentencePieceUnderline ? " " : scalar)
        }
        return String(scalars)
    }

    private static let sentencePieceUnderline: Unicode.Scalar = "\u{2581}"

    /// Whether `tokenizer.json` declares exactly the sequence this type
    /// reproduces. Checked structurally, so a foreign decoder is never decoded
    /// with these rules and a benign library change cannot break loading.
    static func declaresPinnedDecoder(_ tokenizerData: Config) -> Bool {
        let decoder = tokenizerData["decoder"]
        let steps = decoder.decoders.array(or: [])
        return decoder.type.string() == "Sequence"
            && steps.count == 3
            && steps[0].type.string() == "Replace"
            && steps[0].pattern.String.string() == "▁"
            && steps[0].content.string() == " "
            && steps[1].type.string() == "ByteFallback"
            && steps[2].type.string() == "Fuse"
    }

    /// The `added_tokens[special == true]` IDs: exactly what the library's
    /// `decode(skipSpecialTokens: true)` removes before its decoder chain.
    static func specialTokenIDs(_ tokenizerData: Config) -> Set<Int32> {
        var specials: Set<Int32> = []
        for added in tokenizerData["addedTokens"].array(or: []) {
            guard added["special"].boolean(or: false),
                  let id = added["id"].integer(), let value = Int32(exactly: id) else { continue }
            specials.insert(value)
        }
        return specials
    }
}

/// A run of `<0xXX>` byte-fallback tokens, decoded with the reference HF
/// `tokenizers` ByteFallback semantics: the run commits as a whole — valid
/// UTF-8 becomes its text, anything else (including an incomplete trailing
/// sequence) becomes one U+FFFD per byte of the run.
///
/// Streaming: while the run is valid so far, nothing can be emitted, because
/// one more byte can invalidate the whole run retroactively. The moment the
/// run becomes invalid its fate is sealed — every byte, past and future,
/// decodes to exactly one U+FFFD — so an invalid run streams replacement
/// characters live rather than freezing the stream.
struct ByteFallbackRun {
    private var bytes: [UInt8] = []
    private var poisoned = false
    /// Continuation bytes still owed by the current lead, and the allowed
    /// range for the next one (RFC 3629 constrains the first continuation
    /// after E0/ED/F0/F4 leads).
    private var pendingContinuations = 0
    private var nextContinuation: ClosedRange<UInt8> = 0x80...0xBF

    /// Text this byte contributes to the stream: `""` while the run is valid
    /// so far, replacement characters once it can no longer become valid.
    mutating func push(_ byte: UInt8) -> String {
        if poisoned { return "\u{FFFD}" }
        bytes.append(byte)
        if accept(byte) { return "" }
        poisoned = true
        defer { bytes.removeAll(keepingCapacity: true) }
        return String(repeating: "\u{FFFD}", count: bytes.count)
    }

    /// Close the run: a non-byte token follows, or the stream ends.
    mutating func commit() -> String {
        defer {
            bytes.removeAll(keepingCapacity: true)
            poisoned = false
            pendingContinuations = 0
            nextContinuation = 0x80...0xBF
        }
        guard !poisoned, !bytes.isEmpty else { return "" }
        return String(bytes: bytes, encoding: .utf8)
            ?? String(repeating: "\u{FFFD}", count: bytes.count)
    }

    /// Advance the incremental UTF-8 validator. `false` means no suffix can
    /// ever make the run valid again.
    private mutating func accept(_ byte: UInt8) -> Bool {
        if pendingContinuations > 0 {
            guard nextContinuation.contains(byte) else { return false }
            pendingContinuations -= 1
            nextContinuation = 0x80...0xBF
            return true
        }
        switch byte {
        case 0x00...0x7F: break
        case 0xC2...0xDF: pendingContinuations = 1
        case 0xE0: pendingContinuations = 2; nextContinuation = 0xA0...0xBF
        case 0xE1...0xEC, 0xEE, 0xEF: pendingContinuations = 2
        case 0xED: pendingContinuations = 2; nextContinuation = 0x80...0x9F
        case 0xF0: pendingContinuations = 3; nextContinuation = 0x90...0xBF
        case 0xF1...0xF3: pendingContinuations = 3
        case 0xF4: pendingContinuations = 3; nextContinuation = 0x80...0x8F
        default: return false // stray continuation, overlong lead, or > U+10FFFF
        }
        return true
    }
}
