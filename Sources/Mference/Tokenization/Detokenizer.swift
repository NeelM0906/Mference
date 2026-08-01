import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// Three challenges drive the design:
///
/// 1. BPE byte-fallback splits multi-byte codepoints (e.g. emoji) across several
///    tokens. Naively decoding each token in isolation yields broken UTF-8.
/// 2. swift-transformers' decoder silently drops byte-fallback tokens that sit
///    at the **end** of the decoded sequence (the bytes are committed only once
///    a non-byte-fallback token follows). For us this matters at `flush()`.
/// 3. Byte-level BPE (Qwen) has no byte-fallback tokens at all: a scalar's UTF-8
///    bytes simply span ordinary tokens, and `ByteLevelDecoder` ends with
///    `String(decoding:as:)`, so a decode stopping mid-scalar already contains
///    U+FFFD. Nothing upstream of us can hold those bytes back.
///
/// Strategy:
///   - During `push(_:)` we decode the longest prefix of accumulated IDs that
///     does NOT end with byte-fallback tokens, then emit the delta vs. previously
///     emitted text. Any trailing byte-fallback IDs are held back.
///   - Also during `push(_:)`, a trailing run of U+FFFD in the decoded text is
///     withheld: it is the decoder's rendering of a scalar whose remaining bytes
///     are still in flight, and the next decode replaces it with the real
///     character. Interior U+FFFD is emitted normally.
///   - During `flush()` we decode the stable prefix as above AND manually
///     assemble the trailing byte-fallback bytes into a UTF-8 string. This
///     recovers text the library would otherwise drop on a sequence-ending
///     codepoint. `flush()` also releases any withheld U+FFFD, so a replacement
///     character the model genuinely produced — or a stream that really did stop
///     mid-scalar — still reaches the caller.
struct MFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    @usableFromInline var stableIDs: [Int] = []
    @usableFromInline var trailingByteIDs: [Int] = []
    @usableFromInline var emitted: String = ""

    init(tokenizer: MFTokenizer) {
        self.tokenizer = tokenizer.tokenizer
    }

    mutating func push(_ id: Int32) -> String {
        let tokenID = Int(id)
        let token = tokenizer.convertIdToToken(tokenID) ?? ""
        if Self.isByteFallback(token) {
            trailingByteIDs.append(tokenID)
            return ""
        }

        if !trailingByteIDs.isEmpty {
            stableIDs.append(contentsOf: trailingByteIDs)
            trailingByteIDs.removeAll(keepingCapacity: true)
        }
        stableIDs.append(tokenID)

        let current = tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)
        return commitDelta(current)
    }

    mutating func flush() -> String {
        let stableText = stableIDs.isEmpty
            ? ""
            : tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)

        let trailingText = assembleByteFallback(trailingByteIDs)
        let fullText = stableText + trailingText
        return commitDelta(fullText, holdingPartialScalar: false)
    }

    @usableFromInline
    mutating func commitDelta(_ current: String, holdingPartialScalar: Bool = true) -> String {
        // Compare UTF-8 bytes, not graphemes. `hasPrefix` / `dropFirst` work on
        // extended grapheme clusters, so a token boundary that splits a cluster
        // — a Thai consonant in one token, its tone mark in the next — makes
        // "ห้าม" fail the prefix test against "ห" and the resync path below
        // swallows the delta. The views are borrowed, not copied: this runs per
        // token.
        //
        // The decode is *not* guaranteed to extend byte-wise either. A byte-level
        // BPE token can end mid-scalar, and the decoder renders the dangling
        // bytes as U+FFFD; the next decode replaces those bytes outright. Holding
        // the trailing U+FFFD run back keeps `emitted` a genuine byte prefix of
        // every later decode, which is what makes the guard below meaningful.
        let held = holdingPartialScalar ? Self.trailingReplacementByteCount(current) : 0
        let cur = current.utf8
        let committed = cur.dropLast(held)
        // `current` itself when nothing is withheld: no copy on the hot path.
        func committedText() -> String {
            held == 0 ? current : String(decoding: committed, as: UTF8.self)
        }

        guard cur.starts(with: emitted.utf8) else {
            // Decoder altered the prefix — extremely rare in append-only streams.
            // Resync rather than emit garbage; the user-visible loss is bounded
            // to whatever was retokenized.
            emitted = committedText()
            return ""
        }
        let delta = String(decoding: committed.dropFirst(emitted.utf8.count), as: UTF8.self)
        emitted = committedText()
        return delta
    }

    /// UTF-8 length of the trailing run of U+FFFD in `text`, which is how the
    /// decoder renders bytes it could not (yet) complete into a scalar.
    /// Zero for every decode that ends on a whole character — the common case,
    /// kept allocation-free by walking the borrowed scalar view backwards.
    @usableFromInline
    static func trailingReplacementByteCount(_ text: String) -> Int {
        let scalars = text.unicodeScalars
        var index = scalars.endIndex
        var bytes = 0
        while index > scalars.startIndex {
            let previous = scalars.index(before: index)
            guard scalars[previous] == "\u{FFFD}" else { break }
            bytes += 3  // U+FFFD is EF BF BD.
            index = previous
        }
        return bytes
    }

    @usableFromInline
    func assembleByteFallback(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count)
        for id in ids {
            guard let tok = tokenizer.convertIdToToken(id) else { continue }
            guard Self.isByteFallback(tok),
                  let byte = UInt8(tok.dropFirst(3).dropLast(), radix: 16)
            else { continue }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    @usableFromInline
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }
}
