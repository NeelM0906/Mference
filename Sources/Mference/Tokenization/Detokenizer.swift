import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// Two challenges drive the design:
///
/// 1. BPE byte-fallback splits multi-byte codepoints (e.g. emoji) across several
///    tokens. Naively decoding each token in isolation yields broken UTF-8.
/// 2. swift-transformers' decoder silently drops byte-fallback tokens that sit
///    at the **end** of the decoded sequence (the bytes are committed only once
///    a non-byte-fallback token follows). For us this matters at `flush()`.
///
/// Strategy:
///   - During `push(_:)` we decode the longest prefix of accumulated IDs that
///     does NOT end with byte-fallback tokens, then emit the delta vs. previously
///     emitted text. Any trailing byte-fallback IDs are held back.
///   - During `flush()` we decode the stable prefix as above AND manually
///     assemble the trailing byte-fallback bytes into a UTF-8 string. This
///     recovers text the library would otherwise drop on a sequence-ending
///     codepoint.
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
        return commitDelta(fullText)
    }

    @usableFromInline
    mutating func commitDelta(_ current: String) -> String {
        // Compare UTF-8 bytes, not graphemes. `hasPrefix` / `dropFirst` work on
        // extended grapheme clusters, so a token boundary that splits a cluster
        // — a Thai consonant in one token, its tone mark in the next — makes
        // "ห้าม" fail the prefix test against "ห" and the resync path below
        // swallows the delta. Byte-wise an append-only decode is always a strict
        // prefix. The views are borrowed, not copied: this runs per token.
        let cur = current.utf8
        guard cur.starts(with: emitted.utf8) else {
            // Decoder altered the prefix — extremely rare in append-only streams.
            // Resync rather than emit garbage; the user-visible loss is bounded
            // to whatever was retokenized.
            emitted = current
            return ""
        }
        let delta = String(decoding: cur.dropFirst(emitted.utf8.count), as: UTF8.self)
        emitted = current
        return delta
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
