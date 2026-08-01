import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// Four challenges drive the design:
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
/// 4. `Tokenizer.decode` applies `cleanUpTokenizationSpaces`, and it defaults to
///    **true** when `tokenizer_config.json` omits the key. Gemma omits it; Qwen
///    sets it false. Its rewrites delete a space an earlier decode already
///    emitted (" 's" -> "'s"), so on Gemma the decode genuinely is not
///    append-only and no amount of holding back can make it so.
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
///   - When a decode contradicts text already handed out, the resync path emits
///     the divergent tail minus the part the caller has already seen, instead of
///     dropping it. The stale text cannot be retracted; losing characters on top
///     of that is strictly worse. On Gemma this is the difference between
///     streaming "do n't" and streaming "do n'".
///
/// `flush()` is terminal in the generation loop, but pushing afterwards is not a
/// trap: the committed prefix is clamped so that it can never rewind.
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
        // The decode is *not* guaranteed to extend, byte-wise or otherwise, and
        // two separate things break it:
        //
        //   - A byte-level BPE token can end mid-scalar, and the decoder renders
        //     the dangling bytes as U+FFFD; the next decode replaces those bytes
        //     outright. Withholding the trailing U+FFFD run removes this case.
        //   - `Tokenizer.decode` applies `cleanUpTokenizationSpaces`, which
        //     defaults to true when the key is absent from `tokenizer_config.json`
        //     — as it is for Gemma. Its rewrites delete a space an earlier decode
        //     already emitted (" 's" -> "'s"), so on that path the decode really
        //     can contradict what was handed out. Nothing here can prevent that;
        //     the resync branch keeps the damage to the retracted space.
        let cur = current.utf8
        // Never withhold text already handed out. `flush()` releases the withheld
        // run, so afterwards `emitted` can itself end with U+FFFD; without this
        // clamp a subsequent `push` would re-withhold that run, shrink `emitted`,
        // and hand the caller a duplicate on the following token. Clamping to
        // `emitted`'s length keeps the committed prefix monotonic under any call
        // order, and lands on a scalar boundary because `emitted` is one.
        let held = min(holdingPartialScalar ? Self.trailingReplacementByteCount(current) : 0,
                       max(0, cur.count - emitted.utf8.count))
        let committed = cur.dropLast(held)
        // `current` itself when nothing is withheld: no copy on the hot path.
        func committedText() -> String {
            held == 0 ? current : String(decoding: committed, as: UTF8.self)
        }

        guard cur.starts(with: emitted.utf8) else {
            // The decode contradicts what was already handed out (see the
            // clean-up note above). The stale text cannot be retracted, but
            // dropping the divergent tail loses characters outright — "do n't"
            // decodes to "don't" and the "t" simply never reaches the caller.
            // Emit the tail past the common prefix instead: worst case the
            // caller sees a space that clean-up would have removed.
            let shared = Self.sharedScalarAlignedPrefix(cur, emitted.utf8)
            let newTail = committed.dropFirst(shared)
            // The rewrite deleted a space *inside* the divergent region, so the
            // new tail usually restates characters already shown ("it '" becomes
            // "it's": the apostrophe moved, it is not new). Skip the part that
            // the caller has already seen, or the stream gains "it ''s".
            let overlap = Self.overlapCount(shown: emitted.utf8.dropFirst(shared),
                                            tail: newTail)
            let tail = String(decoding: newTail.dropFirst(overlap), as: UTF8.self)
            emitted = committedText()
            return tail
        }
        let delta = String(decoding: committed.dropFirst(emitted.utf8.count), as: UTF8.self)
        emitted = committedText()
        return delta
    }

    /// Byte length of the longest common prefix of `a` and `b`, backed off to a
    /// UTF-8 scalar boundary so slicing `a` at it cannot split a codepoint.
    /// Only the resync branch needs this, so the linear walk is off the hot path.
    @usableFromInline
    static func sharedScalarAlignedPrefix(_ a: String.UTF8View, _ b: String.UTF8View) -> Int {
        var matched = 0
        var lastBoundary = 0
        for (x, y) in zip(a, b) {
            guard x == y else { break }
            // Continuation bytes are 0b10xxxxxx; anything else starts a scalar.
            if x & 0xC0 != 0x80 { lastBoundary = matched }
            matched += 1
        }
        let splitsAScalar = a.dropFirst(matched).first.map { $0 & 0xC0 == 0x80 } ?? false
        return splitsAScalar ? lastBoundary : matched
    }

    /// Largest `k` where the first `k` bytes of `tail` are also the last `k`
    /// bytes of `shown`, i.e. how much of the rewritten tail the caller has
    /// already been given. Candidates that would split a scalar are rejected.
    ///
    /// Both slices start at the first byte where the two decodes disagree, and a
    /// clean-up rewrite disagrees only within its own match (at most four bytes),
    /// so the quadratic scan is over a handful of bytes.
    @usableFromInline
    static func overlapCount(shown: String.UTF8View.SubSequence,
                             tail: String.UTF8View.SubSequence) -> Int {
        var k = min(shown.count, tail.count)
        while k > 0 {
            let splitsAScalar = tail.dropFirst(k).first.map { $0 & 0xC0 == 0x80 } ?? false
            if !splitsAScalar, shown.suffix(k).elementsEqual(tail.prefix(k)) { return k }
            k -= 1
        }
        return 0
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
        // Not `String(bytes:encoding:)`: that returns nil for the whole buffer
        // if any part of it is invalid, so a tail holding a complete codepoint
        // followed by a partial one would discard both. Decoding lossily keeps
        // the complete codepoints and renders the remainder as U+FFFD, which is
        // exactly what the byte-level path produces for the same situation.
        return String(decoding: bytes, as: UTF8.self)
    }

    @usableFromInline
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }
}
