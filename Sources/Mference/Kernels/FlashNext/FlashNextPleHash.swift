import Foundation

/// The `Qwen4ExpTextNGramEmbedding` hash: token ids in, n-gram table row indices
/// out. Sixteen rows per token, in exact 64-bit integer arithmetic.
///
/// # Why the integer width is load-bearing
///
/// `mix_n` consumes the full positive `Int64` range by construction, so a
/// `Double` intermediate (53-bit significand) or a 32-bit one selects a
/// *different* row, and nothing downstream can detect it — the gathered
/// embedding is simply wrong, plausibly shaped, and silent. Everything here is
/// `UInt64` wrapping arithmetic; the residues are non-negative because the mix
/// stays below 2^63.
///
/// # Never re-derive the constants
///
/// `layer_multipliers` (I64 [3]) and `ngram_heads_{offsets,vocab_sizes}`
/// (I64 [16]) are carried byte-for-byte by the installer and loaded typed. The
/// design doc gives the splitmix64 recipe that produced them; that recipe is
/// documentation, not an implementation. This type takes the loaded values.
struct FlashNextPleHash {

    let multipliers: [Int64]
    let headOffsets: [Int64]
    let headVocabSizes: [Int64]
    let eosTokenID: Int

    /// `ngram_size`; also the conv dilation and one more than the id history.
    var ngramSize: Int { multipliers.count }
    var ngramHeads: Int { headOffsets.count }
    /// `context_len` — how many previous ids a decode step has to carry.
    var historyLength: Int { max(0, ngramSize - 1) }

    init(multipliers: [Int64], headOffsets: [Int64], headVocabSizes: [Int64],
         eosTokenID: Int) {
        precondition(!multipliers.isEmpty, "ngram_size must be positive")
        precondition(headOffsets.count == headVocabSizes.count,
                     "n-gram head offsets and vocab sizes must agree in length")
        precondition(headVocabSizes.allSatisfy { $0 > 0 },
                     "n-gram head vocab sizes must be positive")
        self.multipliers = multipliers
        self.headOffsets = headOffsets
        self.headVocabSizes = headVocabSizes
        self.eosTokenID = eosTokenID
    }

    /// The initial id history for a fresh sequence: `context_len` copies of the
    /// EOS id, NOT zeros.
    ///
    /// The reference's conv-state cache left-pads a short prefill with zeros, so
    /// the PLE call site pads the id stream with `eos_token_id` *before* handing
    /// it over — padding ids with 0 would hash as token 0. Verified against the
    /// pinned `transformers` `cache_utils.update_conv_state`.
    func initialHistory() -> [Int] {
        [Int](repeating: eosTokenID, count: historyLength)
    }

    /// Absolute row indices into the padded n-gram table, one list of
    /// `ngramHeads` per position of `window`.
    ///
    /// `window` is `history + tokens`: the decode path reads its cached history
    /// before overwriting it, and the EOS-aware shift runs over that window
    /// only. Only the last `tokens.count` entries are the caller's; the leading
    /// `historyLength` exist to give the shifts something to look back at.
    func rowIDs(window: [Int]) -> [[Int]] {
        let n = window.count
        guard n > 0 else { return [] }
        var shifted: [[Int]] = []
        shifted.reserveCapacity(ngramSize)
        for s in 0..<ngramSize { shifted.append(shiftRightIgnoringEOS(window, by: s)) }

        var rows = [[Int]](repeating: [], count: n)
        let perOrder = ngramHeads / max(1, ngramSize - 1)
        for order in 2...max(2, ngramSize) {
            let start = (order - 2) * perOrder
            for i in 0..<n {
                var mix: UInt64 = 0
                for p in 0..<order {
                    mix ^= UInt64(bitPattern: Int64(shifted[p][i]))
                        &* UInt64(bitPattern: multipliers[p])
                }
                for h in 0..<perOrder {
                    let head = start + h
                    let residue = mix % UInt64(headVocabSizes[head])
                    rows[i].append(Int(Int64(residue) + headOffsets[head]))
                }
            }
        }
        return rows
    }

    /// `_shift_right_ignore_eos`: look back `shift` tokens, but never across an
    /// EOS and never before position 0 — either violation reads `eos_token_id`.
    /// The boundary uses a strict `<`, so the EOS token is the last token of its
    /// own segment rather than the first of the next.
    func shiftRightIgnoringEOS(_ ids: [Int], by shift: Int) -> [Int] {
        if shift == 0 { return ids }
        var out = [Int](repeating: eosTokenID, count: ids.count)
        var lastEOS = -1
        for i in 0..<ids.count {
            let segmentStart = lastEOS + 1          // strictly before i
            let source = i - shift
            if i - segmentStart >= shift, source >= 0 { out[i] = ids[source] }
            if ids[i] == eosTokenID { lastEOS = i }
        }
        return out
    }
}
