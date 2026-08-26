import Foundation

/// Which pages a decode token attends: StreamingLLM-style sink pages, the
/// recent window (including the unsealed tail), and the top-k sealed pages by
/// Quest criticality score. Pure policy — no storage, no GPU.
public struct KVPageSelector: Sendable, Equatable {
    public let sinkPages: Int
    public let recentPages: Int
    public let topKPages: Int

    public init(sinkPages: Int = 2, recentPages: Int = 4, topKPages: Int = 60) {
        precondition(sinkPages >= 0 && recentPages >= 1 && topKPages >= 0,
                     "invalid selector parameters (recent must cover the tail)")
        self.sinkPages = sinkPages
        self.recentPages = recentPages
        self.topKPages = topKPages
    }

    public struct Selection: Equatable, Sendable {
        /// Ascending page indices.
        public let pages: [Int]
        /// Selected logical token count; the tail page contributes only its
        /// valid rows.
        public let selTokens: Int
    }

    /// - Parameters:
    ///   - scores: Quest score per *sealed* page, from the previous token's
    ///     query. Empty on the first decode token after a prefill (warmup:
    ///     sinks + recent only, plus trailing fill up to the top-k budget).
    ///     May cover fewer pages than `sealedPages` — a page that sealed on
    ///     the previous token has no score yet; unscored trailing pages are
    ///     recent by construction and the recent window covers them.
    ///   - sealedPages: pages fully written (64 valid tokens each).
    ///   - tailValidTokens: valid rows in the unsealed tail page (up to a
    ///     full 64 for a page written but not yet sealed by `advance`); 0
    ///     when the position sits exactly on a page boundary.
    public func select(scores: [Float],
                       sealedPages: Int,
                       tailValidTokens: Int) -> Selection {
        precondition(tailValidTokens >= 0 && tailValidTokens <= 64,
                     "tailValidTokens must be 0...64")
        let totalPages = sealedPages + (tailValidTokens > 0 ? 1 : 0)
        guard totalPages > 0 else { return Selection(pages: [], selTokens: 0) }

        var picked = Set<Int>()
        for page in 0..<min(sinkPages, totalPages) { picked.insert(page) }
        for page in max(0, totalPages - recentPages)..<totalPages { picked.insert(page) }

        if topKPages > 0 {
            if scores.isEmpty {
                // Warmup: no query yet — fill the budget with the most recent
                // sealed pages, the best context-free prior.
                var page = totalPages - 1
                var budget = topKPages
                while budget > 0 && page >= 0 {
                    if picked.insert(page).inserted { budget -= 1 }
                    page -= 1
                }
            } else {
                // Deterministic top-k: score desc, index asc on ties.
                let scoredPages = min(scores.count, sealedPages)
                let candidates = (0..<scoredPages)
                    .filter { !picked.contains($0) }
                    .sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
                for page in candidates.prefix(topKPages) { picked.insert(page) }
            }
        }

        let pages = picked.sorted()
        let tailIndex = tailValidTokens > 0 ? totalPages - 1 : -1
        let selTokens = pages.reduce(0) { acc, page in
            acc + (page == tailIndex ? tailValidTokens : 64)
        }
        return Selection(pages: pages, selTokens: selTokens)
    }

    /// True when `select` over `totalPages` pages provably picks every page
    /// no matter what the scores contain, provided at most
    /// `maxUnscoredSealedPages` trailing sealed pages lack Quest scores
    /// (plain decode lags one page; speculative rounds can lag by their
    /// span). Gap pages — neither sink nor recent — are picked by top-k
    /// only when scored, so any unscored page must sit inside the recent
    /// window for coverage to be unconditional.
    public func coversEntireContext(totalPages: Int,
                                    maxUnscoredSealedPages: Int) -> Bool {
        guard totalPages > 0 else { return true }
        let sinks = min(sinkPages, totalPages)
        let recentStart = max(0, totalPages - recentPages)
        let unionCount = sinks + (totalPages - recentStart)
            - max(0, sinks - recentStart)
        let gap = totalPages - unionCount
        if gap == 0 { return true }
        return gap <= topKPages && recentPages >= 1 + maxUnscoredSealedPages
    }
}
