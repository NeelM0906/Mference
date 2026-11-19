import Foundation

extension Native {
    /// Bookkeeping for the physical pages of a `PagedKVStore` arena.
    ///
    /// A page has two kinds of holder: the live sequence (at most once, in
    /// one logical slot of its page table) and the prefix cache (any number
    /// of references: radix-tree nodes, an entry's private tail page, an
    /// entry's recurrent-snapshot pages, transient staging). A page is free
    /// when it has neither. Pages held by the cache and not by the sequence
    /// are the cache's RAM (`cacheOnlyCount`); pages shared with the sequence
    /// are charged to the sequence's KV arena.
    ///
    /// Free pages are handed out from the top of the arena down, so a page
    /// table is never the identity (the same choice as the reference store:
    /// it catches a reader that ignores the table).
    struct MemoryPagePool {
        let total: Int
        private var free: [Int32]
        private var sequenceHeld: [Bool]
        private var cacheRefs: [Int32]
        private(set) var cacheOnlyCount = 0
        /// Unheld pages waiting for a command buffer before they are free.
        private(set) var quarantinedCount = 0

        init(total: Int) {
            self.total = total
            free = (0..<Int32(total)).map { $0 }
            sequenceHeld = [Bool](repeating: false, count: total)
            cacheRefs = [Int32](repeating: 0, count: total)
        }

        var freeCount: Int { free.count }
        var usedCount: Int { total - free.count }

        func isSequenceHeld(_ p: Int) -> Bool { sequenceHeld[p] }
        func cacheReferences(_ p: Int) -> Int { Int(cacheRefs[p]) }

        /// A free page for the sequence, or nil when none is free.
        mutating func allocateForSequence() -> Int? {
            guard let p = free.popLast() else { return nil }
            sequenceHeld[Int(p)] = true
            return Int(p)
        }

        /// A free page with one cache reference, or nil when none is free.
        mutating func allocateForCache() -> Int? {
            guard let p = free.popLast() else { return nil }
            cacheRefs[Int(p)] = 1
            cacheOnlyCount += 1
            return Int(p)
        }

        /// The sequence starts holding a page the cache already holds.
        mutating func sequenceAcquire(_ p: Int) {
            precondition(!sequenceHeld[p] && cacheRefs[p] > 0, "page \(p) is not a cached page")
            sequenceHeld[p] = true
            cacheOnlyCount -= 1
        }

        /// The sequence stops holding `p`. When no holder is left the page
        /// is freed, or, with `quarantine`, left unheld and not free (the
        /// caller frees it with `releaseQuarantined` once the command buffer
        /// that may still use it has completed). Returns true when the page
        /// went into quarantine.
        @discardableResult
        mutating func sequenceRelease(_ p: Int, quarantine: Bool = false) -> Bool {
            precondition(sequenceHeld[p], "page \(p) is not held by the sequence")
            sequenceHeld[p] = false
            if cacheRefs[p] > 0 {
                cacheOnlyCount += 1
                return false
            }
            return unheld(p, quarantine: quarantine)
        }

        mutating func cacheRetain(_ p: Int) {
            precondition(sequenceHeld[p] || cacheRefs[p] > 0, "page \(p) is free")
            // A page the sequence holds stays charged to the sequence while shared.
            cacheRefs[p] += 1
        }

        /// Drops one cache reference (see `sequenceRelease` for `quarantine`).
        @discardableResult
        mutating func cacheRelease(_ p: Int, quarantine: Bool = false) -> Bool {
            precondition(cacheRefs[p] > 0, "page \(p) has no cache reference")
            cacheRefs[p] -= 1
            guard cacheRefs[p] == 0, !sequenceHeld[p] else { return false }
            cacheOnlyCount -= 1
            return unheld(p, quarantine: quarantine)
        }

        private mutating func unheld(_ p: Int, quarantine: Bool) -> Bool {
            if quarantine {
                quarantinedCount += 1
                return true
            }
            free.append(Int32(p))
            return false
        }

        /// Frees a page `sequenceRelease` or `cacheRelease` quarantined.
        mutating func releaseQuarantined(_ p: Int) {
            precondition(!sequenceHeld[p] && cacheRefs[p] == 0 && quarantinedCount > 0, "page \(p) is not quarantined")
            quarantinedCount -= 1
            free.append(Int32(p))
        }
    }
}
