import Foundation
import Metal

extension Native {
    /// The prefix cache's index: a radix tree over token ids whose edges are
    /// whole KV pages (`tokensPerPage` tokens each).
    ///
    /// - A node at depth `d` stands for the first `d * tokensPerPage` tokens
    ///   of every prefix below it. Its `page`, when present, is the physical
    ///   KV page holding that node's tokens, shared by every cached prefix
    ///   below it and by the live sequence when it attached one of them.
    /// - A cached prefix (`MemoryPrefixEntry`) hangs at the node of its last
    ///   whole page, with the remaining `< tokensPerPage` tokens as its
    ///   `tail`. Its partial last page is private to it (copied at insert),
    ///   so the sequence can keep appending to its own copy.
    /// - Every entry has a recurrent snapshot (GDN state after its last
    ///   token), in RAM pages of the arena or in an SSD file.
    ///
    /// Pages are shared by token identity: two prefixes with the same first
    /// `k * tokensPerPage` tokens share those `k` pages. That is exact under
    /// the rule the contract's exactness rests on (a cold run over the same
    /// tokens produces byte-identical KV).
    final class MemoryPrefixNode {
        let chunk: [Int32]
        let depth: Int
        weak var parent: MemoryPrefixNode?
        var children: [[Int32]: MemoryPrefixNode] = [:]
        /// Physical page of this node's tokens (one cache reference).
        var page: Int?
        var entries: [MemoryPrefixEntry] = []
        /// Entries in this subtree whose snapshot is in RAM (they need `page`).
        var ramEntriesBelow = 0
        /// Entries in this subtree, any tier (the node exists while > 0).
        var entriesBelow = 0

        init(chunk: [Int32], depth: Int, parent: MemoryPrefixNode?) {
            self.chunk = chunk
            self.depth = depth
            self.parent = parent
        }

        /// Nodes from depth 1 down to this one.
        var path: [MemoryPrefixNode] {
            var out: [MemoryPrefixNode] = []
            var n: MemoryPrefixNode? = self
            while let node = n, node.depth > 0 {
                out.append(node)
                n = node.parent
            }
            return out.reversed()
        }
    }

    final class MemoryPrefixEntry {
        enum Storage {
            /// Private tail page (if any) and recurrent snapshot pages, each
            /// with one cache reference.
            case ram(tailPage: Int?, snapshotPages: [Int])
            case ssd(url: URL, bytes: UInt64)
        }

        let handle: PrefixHandle
        let tokens: [Int32]
        unowned let node: MemoryPrefixNode
        var storage: Storage
        var lastUse: UInt64
        /// The command buffer that fills the entry's pages; until it has
        /// completed the entry is neither evicted nor spilled.
        var fill: MTLCommandBuffer?
        var pinned = false

        init(handle: PrefixHandle, tokens: [Int32], node: MemoryPrefixNode, storage: Storage, lastUse: UInt64,
             fill: MTLCommandBuffer?) {
            self.handle = handle
            self.tokens = tokens
            self.node = node
            self.storage = storage
            self.lastUse = lastUse
            self.fill = fill
        }

        var length: Int { tokens.count }
        /// Tokens after the last whole page (`node.depth` pages).
        func tail(tokensPerPage: Int) -> ArraySlice<Int32> { tokens[(node.depth * tokensPerPage)...] }

        var isRAM: Bool {
            if case .ram = storage { return true }
            return false
        }

        /// True once the fill has completed. A committed fill is waited on;
        /// a fill not yet committed (the caller's open command buffer) is not.
        /// A failed fill reports `failed`.
        func settle() -> (sealed: Bool, failed: Bool) {
            guard let cb = fill else { return (true, false) }
            switch cb.status {
            case .completed:
                fill = nil
                return (true, false)
            case .error:
                return (true, true)
            case .committed, .scheduled:
                cb.waitUntilCompleted()
                return settle()
            default:
                return (false, false)
            }
        }

        /// Fill status without waiting.
        var fillFailed: Bool { fill?.status == .error }
    }
}
