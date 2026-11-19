import Metal

extension Native {
    /// Storage precision of attention K/V (D4: FP16 default, INT8 opt-in).
    public enum KVPrecision: String, Codable, Sendable {
        case f16
        case int8
    }

    /// Normative paged-KV geometry shared by `KVStore` (W4, writes pages) and
    /// `AttentionOp` (W3, reads and appends).
    ///
    /// All full-attention layers share one page table: logical page `i` of a
    /// sequence (tokens `i*tokensPerPage ..< (i+1)*tokensPerPage`) is one
    /// physical page `p` holding that token range for every full-attention
    /// layer. Physical page `p` occupies `pageBytes` bytes at `p * pageBytes`
    /// in the arena buffer, laid out as:
    ///
    /// - values: `[attnLayer][kv: 0 = K, 1 = V][kvHead][token][dim]`, element
    ///   type F16 (`.f16`) or Int8 (`.int8`)
    /// - `.int8` only, after all values: scales
    ///   `[attnLayer][kv][kvHead][token]` F16, symmetric per (token, head):
    ///   `x ≈ scale * q`, `scale = max|x| / 127`, `q = round(x / scale)`
    ///   clamped to -127...127 (`scale = 0`, `q = 0` for an all-zero row)
    ///
    /// `attnLayer` is the index among full-attention layers (0..<8 for Quick),
    /// not the block index. K is stored after q/k norm and RoPE; V as projected.
    public struct KVPageGeometry: Codable, Hashable, Sendable {
        public var tokensPerPage: Int
        public var attentionLayers: Int
        public var kvHeads: Int
        public var headDim: Int
        public var precision: KVPrecision

        public init(tokensPerPage: Int = 64, attentionLayers: Int, kvHeads: Int,
                    headDim: Int, precision: KVPrecision = .f16) {
            self.tokensPerPage = tokensPerPage
            self.attentionLayers = attentionLayers
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.precision = precision
        }

        private var valueBytes: Int { precision == .f16 ? 2 : 1 }
        private var rowsPerPage: Int { attentionLayers * 2 * kvHeads * tokensPerPage }

        /// Bytes of one physical page.
        public var pageBytes: Int {
            rowsPerPage * headDim * valueBytes + (precision == .int8 ? rowsPerPage * 2 : 0)
        }

        /// Byte offset, inside a page, of element `dim` of (layer, kv, head, token).
        public func valueOffset(attnLayer: Int, kv: Int, kvHead: Int, token: Int, dim: Int) -> Int {
            let row = ((attnLayer * 2 + kv) * kvHeads + kvHead) * tokensPerPage + token
            return (row * headDim + dim) * valueBytes
        }

        /// Byte offset, inside a page, of the F16 scale of (layer, kv, head, token); `.int8` only.
        public func scaleOffset(attnLayer: Int, kv: Int, kvHead: Int, token: Int) -> Int {
            precondition(precision == .int8, "scales exist only in INT8 pages")
            let row = ((attnLayer * 2 + kv) * kvHeads + kvHead) * tokensPerPage + token
            return rowsPerPage * headDim + row * 2
        }
    }

    /// The GPU view of one sequence's KV that attention kernels read: the
    /// arena, the page table (`u32` physical page ids in logical order) and
    /// the number of valid tokens. Valid for the step it was obtained for.
    public struct KVPageTable {
        public var geometry: KVPageGeometry
        public var arena: MTLBuffer
        public var pageTable: MTLBuffer
        public var pageTableOffset: Int
        public var pageCount: Int
        /// Tokens stored, including the rows reserved for the current step.
        public var tokenCount: Int
        public init(geometry: KVPageGeometry, arena: MTLBuffer, pageTable: MTLBuffer,
                    pageTableOffset: Int, pageCount: Int, tokenCount: Int) {
            self.geometry = geometry
            self.arena = arena
            self.pageTable = pageTable
            self.pageTableOffset = pageTableOffset
            self.pageCount = pageCount
            self.tokenCount = tokenCount
        }
    }

    /// Normative recurrent-state layout of one Gated DeltaNet layer, shared by
    /// `KVStore` (owns the memory, snapshots) and `GDNOp` (reads/updates it).
    ///
    /// - conv state: F32 `[convKernel - 1][convChannels]`, row 0 oldest
    /// - delta-rule state S: F32 `[valueHeads][valueDim][keyDim]`, so that the
    ///   head output is `o[h][j] = sum_k S[h][j][k] * q[h][k]`
    ///
    /// `ssmOffset` is 256-byte aligned after the conv state.
    public struct RecurrentLayout: Codable, Hashable, Sendable {
        public var convKernel: Int
        public var convChannels: Int
        public var valueHeads: Int
        public var keyDim: Int
        public var valueDim: Int
        public init(convKernel: Int, convChannels: Int, valueHeads: Int, keyDim: Int, valueDim: Int) {
            self.convKernel = convKernel
            self.convChannels = convChannels
            self.valueHeads = valueHeads
            self.keyDim = keyDim
            self.valueDim = valueDim
        }
        public var convBytes: Int { (convKernel - 1) * convChannels * 4 }
        public var ssmOffset: Int { (convBytes + 255) / 256 * 256 }
        public var ssmBytes: Int { valueHeads * valueDim * keyDim * 4 }
        /// Bytes of one layer's state (conv + padding + S).
        public var layerBytes: Int { ssmOffset + ssmBytes }
    }

    /// One GDN layer's live recurrent state in GPU memory.
    public struct RecurrentStateRef {
        public var layout: RecurrentLayout
        public var buffer: MTLBuffer
        /// Start of this layer's state (conv state first).
        public var offset: Int
        public init(layout: RecurrentLayout, buffer: MTLBuffer, offset: Int) {
            self.layout = layout
            self.buffer = buffer
            self.offset = offset
        }
    }

    /// Per-row recurrent states written by a GDN verify pass: slot `r` holds
    /// the layer's state after consuming verify row `r` (same layout as the
    /// live state), so the scheduler can roll back to any accepted row.
    public struct RecurrentVerifyArena {
        public var layout: RecurrentLayout
        public var buffer: MTLBuffer
        public var offset: Int
        /// Bytes between consecutive row slots (>= `layout.layerBytes`).
        public var slotStride: Int
        public var slots: Int
        public init(layout: RecurrentLayout, buffer: MTLBuffer, offset: Int, slotStride: Int, slots: Int) {
            self.layout = layout
            self.buffer = buffer
            self.offset = offset
            self.slotStride = slotStride
            self.slots = slots
        }
    }

    /// Opaque handle to a cached prefix (radix-tree node) in a `KVStore`.
    public struct PrefixHandle: Hashable, Sendable {
        public let id: UInt64
        public init(id: UInt64) { self.id = id }
    }

    /// Result of a prefix lookup: the longest cached prefix of the query
    /// whose recurrent snapshot exists.
    public struct PrefixMatch: Sendable {
        public var handle: PrefixHandle
        /// Number of leading query tokens covered.
        public var length: Int
        /// Where the snapshot lives.
        public var tier: Tier
        public enum Tier: Sendable { case ram, ssd }
        public init(handle: PrefixHandle, length: Int, tier: Tier) {
            self.handle = handle
            self.length = length
            self.tier = tier
        }
    }

    /// KV cache and recurrent state of the one active sequence, plus the
    /// prefix cache. One model resident, one sequence at a time (multi-request
    /// batching is out of scope).
    ///
    /// Positions: the sequence holds tokens `0..<length`. A step at position
    /// `p` with `n` rows first calls `reserve(rows: n)` (which requires
    /// `p == length` and grows `length` by `n`), encodes, and on rejection of
    /// trailing rows calls `rollback(to:)`. Methods that take a command buffer
    /// encode their own blit/compute work (no compute encoder may be open on
    /// it) and are ordered with the step work by the scheduler.
    ///
    /// Exactness (D1, inside-Mference rule): restoring a prefix or rolling
    /// back and then continuing must produce logits byte-identical to a cold
    /// run over the same tokens.
    public protocol KVStore: AnyObject {
        var geometry: KVPageGeometry { get }
        var recurrentLayout: RecurrentLayout { get }
        /// Tokens currently stored.
        var length: Int { get }
        /// Maximum tokens (the resolved context of the `MemoryPlan`).
        var capacity: Int { get }

        /// Make room for `rows` new tokens at `length` and extend `length`.
        /// Throws `NativeError.stateOutOfRange` past `capacity`.
        func reserve(rows: Int) throws
        /// The page table covering `0..<length` for the current step.
        func pageTable() -> KVPageTable
        /// Live recurrent state of GDN layer `gdnLayer` (index among GDN layers).
        func recurrentState(gdnLayer: Int) -> RecurrentStateRef
        /// Verify arena for GDN layer `gdnLayer`, `slots` >= the verify rows.
        func verifyArena(gdnLayer: Int) -> RecurrentVerifyArena
        /// Drop tokens `position..<length`. For recurrent state, `position`
        /// must be the start of the last step (restores the pre-step snapshot)
        /// or `start + r + 1` after a verify step of rows starting at `start`
        /// (copies verify slot `r` into the live state).
        func rollback(to position: Int, commandBuffer: MTLCommandBuffer) throws
        /// Snapshot the live recurrent state as "before the next step" (the
        /// rollback point); called by the scheduler before each step.
        func markStepStart(commandBuffer: MTLCommandBuffer) throws
        /// Clear the sequence (length 0, zero recurrent state); cache kept.
        func reset(commandBuffer: MTLCommandBuffer) throws

        /// Longest cached prefix of `tokens` with a recurrent snapshot.
        func lookupPrefix(_ tokens: [Int32]) -> PrefixMatch?
        /// Make the sequence equal to the cached prefix: shares its KV pages
        /// (copy-on-write for a partial last page), restores its recurrent
        /// snapshot, sets `length = match.length`.
        func attachPrefix(_ match: PrefixMatch, commandBuffer: MTLCommandBuffer) throws
        /// Cache the current sequence state as the prefix `tokens` (whose count
        /// must equal `length`).
        func insertPrefix(_ tokens: [Int32], commandBuffer: MTLCommandBuffer) throws -> PrefixHandle

        /// Live footprint accounting.
        var footprint: MemoryPlan.Footprint { get }
    }

    /// Footprint plan for one resident model at a resolved context.
    public struct MemoryPlan: Sendable {
        public struct Footprint: Sendable, Equatable {
            public var weightsMapped: UInt64
            public var kvArena: UInt64
            public var recurrentState: UInt64
            public var verifyArena: UInt64
            public var activations: UInt64
            public var prefixCacheRAM: UInt64
            public var other: UInt64
            public init(weightsMapped: UInt64 = 0, kvArena: UInt64 = 0, recurrentState: UInt64 = 0,
                        verifyArena: UInt64 = 0, activations: UInt64 = 0, prefixCacheRAM: UInt64 = 0,
                        other: UInt64 = 0) {
                self.weightsMapped = weightsMapped
                self.kvArena = kvArena
                self.recurrentState = recurrentState
                self.verifyArena = verifyArena
                self.activations = activations
                self.prefixCacheRAM = prefixCacheRAM
                self.other = other
            }
            public var total: UInt64 {
                weightsMapped + kvArena + recurrentState + verifyArena + activations + prefixCacheRAM + other
            }
        }

        /// Context the plan was resolved for (tokens).
        public var contextLength: Int
        public var precision: KVPrecision
        /// Maximum verify rows the arenas are sized for (1 = no speculation).
        public var maxVerifyRows: Int
        /// Largest prefill chunk (rows) activations are sized for.
        public var maxPrefillChunk: Int
        public var planned: Footprint
        /// Budget the plan was checked against.
        public var budget: UInt64

        public init(contextLength: Int, precision: KVPrecision, maxVerifyRows: Int,
                    maxPrefillChunk: Int, planned: Footprint, budget: UInt64) {
            self.contextLength = contextLength
            self.precision = precision
            self.maxVerifyRows = maxVerifyRows
            self.maxPrefillChunk = maxPrefillChunk
            self.planned = planned
            self.budget = budget
        }
    }

    /// Resolves a `MemoryPlan` before any allocation.
    public protocol MemoryPlanner: AnyObject {
        /// Throws `NativeError.memoryBudgetExceeded` when it does not fit.
        func resolve(config: Qwen35Config, weightsMapped: UInt64, contextLength: Int,
                     precision: KVPrecision, maxVerifyRows: Int, maxPrefillChunk: Int,
                     budget: UInt64) throws -> MemoryPlan
    }
}
