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
    ///   `s = Float16(min(max|x| / 127, 65504))` (computed in F32, saturated
    ///   at the largest finite F16, then rounded to F16),
    ///   `q = clamp(rint(x / Float(s)), -127, 127)` with `rint` rounding half
    ///   to even, dequantized as `Float(s) * q`; whenever `s == 0` (an
    ///   all-zero row, or one so small that the scale underflows F16) every
    ///   `q` is 0
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
    ///   (the transpose of llama.cpp's `[channels][convKernel - 1]`; golden
    ///   state dumps are transposed before comparison)
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

    /// One GDN layer's recurrent state (one bank) in GPU memory.
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
    /// Positions: the sequence holds tokens `0..<length`. A step with `n`
    /// rows calls `beginStep(rows: n, kind:)`, which reserves positions
    /// `length ..< length + n`, grows `length` and makes this step's state
    /// views current (`pageTable()`, `recurrentState(gdnLayer:)`); then the
    /// step is encoded and committed.
    ///
    /// Recurrent state is double-banked so no step copies it. The store keeps
    /// a live bank index L. `beginStep` sets the step's `input = L` and
    /// `output = 1 - L`, then `L = output` (optimistic: the step is assumed
    /// kept). Only the most recent step can be undone: `rollback(to:)` with
    /// that step's start sets `L` back to its `input`; with `start + r + 1`
    /// after a verify step it copies verify-arena slot `r` into the step's
    /// `output` bank, which stays live. That covers verify rejection and the
    /// one pipelined step that runs past a stop. `insertPrefix` snapshots bank
    /// L, `attachPrefix` restores into bank L and `reset` zeroes bank L; all
    /// three require no step in flight, and after any of them `rollback`
    /// throws `stateOutOfRange` until the next `beginStep`.
    ///
    /// Pipelining: up to two steps may be in flight. Every CPU-written region a
    /// step reads (page table, step parameters) is ring-buffered, so
    /// `beginStep` for step t+1 never rewrites what step t reads.
    ///
    /// Prefix cache memory: cached pages live in the same arena. `kvArena` in
    /// the plan covers `contextLength` tokens; pages and recurrent snapshots
    /// held only by the cache are `prefixCacheRAM`, capped by the planner's
    /// `prefixCacheRAMBudget`.
    ///
    /// Methods that take a command buffer encode their own blit/compute work
    /// (no compute encoder may be open on it); the scheduler orders them with
    /// the step work.
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

        /// Begin a step of `rows` tokens of `kind` at position `length` (pages
        /// allocated, `length += rows`, state views switched to this step).
        /// The store records `kind`: `rollback(to: start + r + 1)` is legal
        /// only after a `.verify` step (whose verify arena holds the per-row
        /// states) and throws `stateOutOfRange` after `.prefill` or `.decode`.
        /// Throws `NativeError.stateOutOfRange` past `capacity`.
        func beginStep(rows: Int, kind: StepKind) throws
        /// The page table covering `0..<length` for the current step.
        func pageTable() -> KVPageTable
        /// Recurrent state banks of GDN layer `gdnLayer` (index among GDN
        /// layers) for the current step: read `input`, write `output`.
        func recurrentState(gdnLayer: Int) -> (input: RecurrentStateRef, output: RecurrentStateRef)
        /// Verify arena for GDN layer `gdnLayer`, `slots` >= the verify rows.
        func verifyArena(gdnLayer: Int) -> RecurrentVerifyArena
        /// Drop tokens `position..<length`. `position` must be the start of
        /// the most recent step, or, only when that step was `.verify`,
        /// `start + r + 1` for its rows starting at `start` (keeps rows `0...r`).
        /// `position == length` (keep everything) is accepted after any step
        /// as a no-op.
        func rollback(to position: Int, commandBuffer: MTLCommandBuffer) throws
        /// Clear the sequence (length 0, zero recurrent state); prefix cache kept.
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
        /// `activationBytes` is the runner's scratch at `maxPrefillChunk`
        /// rows; `prefixCacheRAMBudget` caps the RAM prefix tier.
        func resolve(config: Qwen35Config, weightsMapped: UInt64, contextLength: Int,
                     precision: KVPrecision, maxVerifyRows: Int, maxPrefillChunk: Int,
                     activationBytes: UInt64, prefixCacheRAMBudget: UInt64,
                     budget: UInt64) throws -> MemoryPlan
    }
}
