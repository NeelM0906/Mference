import Foundation
import Metal

extension Native {
    /// The reference `MemoryPlanner`: plain arithmetic over the contract's
    /// geometry, refusing plans over the budget.
    public final class ReferenceMemoryPlanner: MemoryPlanner {
        public init() {}

        public static func geometry(config: Qwen35Config, precision: KVPrecision) -> KVPageGeometry {
            KVPageGeometry(attentionLayers: config.fullAttentionLayers.count, kvHeads: config.kvHeadCount,
                           headDim: config.keyLength, precision: precision)
        }

        public static func recurrentLayout(config: Qwen35Config) -> RecurrentLayout {
            RecurrentLayout(convKernel: config.ssmConvKernel, convChannels: config.ssmConvChannels,
                            valueHeads: config.ssmTimeStepRank, keyDim: config.ssmStateSize,
                            valueDim: config.ssmStateSize)
        }

        public static func pageCount(contextLength: Int, geometry: KVPageGeometry) -> Int {
            (contextLength + geometry.tokensPerPage - 1) / geometry.tokensPerPage
        }

        public func resolve(config: Qwen35Config, weightsMapped: UInt64, contextLength: Int, precision: KVPrecision,
                            maxVerifyRows: Int, maxPrefillChunk: Int, activationBytes: UInt64,
                            prefixCacheRAMBudget: UInt64, budget: UInt64) throws -> MemoryPlan {
            guard contextLength > 0, maxPrefillChunk > 0, maxVerifyRows >= 1 else {
                throw NativeError.invalidShape("plan limits must be positive")
            }
            let geometry = Self.geometry(config: config, precision: precision)
            let layout = Self.recurrentLayout(config: config)
            let pages = UInt64(Self.pageCount(contextLength: contextLength, geometry: geometry))
            let gdnLayers = UInt64(config.gatedDeltaNetLayers.count)
            let footprint = MemoryPlan.Footprint(
                weightsMapped: weightsMapped,
                kvArena: pages * UInt64(geometry.pageBytes),
                recurrentState: 2 * gdnLayers * UInt64(layout.layerBytes),
                verifyArena: gdnLayers * UInt64(maxVerifyRows) * UInt64(layout.layerBytes),
                activations: activationBytes,
                prefixCacheRAM: prefixCacheRAMBudget,
                other: 2 * pages * 4)
            guard footprint.total <= budget else {
                throw NativeError.memoryBudgetExceeded(required: footprint.total, budget: budget)
            }
            return MemoryPlan(contextLength: contextLength, precision: precision, maxVerifyRows: maxVerifyRows,
                              maxPrefillChunk: maxPrefillChunk, planned: footprint, budget: budget)
        }
    }

    /// The reference `KVStore`: one arena of physical pages handed out from a
    /// free list (in descending order, so page tables are never the
    /// identity), a two-slot page-table ring, double-banked recurrent state,
    /// a verify arena, and a RAM prefix cache that copies pages and the live
    /// recurrent bank into its own buffers (copy, not sharing: slow but
    /// obviously exact). Everything is shared storage so tests can inspect it.
    public final class ReferenceKVStore: KVStore {
        public let geometry: KVPageGeometry
        public let recurrentLayout: RecurrentLayout
        public let capacity: Int
        public private(set) var length = 0
        public let gdnLayerCount: Int
        public let verifySlots: Int

        let arena: MTLBuffer
        let pageTableRing: MTLBuffer
        let recurrent: MTLBuffer
        let verify: MTLBuffer
        let pagesTotal: Int
        let device: MTLDevice
        let prefixBudget: UInt64

        private var freePages: [Int]
        private var pages: [Int] = []
        /// Live recurrent bank L.
        private(set) var liveBank = 0
        private var stepCounter = 0
        private var ringSlot = 0
        private struct Step { var start: Int; var rows: Int; var input: Int; var output: Int }
        private var lastStep: Step?

        private struct PrefixEntry {
            var handle: PrefixHandle
            var tokens: [Int32]
            var kv: MTLBuffer?
            var recurrent: MTLBuffer
            var bytes: UInt64
        }
        private var prefixes: [PrefixEntry] = []
        private var nextPrefixID: UInt64 = 1

        public init(config: Qwen35Config, plan: MemoryPlan, device: MTLDevice) throws {
            geometry = ReferenceMemoryPlanner.geometry(config: config, precision: plan.precision)
            recurrentLayout = ReferenceMemoryPlanner.recurrentLayout(config: config)
            capacity = plan.contextLength
            gdnLayerCount = config.gatedDeltaNetLayers.count
            verifySlots = max(1, plan.maxVerifyRows)
            pagesTotal = ReferenceMemoryPlanner.pageCount(contextLength: capacity, geometry: geometry)
            prefixBudget = plan.planned.prefixCacheRAM
            self.device = device
            func make(_ bytes: Int, _ what: String) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                    throw NativeError.resource("cannot allocate \(what) (\(bytes) bytes)")
                }
                return b
            }
            arena = try make(pagesTotal * geometry.pageBytes, "KV arena")
            pageTableRing = try make(2 * pagesTotal * 4, "page table ring")
            recurrent = try make(2 * gdnLayerCount * recurrentLayout.layerBytes, "recurrent banks")
            verify = try make(gdnLayerCount * verifySlots * recurrentLayout.layerBytes, "verify arena")
            memset(recurrent.contents(), 0, recurrent.length)
            // Popped from the end: logical page 0 gets the highest physical page.
            freePages = Array(0..<pagesTotal)
        }

        // MARK: Steps

        public func beginStep(rows: Int) throws {
            guard rows >= 1 else { throw NativeError.invalidShape("beginStep rows \(rows)") }
            guard length + rows <= capacity else {
                throw NativeError.stateOutOfRange("step of \(rows) at \(length) exceeds capacity \(capacity)")
            }
            try ensurePages(for: length + rows)
            ringSlot = stepCounter % 2
            stepCounter += 1
            let table = pageTableRing.contents().advanced(by: ringSlot * pagesTotal * 4)
                .assumingMemoryBound(to: UInt32.self)
            for (i, p) in pages.enumerated() { table[i] = UInt32(p) }
            let input = liveBank, output = 1 - liveBank
            lastStep = Step(start: length, rows: rows, input: input, output: output)
            liveBank = output
            length += rows
        }

        private func ensurePages(for tokens: Int) throws {
            let needed = (tokens + geometry.tokensPerPage - 1) / geometry.tokensPerPage
            while pages.count < needed {
                guard let p = freePages.popLast() else { throw NativeError.stateOutOfRange("KV arena is full") }
                pages.append(p)
            }
        }

        private func releasePages(keeping tokens: Int) {
            let keep = (tokens + geometry.tokensPerPage - 1) / geometry.tokensPerPage
            while pages.count > keep { freePages.append(pages.removeLast()) }
        }

        public func pageTable() -> KVPageTable {
            KVPageTable(geometry: geometry, arena: arena, pageTable: pageTableRing,
                        pageTableOffset: ringSlot * pagesTotal * 4, pageCount: pages.count, tokenCount: length)
        }

        private func bankRef(_ bank: Int, gdnLayer: Int) -> RecurrentStateRef {
            RecurrentStateRef(layout: recurrentLayout, buffer: recurrent,
                              offset: (bank * gdnLayerCount + gdnLayer) * recurrentLayout.layerBytes)
        }

        public func recurrentState(gdnLayer: Int) -> (input: RecurrentStateRef, output: RecurrentStateRef) {
            if let s = lastStep {
                return (bankRef(s.input, gdnLayer: gdnLayer), bankRef(s.output, gdnLayer: gdnLayer))
            }
            return (bankRef(liveBank, gdnLayer: gdnLayer), bankRef(1 - liveBank, gdnLayer: gdnLayer))
        }

        public func verifyArena(gdnLayer: Int) -> RecurrentVerifyArena {
            RecurrentVerifyArena(layout: recurrentLayout, buffer: verify,
                                 offset: gdnLayer * verifySlots * recurrentLayout.layerBytes,
                                 slotStride: recurrentLayout.layerBytes, slots: verifySlots)
        }

        public func rollback(to position: Int, commandBuffer: MTLCommandBuffer) throws {
            guard let s = lastStep else { throw NativeError.stateOutOfRange("no step to roll back") }
            if position == s.start {
                liveBank = s.input
            } else if position > s.start && position <= s.start + s.rows {
                let r = position - s.start - 1
                if r < s.rows - 1 {
                    guard r < verifySlots else { throw NativeError.stateOutOfRange("verify slot \(r)") }
                    guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                        throw NativeError.resource("blit encoder")
                    }
                    for layer in 0..<gdnLayerCount {
                        blit.copy(from: verify,
                                  sourceOffset: (layer * verifySlots + r) * recurrentLayout.layerBytes,
                                  to: recurrent, destinationOffset: bankRef(s.output, gdnLayer: layer).offset,
                                  size: recurrentLayout.layerBytes)
                    }
                    blit.endEncoding()
                }
                liveBank = s.output
            } else {
                throw NativeError.stateOutOfRange("rollback to \(position): last step is \(s.start)..<\(s.start + s.rows)")
            }
            length = position
            releasePages(keeping: length)
            lastStep = nil
        }

        public func reset(commandBuffer: MTLCommandBuffer) throws {
            length = 0
            releasePages(keeping: 0)
            lastStep = nil
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw NativeError.resource("blit encoder") }
            blit.fill(buffer: recurrent,
                      range: liveBank * gdnLayerCount * recurrentLayout.layerBytes
                        ..< (liveBank + 1) * gdnLayerCount * recurrentLayout.layerBytes,
                      value: 0)
            blit.endEncoding()
        }

        // MARK: Prefix cache (RAM)

        public func lookupPrefix(_ tokens: [Int32]) -> PrefixMatch? {
            var best: PrefixEntry?
            for e in prefixes where e.tokens.count <= tokens.count && tokens.starts(with: e.tokens) {
                if best == nil || e.tokens.count > best!.tokens.count { best = e }
            }
            return best.map { PrefixMatch(handle: $0.handle, length: $0.tokens.count, tier: .ram) }
        }

        public func attachPrefix(_ match: PrefixMatch, commandBuffer: MTLCommandBuffer) throws {
            guard let e = prefixes.first(where: { $0.handle == match.handle }), e.tokens.count == match.length else {
                throw NativeError.stateOutOfRange("unknown prefix \(match.handle.id)")
            }
            releasePages(keeping: 0)
            try ensurePages(for: e.tokens.count)
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { throw NativeError.resource("blit encoder") }
            if let kv = e.kv {
                for (i, p) in pages.enumerated() {
                    blit.copy(from: kv, sourceOffset: i * geometry.pageBytes, to: arena,
                              destinationOffset: p * geometry.pageBytes, size: geometry.pageBytes)
                }
            }
            blit.copy(from: e.recurrent, sourceOffset: 0, to: recurrent,
                      destinationOffset: liveBank * gdnLayerCount * recurrentLayout.layerBytes,
                      size: gdnLayerCount * recurrentLayout.layerBytes)
            blit.endEncoding()
            length = e.tokens.count
            lastStep = nil
            // Most recently used last.
            prefixes.removeAll { $0.handle == e.handle }
            prefixes.append(e)
        }

        public func insertPrefix(_ tokens: [Int32], commandBuffer: MTLCommandBuffer) throws -> PrefixHandle {
            guard tokens.count == length else {
                throw NativeError.invalidShape("prefix of \(tokens.count) tokens, sequence holds \(length)")
            }
            let kvBytes = pages.count * geometry.pageBytes
            let recBytes = gdnLayerCount * recurrentLayout.layerBytes
            let bytes = UInt64(kvBytes + recBytes)
            guard bytes <= prefixBudget else {
                throw NativeError.memoryBudgetExceeded(required: bytes, budget: prefixBudget)
            }
            prefixes.removeAll { $0.tokens == tokens }
            while prefixRAM + bytes > prefixBudget, !prefixes.isEmpty { prefixes.removeFirst() }
            let kv = kvBytes > 0 ? device.makeBuffer(length: kvBytes, options: .storageModeShared) : nil
            guard kvBytes == 0 || kv != nil,
                  let rec = device.makeBuffer(length: recBytes, options: .storageModeShared),
                  let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw NativeError.resource("prefix snapshot buffers")
            }
            if let kv {
                for (i, p) in pages.enumerated() {
                    blit.copy(from: arena, sourceOffset: p * geometry.pageBytes, to: kv,
                              destinationOffset: i * geometry.pageBytes, size: geometry.pageBytes)
                }
            }
            blit.copy(from: recurrent, sourceOffset: liveBank * recBytes, to: rec, destinationOffset: 0,
                      size: recBytes)
            blit.endEncoding()
            let handle = PrefixHandle(id: nextPrefixID)
            nextPrefixID += 1
            prefixes.append(PrefixEntry(handle: handle, tokens: tokens, kv: kv, recurrent: rec, bytes: bytes))
            lastStep = nil
            return handle
        }

        var prefixRAM: UInt64 { prefixes.reduce(0) { $0 + $1.bytes } }

        public var footprint: MemoryPlan.Footprint {
            MemoryPlan.Footprint(kvArena: UInt64(arena.length), recurrentState: UInt64(recurrent.length),
                                 verifyArena: UInt64(verify.length), prefixCacheRAM: prefixRAM,
                                 other: UInt64(pageTableRing.length))
        }

        /// Physical page ids of the sequence, in logical order (tests).
        public var physicalPages: [Int] { pages }
    }
}
