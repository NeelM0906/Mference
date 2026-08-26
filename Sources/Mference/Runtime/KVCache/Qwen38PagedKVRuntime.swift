import Foundation
import Metal

/// Shared paged long-context state for Qwen 3.8 (kvPagedPolicy == .on): the
/// page store owning full-attention KV, the selection policy and its pinned
/// working set, the per-token page tables and Quest score buffers, and the
/// metadata bookkeeping. Both the plain decode path (`Qwen38ForwardRunner`)
/// and the MTP speculative verify path (`Qwen38MTPSpeculator`) drive one
/// instance, so cursors, pins, and scores stay coherent across round and
/// plain tokens.
final class Qwen38PagedKVRuntime {
    let store: KVPageStore
    let kernels: KVPageKernels
    let selector: KVPageSelector
    /// [numFull][pagesPerLayer] float — Quest scores written per token,
    /// read back after the command buffer completes (lag-one selection).
    let scoresBuf: MTLBuffer
    /// [numFull][pagesPerLayer] uint32 — CPU-built page tables bound by the
    /// paged attention kernel.
    let tablesBuf: MTLBuffer
    var lastScores: [[Float]]
    var pendingMetadata: [Int] = []
    var selections: [KVPageSelector.Selection]
    /// Pages pinned for the in-flight token or verify round, per ordinal —
    /// the selection must survive its own fetches under a tight pool, where
    /// LRU alone could evict an earlier selection member to admit a later
    /// one.
    var pinnedSelections: [[Int]]
    let poolPagesPerLayer: Int

    private let numQHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let spillDir: URL

    init(context: MetalContext, config: ArchConfig, maxContext: Int,
         runtimeConfiguration: RuntimeConfiguration) throws {
        let device = context.device
        self.numQHeads = config.numHeads
        self.numKVHeads = config.numFullKVHeads
        self.headDim = config.fullHeadDim
        let pagesPerLayer = (maxContext + KVPageGeometry.tokensPerPage - 1)
            / KVPageGeometry.tokensPerPage
        let poolPages = min(
            runtimeConfiguration.kvPoolPagesPerLayer
                ?? RuntimeConfiguration.defaultKVPoolPagesPerLayer(config: config,
                                                                   maxContext: maxContext),
            pagesPerLayer)
        self.poolPagesPerLayer = poolPages
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mference-kvpages-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.spillDir = dir
        self.store = try KVPageStore(device: device,
                                     config: config,
                                     maxContext: maxContext,
                                     poolPagesPerLayer: poolPages,
                                     spillDirectory: dir)
        self.kernels = try KVPageKernels(context: context)
        self.selector = KVPageSelector(sinkPages: runtimeConfiguration.kvSinkPages,
                                       recentPages: runtimeConfiguration.kvRecentPages,
                                       topKPages: runtimeConfiguration.kvTopKPages)
        let numFull = store.geometry.fullLayerOrdinals.count
        let tableEntries = numFull * store.geometry.pagesPerLayer
        guard let scores = device.makeBuffer(length: tableEntries * 4,
                                             options: .storageModeShared),
              let tables = device.makeBuffer(length: tableEntries * 4,
                                             options: .storageModeShared) else {
            throw KVPageStoreError.allocationFailed("paged KV selection buffers")
        }
        scores.label = "kvpage.scores"
        tables.label = "kvpage.tables"
        self.scoresBuf = scores
        self.tablesBuf = tables
        self.lastScores = Array(repeating: [], count: numFull)
        self.selections = Array(repeating: .init(pages: [], selTokens: 0),
                                count: numFull)
        self.pinnedSelections = Array(repeating: [], count: numFull)
        // The pinned per-token selection (plus the unsealed tail and one
        // slot of eviction slack) must fit the pool.
        let worstSelection = runtimeConfiguration.kvSinkPages
            + runtimeConfiguration.kvRecentPages
            + runtimeConfiguration.kvTopKPages + 2
        guard poolPages >= min(pagesPerLayer, worstSelection) else {
            throw KVPageStoreError.allocationFailed(
                "kv pool (\(poolPages) pages/layer) smaller than the selection budget")
        }
    }

    deinit { try? FileManager.default.removeItem(at: spillDir) }

    func resetState() {
        store.reset()
        for i in 0..<lastScores.count { lastScores[i] = [] }
        pendingMetadata.removeAll()
        for i in 0..<selections.count { selections[i] = .init(pages: [], selTokens: 0) }
        for i in 0..<pinnedSelections.count { pinnedSelections[i] = [] }
    }

    /// Pick the token's page selection per layer (lag-one scores from the
    /// previous token), fetch any spilled members, pin them, and build the
    /// page tables — all before the command buffer encodes.
    func prepareSelections(position: Int) throws {
        let g = store.geometry
        let sealedPages = position / KVPageGeometry.tokensPerPage
        let tailValid = position % KVPageGeometry.tokensPerPage + 1
        let tables = tablesBuf.contents()
            .bindMemory(to: UInt32.self,
                        capacity: g.fullLayerOrdinals.count * g.pagesPerLayer)
        for (ordinal, layerIndex) in g.fullLayerOrdinals.enumerated() {
            // Touch the tail page so it has a slot before selection maps it.
            _ = try store.kSlot(layer: layerIndex, position: position)
            let selection = selector.select(scores: lastScores[ordinal],
                                            sealedPages: sealedPages,
                                            tailValidTokens: tailValid)
            // Swap pins to the new selection before fetching: members fetched
            // early must survive fetches of later members under LRU pressure.
            for page in pinnedSelections[ordinal] {
                store.unpin(layer: layerIndex, pageIndex: page)
            }
            var pinned: [Int] = []
            pinned.reserveCapacity(selection.pages.count)
            let base = ordinal * g.pagesPerLayer
            for (i, page) in selection.pages.enumerated() {
                let slot = try store.ensureResident(layer: layerIndex, pageIndex: page)
                store.pin(layer: layerIndex, pageIndex: page)
                pinned.append(page)
                tables[base + i] = UInt32(slot)
            }
            pinnedSelections[ordinal] = pinned
            selections[ordinal] = selection
        }
    }

    /// True when the selection at `position` provably includes every context
    /// page regardless of score staleness — the regime where speculative
    /// rounds are byte-identical to plain paged decode. `maxSpanTokens`
    /// bounds how many tokens can commit between Quest score refreshes (a
    /// verify span's accepted rows are emitted without their own score
    /// pass), which bounds how many trailing sealed pages may be unscored.
    func selectionIsExhaustive(at position: Int, maxSpanTokens: Int) -> Bool {
        let totalPages = position / KVPageGeometry.tokensPerPage + 1
        let lagPages = (maxSpanTokens + KVPageGeometry.tokensPerPage - 1)
            / KVPageGeometry.tokensPerPage
        return selector.coversEntireContext(totalPages: totalPages,
                                            maxUnscoredSealedPages: max(1, lagPages))
    }

    /// Extend each layer's table past the selection tail with the unsealed
    /// pages a speculative verify span [position, position + count) writes
    /// into, so per-position paged attention can address the whole span.
    /// Call after `prepareSelections(position:)`.
    func appendVerifySpan(position: Int, count: Int) throws {
        let g = store.geometry
        let tailPage = position / KVPageGeometry.tokensPerPage
        let lastPage = (position + count - 1) / KVPageGeometry.tokensPerPage
        guard lastPage > tailPage else { return }
        let tables = tablesBuf.contents()
            .bindMemory(to: UInt32.self,
                        capacity: g.fullLayerOrdinals.count * g.pagesPerLayer)
        for (ordinal, layerIndex) in g.fullLayerOrdinals.enumerated() {
            let base = ordinal * g.pagesPerLayer
            var entry = selections[ordinal].pages.count
            for page in (tailPage + 1)...lastPage {
                let slot = try store.kSlot(layer: layerIndex,
                                           position: page * KVPageGeometry.tokensPerPage)
                let slotIndex = slot.offset / g.kPageBytes
                store.pin(layer: layerIndex, pageIndex: page)
                pinnedSelections[ordinal].append(page)
                tables[base + entry] = UInt32(slotIndex)
                entry += 1
            }
        }
    }

    /// Selected logical tokens strictly before the tail page's first row for
    /// a verify round at `position` — verify position `i` attends
    /// `verifyBaseTokens + (position % 64) + i + 1` logical tokens.
    func verifyBaseTokens(ordinal: Int) -> Int {
        64 * max(0, selections[ordinal].pages.count - 1)
    }

    /// Quest min/max summaries for pages sealed by earlier tokens, encoded
    /// before this command buffer's score pass reads them.
    func encodePendingMetadata(commandBuffer cb: MTLCommandBuffer) throws {
        guard !pendingMetadata.isEmpty else { return }
        let g = store.geometry
        for pageIndex in pendingMetadata {
            for (ordinal, layerIndex) in g.fullLayerOrdinals.enumerated() {
                let slot = try store.ensureResident(layer: layerIndex,
                                                    pageIndex: pageIndex)
                kernels.encodePageMinMax(
                    commandBuffer: cb,
                    kPool: store.kPoolBuffer(layer: layerIndex),
                    slot: UInt32(slot),
                    validTokens: UInt32(KVPageGeometry.tokensPerPage),
                    metadata: store.metadataBuffer,
                    metadataOffset: g.metadataOffset(layerOrdinal: ordinal,
                                                     pageIndex: pageIndex),
                    numKVHeads: UInt32(numKVHeads),
                    headDim: UInt32(headDim))
            }
        }
        pendingMetadata.removeAll(keepingCapacity: true)
    }

    /// Quest criticality of every sealed page against `q` — the selection
    /// input for the next token.
    func encodeScores(commandBuffer cb: MTLCommandBuffer,
                      ordinal: Int, q: MTLBuffer, qOffset: Int,
                      sealedPages: Int) {
        guard sealedPages > 0 else { return }
        let g = store.geometry
        kernels.encodePageScores(
            commandBuffer: cb,
            q: q, qOffset: qOffset,
            metadata: store.metadataBuffer,
            metadataOffset: g.metadataOffset(layerOrdinal: ordinal, pageIndex: 0),
            scores: scoresBuf,
            scoresOffset: ordinal * g.pagesPerLayer * MemoryLayout<Float>.stride,
            numPages: UInt32(sealedPages),
            headDim: UInt32(headDim),
            numQHeads: UInt32(numQHeads),
            numKVHeads: UInt32(numKVHeads))
    }

    func readBackScores(sealedPages: Int) {
        guard sealedPages > 0 else { return }
        let g = store.geometry
        let numFull = g.fullLayerOrdinals.count
        let ptr = scoresBuf.contents()
            .bindMemory(to: Float.self, capacity: numFull * g.pagesPerLayer)
        for ordinal in 0..<numFull {
            lastScores[ordinal] = Array(UnsafeBufferPointer(
                start: ptr + ordinal * g.pagesPerLayer, count: sealedPages))
        }
    }

    /// Record pages fully sealed by a cursor move [from, to) for the next
    /// command buffer's metadata pass. Decode seals never carry their own
    /// min/max encode (the rows finalize in the same buffer that seals them
    /// only during prefill, which encodes metadata in-chunk instead).
    func noteAdvance(from oldPosition: Int, to newPosition: Int) {
        let before = oldPosition / KVPageGeometry.tokensPerPage
        let after = newPosition / KVPageGeometry.tokensPerPage
        if after > before {
            pendingMetadata.append(contentsOf: before..<after)
        }
    }
}
