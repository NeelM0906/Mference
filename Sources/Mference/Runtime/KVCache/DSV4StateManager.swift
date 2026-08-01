import Foundation
import Metal

/// Per-layer attention state for DeepSeek-V4 layers.
///
/// Every layer keeps a sliding-window K=V ring (`slidingWindow` rows of
/// `headDim` FP16; K and V are the same storage by construction). CSA/HCA
/// layers additionally keep:
///   - the compressed-entry cache (`maxContext / rate` rows of `headDim`),
///   - pending source rows awaiting a full window (`rate` rows of the
///     projection width: 2*headDim for CSA's dual series, headDim for HCA),
///   - CSA only: the prior window's Ca slices (`rate` rows of `headDim`,
///     gate values carry the position bias, matching the reference), and the
///     indexer's parallel set at `indexHeadDim`.
///
/// All buffers are allocated once at init; `reset()` zeroes the counters and
/// returns pages with `POSIX_MADV_DONTNEED` like `KVCacheManager.reset`.
final class DSV4StateManager {
    struct LayerCounters {
        /// Source tokens seen (absolute position + 1 after produce).
        var tokens: Int = 0
        /// Compressed entries emitted (compressor).
        var compressedEntries: Int = 0
        /// Rows currently pending in the compressor buffer.
        var pendingRows: Int = 0
        /// Indexer entries emitted / rows pending (CSA only).
        var indexerEntries: Int = 0
        var indexerPendingRows: Int = 0
        /// Whether a prior-window Ca slice is available (CSA only).
        var hasPrior: Bool = false
        var indexerHasPrior: Bool = false
    }

    let config: ArchConfig
    let maxContext: Int

    /// Window K=V ring per layer: [ringCapacity, headDim] FP16.
    private(set) var windowKV: [MTLBuffer]
    let ringCapacity: Int

    /// Compressed caches, non-nil at CSA/HCA indices.
    private(set) var compressedKV: [MTLBuffer?]
    private(set) var pendingKV: [MTLBuffer?]
    private(set) var pendingGate: [MTLBuffer?]
    private(set) var priorCaKV: [MTLBuffer?]
    private(set) var priorCaGate: [MTLBuffer?]

    /// Indexer caches, non-nil at CSA indices.
    private(set) var indexerKeys: [MTLBuffer?]
    private(set) var indexerPendingKV: [MTLBuffer?]
    private(set) var indexerPendingGate: [MTLBuffer?]
    private(set) var indexerPriorCaKV: [MTLBuffer?]
    private(set) var indexerPriorCaGate: [MTLBuffer?]

    /// Written by the decode loop as each token's compressor bookkeeping
    /// commits.
    var counters: [LayerCounters]

    private static let fp16Size = MemoryLayout<Float16>.stride

    init(device: MTLDevice, config: ArchConfig, maxContext: Int) throws {
        precondition(config.hasCompressedAttentionLayers)
        self.config = config
        self.maxContext = maxContext
        let ca = config.compressedAttention
        let headDim = config.fullHeadDim
        // The ring holds the window plus the in-flight row (the reference
        // keeps `window - 1` past rows and appends the current token).
        self.ringCapacity = config.slidingWindow

        var window: [MTLBuffer] = []
        var compressed: [MTLBuffer?] = []
        var pKV: [MTLBuffer?] = []
        var pGate: [MTLBuffer?] = []
        var caKV: [MTLBuffer?] = []
        var caGate: [MTLBuffer?] = []
        var idxKeys: [MTLBuffer?] = []
        var idxPKV: [MTLBuffer?] = []
        var idxPGate: [MTLBuffer?] = []
        var idxCaKV: [MTLBuffer?] = []
        var idxCaGate: [MTLBuffer?] = []

        func makeBuffer(_ bytes: Int) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: max(bytes, 16),
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buf
        }

        for layer in 0..<config.numLayers {
            window.append(try makeBuffer(ringCapacity * headDim * Self.fp16Size))
            let isCSA = config.layerIsCSA(layer)
            let isHCA = config.layerIsHCA(layer)
            if isCSA || isHCA {
                let rate = isCSA ? ca.csaCompressRate : ca.hcaCompressRate
                let rowWidth = isCSA ? 2 * headDim : headDim
                let maxEntries = (maxContext + rate - 1) / rate
                compressed.append(try makeBuffer(maxEntries * headDim * Self.fp16Size))
                pKV.append(try makeBuffer(rate * rowWidth * Self.fp16Size))
                pGate.append(try makeBuffer(rate * rowWidth * Self.fp16Size))
                if isCSA {
                    caKV.append(try makeBuffer(rate * headDim * Self.fp16Size))
                    caGate.append(try makeBuffer(rate * headDim * Self.fp16Size))
                    let idxRate = ca.csaCompressRate
                    let idxEntries = (maxContext + idxRate - 1) / idxRate
                    idxKeys.append(try makeBuffer(
                        idxEntries * ca.indexHeadDim * Self.fp16Size))
                    idxPKV.append(try makeBuffer(
                        idxRate * 2 * ca.indexHeadDim * Self.fp16Size))
                    idxPGate.append(try makeBuffer(
                        idxRate * 2 * ca.indexHeadDim * Self.fp16Size))
                    idxCaKV.append(try makeBuffer(
                        idxRate * ca.indexHeadDim * Self.fp16Size))
                    idxCaGate.append(try makeBuffer(
                        idxRate * ca.indexHeadDim * Self.fp16Size))
                } else {
                    caKV.append(nil); caGate.append(nil)
                    idxKeys.append(nil); idxPKV.append(nil); idxPGate.append(nil)
                    idxCaKV.append(nil); idxCaGate.append(nil)
                }
            } else {
                compressed.append(nil); pKV.append(nil); pGate.append(nil)
                caKV.append(nil); caGate.append(nil)
                idxKeys.append(nil); idxPKV.append(nil); idxPGate.append(nil)
                idxCaKV.append(nil); idxCaGate.append(nil)
            }
        }

        self.windowKV = window
        self.compressedKV = compressed
        self.pendingKV = pKV
        self.pendingGate = pGate
        self.priorCaKV = caKV
        self.priorCaGate = caGate
        self.indexerKeys = idxKeys
        self.indexerPendingKV = idxPKV
        self.indexerPendingGate = idxPGate
        self.indexerPriorCaKV = idxCaKV
        self.indexerPriorCaGate = idxCaGate
        self.counters = Array(repeating: LayerCounters(), count: config.numLayers)
    }

    /// Ring slot receiving the row for absolute `position`.
    func windowSlot(position: Int) -> Int { position % ringCapacity }

    /// Rows currently valid in the window ring after `position` was written.
    func windowCount(position: Int) -> Int { min(position + 1, ringCapacity) }

    /// Absolute position of the oldest valid window row.
    func windowStartPosition(position: Int) -> Int {
        max(0, position + 1 - ringCapacity)
    }

    func reset() {
        counters = Array(repeating: LayerCounters(), count: config.numLayers)
        for buf in windowKV { discard(buf) }
        for buf in compressedKV { if let buf { discard(buf) } }
        for buf in indexerKeys { if let buf { discard(buf) } }
        // Pending / overlap buffers are small; zero them so a fresh sequence
        // never reads a stale row.
        for buf in pendingKV { if let buf { zero(buf) } }
        for buf in pendingGate { if let buf { zero(buf) } }
        for buf in priorCaKV { if let buf { zero(buf) } }
        for buf in priorCaGate { if let buf { zero(buf) } }
        for buf in indexerPendingKV { if let buf { zero(buf) } }
        for buf in indexerPendingGate { if let buf { zero(buf) } }
        for buf in indexerPriorCaKV { if let buf { zero(buf) } }
        for buf in indexerPriorCaGate { if let buf { zero(buf) } }
    }

    private func discard(_ buffer: MTLBuffer) {
        let pageSize = Int(getpagesize())
        let length = buffer.length & ~(pageSize - 1)
        if length > 0 {
            posix_madvise(buffer.contents(), length, POSIX_MADV_DONTNEED)
        }
    }

    private func zero(_ buffer: MTLBuffer) {
        memset(buffer.contents(), 0, buffer.length)
    }
}
