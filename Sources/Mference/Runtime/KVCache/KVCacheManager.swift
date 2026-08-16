import Foundation
import Darwin
import Metal

/// Which attention variant a layer runs. Gemma 4 interleaves 25 sliding-window
/// layers with 5 full-attention layers (the latter carry the K=V shared-tensor
/// quirk). Qwen 3.6 interleaves 30 gated-DeltaNet linear-attention layers with
/// 10 full-attention layers; linear layers keep a fixed-size recurrent state
/// (owned by `GDNStateManager`) instead of per-token K/V rows. Sourced from
/// `ArchConfig.fullAttentionLayerMask` (0 = swa, 1 = full, 2 = linear,
/// 3/4 = DeepSeek V4 CSA/HCA). DeepSeek-V4 architectures keep all attention
/// state — including the sliding-window ring — in `DSV4StateManager`, so
/// every layer of such a model is `.compressed` here with placeholder
/// storage; this manager only supplies the position cursor.
public enum LayerKind: Sendable { case swa, full, linear, compressed }

/// A read view the attention kernels bind. `offset` stays 0; ring-enabled SWA
/// layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Always 0 under linear storage.
    public let offset: Int
    /// Bytes per token (numKVHeads * headDim * sizeof(FP16)).
    public let stride: Int
    /// Number of valid positions written so far (== `position`). Attention reads
    /// `[0, validTokenCount]` (inclusive of the just-written token).
    public let validTokenCount: Int
    /// Ring start slot. 0 under linear storage; the hook for the wrap-aware path.
    public let startSlot: Int
}

/// Per-layer FP16 K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; FP16 ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear.
///
/// Gemma 4 full-attention layers carry the `attention_k_eq_v` quirk: K and V
/// share the `k_proj` weight, so a single 4-bit dequant + GEMV produces the
/// raw projection. But after that the K-slot runs `k_norm` (per-head, with
/// scale) + RoPE while the V-slot runs `v_norm` (per-head, no scale) and
/// skips RoPE — they diverge before entering attention. So the cache buffers
/// must be separate; aliasing them would smash the V values with K's normed,
/// rotated bytes (gemma4-block.md §2.2). We always allocate K and V slots,
/// independent of `attentionKEqV`.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool

    private let kBuffers: [MTLBuffer]
    private let vBuffers: [MTLBuffer]
    private let strides:  [Int]         // bytes per token, per layer
    private let kinds:    [LayerKind]
    private let capacityTokens: [Int]

    public private(set) var position: Int = 0

    private static let fp16Size = 2

    /// `pagedFullAttention` hands full-attention KV storage to a
    /// `KVPageStore`: those layers get the shared placeholder here (no linear
    /// allocation — 16 GiB at 262k context) while this manager keeps serving
    /// the position cursor and the linear-layer placeholders.
    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                fp16RingEnabled: Bool = false,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128,
                fp16RingCapacityOverride: Int? = nil,
                pagedFullAttention: Bool = false) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        self.config = config
        self.maxContext = maxContext
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, fp16RingCapacityOverride
                                  ?? ((slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens)))

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)

        // Linear-attention layers keep no per-token K/V rows; they share one
        // page-sized placeholder so the parallel arrays stay non-optional.
        // DeepSeek-V4 models own all attention state in `DSV4StateManager`,
        // so every layer (mask 0/3/4 alike) gets the placeholder there.
        var linearPlaceholder: MTLBuffer? = nil
        let allPlaceholder = config.hasCompressedAttentionLayers

        for layer in 0..<config.numLayers {
            let maskValue = config.fullAttentionLayerMask[layer]
            if maskValue == 2 || allPlaceholder || pagedFullAttention {
                let placeholder: MTLBuffer
                if let existing = linearPlaceholder {
                    placeholder = existing
                } else {
                    guard let made = device.makeBuffer(length: Int(getpagesize()),
                                                       options: .storageModeShared) else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    made.label = "kv.linear-placeholder"
                    linearPlaceholder = made
                    placeholder = made
                }
                ks.append(placeholder)
                vs.append(placeholder)
                st.append(0)
                if maskValue == 2 {
                    kd.append(.linear)
                } else if pagedFullAttention && maskValue == 1 {
                    // Storage lives in KVPageStore; kind stays .full so
                    // diagnostics report the layer's true attention variant.
                    kd.append(.full)
                } else {
                    kd.append(.compressed)
                }
                caps.append(0)
                continue
            }
            let isFull = maskValue != 0
            let stride = isFull ? fullStride : swaStride
            let capacity = ringEnabled && !isFull ? swaCapacity : maxContext
            let length = capacity * stride

            guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            kBuf.label = "kv.K.layer\(layer)"
            ks.append(kBuf)

            guard let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            vBuf.label = "kv.V.layer\(layer)"
            vs.append(vBuf)

            st.append(stride)
            kd.append(isFull ? .full : .swa)
            caps.append(capacity)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// Bytes per token for `layer` (K and V share the same stride).
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// Total bytes of the K buffer for `layer`.
    public func bufferLength(layer: Int) -> Int {
        return capacityTokens[layer] * strides[layer]
    }

    /// Write target for this layer's K projection at `position`.
    public func kSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear && kinds[layer] != .compressed,
                     "layer kind \(kinds[layer]) has no KV slots")
        validateRange(start: position, count: 1)
        return (kBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`. Always
    /// distinct from `kSlot` — full layers no longer alias K and V (Gemma 4
    /// applies different per-head norms + RoPE to K vs V; gemma4-block.md §2.2).
    public func vSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear && kinds[layer] != .compressed,
                     "layer kind \(kinds[layer]) has no KV slots")
        validateRange(start: position, count: 1)
        return (vBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    public func kRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (kBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (vBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, validTokenCount: position)
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: kBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, validTokenCount: position)
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: vBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
    }

    /// Rewind the position cursor to `position` after a partially accepted
    /// speculative-verify span. Storage is untouched — attention reads only
    /// `[0, validTokenCount)`, so rows past the cursor are dead until the
    /// next writer overwrites their slots.
    public func rewind(to position: Int) {
        precondition(position >= 0, "rewind position must be non-negative")
        precondition(position <= self.position,
                     "rewind \(position) is ahead of cursor \(self.position)")
        self.position = position
    }

    /// Drop all cached positions and return physical pages to the OS.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. `MADV_DONTNEED` on the page-aligned span
    /// releases resident memory between turns; pages fault back in on next write.
    public func reset() {
        position = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
            advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
        }
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        precondition(capacityTokens[layer] > 0, "layer has no KV storage")
        return position % capacityTokens[layer]
    }

    private func ringStartSlot(layer: Int, validTokenCount: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        let capacity = capacityTokens[layer]
        guard validTokenCount > capacity else { return 0 }
        return validTokenCount % capacity
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps FP16 KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
