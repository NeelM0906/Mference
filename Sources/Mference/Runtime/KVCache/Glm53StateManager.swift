import Foundation
import Metal

/// Per-layer state for GLM-5.3-Flash.
///
/// Kimi-Delta-Attention layers (mask 7) keep the depthwise conv's carried tail
/// (`convKernelSize - 1` rows of the `3 * heads * headDim` pre-activation
/// stream, FP16) and the recurrent state (`[heads, headDim, headDim]` FP32,
/// dk fastest — the reference's own layout). Sparse layers (mask 8) keep the
/// shared latent cache (`kvLoraRank` FP16 per token, K = V), the indexer's
/// per-token keys and pooling gates (`indexHeadDim` FP16 each), and the pooled
/// keys of every complete group (`indexHeadDim` FP16 per pool).
final class Glm53StateManager {
    let config: ArchConfig
    let maxContext: Int

    private(set) var convTail: [MTLBuffer?]
    private(set) var kdaState: [MTLBuffer?]
    private(set) var latents: [MTLBuffer?]
    private(set) var indexKeys: [MTLBuffer?]
    private(set) var indexGates: [MTLBuffer?]
    private(set) var pooledKeys: [MTLBuffer?]

    private static let fp16Size = MemoryLayout<Float16>.stride
    let convTailBytes: Int
    let kdaStateBytes: Int

    init(device: MTLDevice, config: ArchConfig, maxContext: Int) throws {
        precondition(config.hasGlm53Axes)
        self.config = config
        self.maxContext = maxContext
        let la = config.linearAttention
        let g = config.glm53
        let ca = config.compressedAttention
        let channels = 3 * la.numVHeads * la.valueHeadDim
        convTailBytes = (la.convKernelSize - 1) * channels * Self.fp16Size
        kdaStateBytes = la.numVHeads * la.valueHeadDim * la.keyHeadDim * MemoryLayout<Float>.stride
        let pools = maxContext / max(g.indexKPool, 1) + 1

        func makeBuffer(_ bytes: Int) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buf
        }
        var tails: [MTLBuffer?] = [], states: [MTLBuffer?] = []
        var lat: [MTLBuffer?] = [], keys: [MTLBuffer?] = [], gates: [MTLBuffer?] = [], pooled: [MTLBuffer?] = []
        for layer in 0..<config.numLayers {
            if config.layerIsKDA(layer) {
                tails.append(try makeBuffer(convTailBytes))
                states.append(try makeBuffer(kdaStateBytes))
                lat.append(nil); keys.append(nil); gates.append(nil); pooled.append(nil)
            } else {
                tails.append(nil); states.append(nil)
                lat.append(try makeBuffer(maxContext * g.kvLoraRank * Self.fp16Size))
                keys.append(try makeBuffer(maxContext * ca.indexHeadDim * Self.fp16Size))
                gates.append(try makeBuffer(maxContext * ca.indexHeadDim * Self.fp16Size))
                pooled.append(try makeBuffer(pools * ca.indexHeadDim * Self.fp16Size))
            }
        }
        convTail = tails; kdaState = states
        latents = lat; indexKeys = keys; indexGates = gates; pooledKeys = pooled
        reset()
    }

    /// Zero the recurrent state and conv tails; the caches are indexed by
    /// position and need no clearing.
    func reset() {
        for buf in convTail { if let buf { memset(buf.contents(), 0, buf.length) } }
        for buf in kdaState { if let buf { memset(buf.contents(), 0, buf.length) } }
    }
}
