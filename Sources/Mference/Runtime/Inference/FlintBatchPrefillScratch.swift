import Metal

/// Reused layer-major scratch for Flint prompt prefill. The buffers are sized
/// once for the largest chunk; a later smaller chunk uses prefixes of them.
final class FlintBatchPrefillScratch {
    static let hiddenSize = 2_048
    static let qHeads = 32
    static let kvHeads = 4
    static let headDim = 128
    static let topK = 6
    static let moeIntermediate = 512
    static let denseIntermediate = 6_144

    let maxTokens: Int
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let q: MTLBuffer
    let attentionOut: MTLBuffer
    let attentionDelta: MTLBuffer
    let routedInput: MTLBuffer
    let sharedDelta: MTLBuffer
    let ffnDelta: MTLBuffer
    let routerIndices: MTLBuffer
    let routerWeights: MTLBuffer
    let routePartials: MTLBuffer
    let h2: MTLBuffer
    let sharedActs: MTLBuffer
    let gateUpAct: MTLBuffer
    let downScratch: MTLBuffer
    let router: FlintPrefillRouter
    let maxPairsPerTile: Int

    init(context: MetalContext, maxTokens: Int) throws {
        precondition(maxTokens > 0)
        let device = context.device
        self.maxTokens = maxTokens
        self.maxPairsPerTile = 16 * maxTokens
        let qStride = Self.qHeads * Self.headDim
        self.hidden = try Self.halfBuffer(device, maxTokens * Self.hiddenSize)
        self.normed = try Self.halfBuffer(device, maxTokens * Self.hiddenSize)
        self.q = try Self.halfBuffer(device, maxTokens * qStride)
        self.attentionOut = try Self.halfBuffer(device, maxTokens * qStride)
        self.attentionDelta = try Self.halfBuffer(device, maxTokens * Self.hiddenSize)
        self.routedInput = try Self.halfBuffer(device, maxTokens * Self.hiddenSize)
        self.sharedDelta = try Self.halfBuffer(device, maxTokens * Self.hiddenSize)
        self.ffnDelta = try Self.halfBuffer(device, maxTokens * Self.hiddenSize)
        self.routerIndices = try Self.buffer(
            device, maxTokens * Self.topK * MemoryLayout<UInt32>.stride)
        self.routerWeights = try Self.halfBuffer(device, maxTokens * Self.topK)
        self.routePartials = try Self.halfBuffer(
            device, maxTokens * Self.topK * Self.hiddenSize)
        self.h2 = try Self.halfBuffer(device, maxTokens * Self.hiddenSize)
        self.sharedActs = try Self.halfBuffer(
            device, maxTokens * Self.denseIntermediate)
        self.gateUpAct = try Self.halfBuffer(
            device, 3 * maxPairsPerTile * Self.moeIntermediate)
        self.downScratch = try Self.halfBuffer(
            device, maxPairsPerTile * Self.hiddenSize)
        self.router = try FlintPrefillRouter(context: context,
                                             maxTokens: maxTokens)
    }

    private static func halfBuffer(_ device: MTLDevice,
                                   _ elements: Int) throws -> MTLBuffer {
        try buffer(device, elements * MemoryLayout<Float16>.stride)
    }

    private static func buffer(_ device: MTLDevice,
                               _ bytes: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes,
                                             options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        return buffer
    }
}
