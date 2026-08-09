import Metal

/// Maple's fixed-shape BF16 Q/K norm and optional sliding-layer RoPE.
final class MapleQKNormRoPE {
    static let numQHeads = 16
    static let numKVHeads = 4
    static let headDim = 128
    static let epsilon: Float = 1e-6

    private let noRoPE: MTLComputePipelineState
    private let partialRoPE: MTLComputePipelineState
    private let invFreq: MTLBuffer

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(
            device: context.device, module: "maple_attention", safeMath: true)
        self.noRoPE = try Self.pipeline(
            library: library, device: context.device, ropeDim: 0)
        self.partialRoPE = try Self.pipeline(
            library: library, device: context.device, ropeDim: 64)
        let values = Self.invFreqBits.map(Float.init(bitPattern:))
        guard let invFreq = context.device.makeBuffer(
            bytes: values,
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.invFreq = invFreq
    }

    /// Normalizes Q=[16,128] and K=[4,128] in place. Sliding layers rotate
    /// only dimensions 0..<64; full layers remain NoPE.
    func encode(commandBuffer: MTLCommandBuffer,
                q: MTLBuffer, qOffset: Int = 0,
                k: MTLBuffer, kOffset: Int = 0,
                qWeight: MTLBuffer, qWeightOffset: Int = 0,
                kWeight: MTLBuffer, kWeightOffset: Int = 0,
                position: UInt32,
                sliding: Bool) {
        precondition(position < MapleDecodeAttention.maximumSequenceLength,
                     "Maple position exceeds the supported 128K context")
        let qBytes = Self.numQHeads * Self.headDim * MemoryLayout<UInt16>.stride
        let kBytes = Self.numKVHeads * Self.headDim * MemoryLayout<UInt16>.stride
        let weightBytes = Self.headDim * MemoryLayout<UInt16>.stride
        Self.requireRange(q, offset: qOffset, bytes: qBytes, named: "Q")
        Self.requireRange(k, offset: kOffset, bytes: kBytes, named: "K")
        Self.requireRange(qWeight, offset: qWeightOffset, bytes: weightBytes,
                          named: "Q norm weight")
        Self.requireRange(kWeight, offset: kWeightOffset, bytes: weightBytes,
                          named: "K norm weight")
        Self.requireDisjoint(q, qOffset, qBytes, k, kOffset, kBytes,
                             left: "Q", right: "K")
        Self.requireDisjoint(q, qOffset, qBytes, qWeight, qWeightOffset, weightBytes,
                             left: "Q", right: "Q norm weight")
        Self.requireDisjoint(q, qOffset, qBytes, kWeight, kWeightOffset, weightBytes,
                             left: "Q", right: "K norm weight")
        Self.requireDisjoint(k, kOffset, kBytes, qWeight, qWeightOffset, weightBytes,
                             left: "K", right: "Q norm weight")
        Self.requireDisjoint(k, kOffset, kBytes, kWeight, kWeightOffset, weightBytes,
                             left: "K", right: "K norm weight")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(sliding ? partialRoPE : noRoPE)
        encoder.setBuffer(q, offset: qOffset, index: 0)
        encoder.setBuffer(k, offset: kOffset, index: 1)
        encoder.setBuffer(qWeight, offset: qWeightOffset, index: 2)
        encoder.setBuffer(kWeight, offset: kWeightOffset, index: 3)
        encoder.setBuffer(invFreq, offset: 0, index: 4)
        var position = position
        var epsilon = Self.epsilon
        encoder.setBytes(&position, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: Self.numQHeads + Self.numKVHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func pipeline(library: MTLLibrary,
                                 device: MTLDevice,
                                 ropeDim: UInt32) throws -> MTLComputePipelineState {
        let constants = MTLFunctionConstantValues()
        var ropeDim = ropeDim
        constants.setConstantValue(&ropeDim, type: .uint, index: 0)
        let function = try library.makeFunction(
            name: "maple_qk_norm_rope_decode", constantValues: constants)
        return try device.makeComputePipelineState(function: function)
    }

    private static func requireRange(_ buffer: MTLBuffer,
                                     offset: Int,
                                     bytes: Int,
                                     named: String) {
        precondition(offset >= 0 && offset <= buffer.length &&
                     bytes <= buffer.length - offset,
                     "Maple \(named) buffer is too small")
        precondition(offset.isMultiple(of: MemoryLayout<UInt16>.stride),
                     "Maple \(named) offset must be BF16-aligned")
    }

    private static func requireDisjoint(_ leftBuffer: MTLBuffer,
                                        _ leftOffset: Int,
                                        _ leftBytes: Int,
                                        _ rightBuffer: MTLBuffer,
                                        _ rightOffset: Int,
                                        _ rightBytes: Int,
                                        left: String,
                                        right: String) {
        guard leftBuffer === rightBuffer else { return }
        precondition(leftOffset + leftBytes <= rightOffset ||
                     rightOffset + rightBytes <= leftOffset,
                     "Maple \(left) and \(right) ranges must not overlap")
    }

    private static let invFreqBits: [UInt32] = [
        0x3f80_0000, 0x3f3f_f912, 0x3f0f_f59a, 0x3ed7_e89b,
        0x3ea1_e89c, 0x3e72_d424, 0x3e36_1888, 0x3e08_8d78,
        0x3dcc_cccd, 0x3d99_940e, 0x3d66_55c3, 0x3d2c_ba16,
        0x3d01_86e2, 0x3cc2_4350, 0x3c91_ad39, 0x3c5a_7bf2,
        0x3c23_d70b, 0x3bf5_b9b0, 0x3bb8_449d, 0x3b8a_2e77,
        0x3b4f_3e37, 0x3b1b_690c, 0x3ae9_1528, 0x3aae_c98e,
        0x3a83_126f, 0x3a44_948d, 0x3a13_6a17, 0x39dd_1726,
        0x39a5_cb60, 0x3978_a815, 0x393a_7754, 0x390b_d472,
    ]
}
