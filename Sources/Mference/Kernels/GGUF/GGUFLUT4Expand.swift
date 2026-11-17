import Metal

/// GPU expansion of native GGUF routed-expert blocks into LUT4 slot sections
/// (WS4 U1 §3a), kernel `gguf_lut4_expand` in `Metal/GGUF/gguf_lut4.metal`.
/// Its CPU twin is `GGUFLUT4Reference.decompose`.
///
/// One `encode` handles any number of jobs of one source type; a job is a
/// run of consecutive source blocks of one projection. Offsets are absolute
/// byte offsets into the bound buffers, so one dispatch can cover every miss
/// of a layer (or a whole layer during prefill).
final class GGUFLUT4Expand {
    struct Job: Equatable, Sendable {
        public var srcOffset: UInt64
        public var dstIndexOffset: UInt64
        public var dstDOffset: UInt64
        public var dstMOffset: UInt64
        public var blocks: UInt32
        var pad: UInt32 = 0

        public init(srcOffset: UInt64, dstIndexOffset: UInt64, dstDOffset: UInt64,
                    dstMOffset: UInt64, blocks: UInt32) {
            self.srcOffset = srcOffset
            self.dstIndexOffset = dstIndexOffset
            self.dstDOffset = dstDOffset
            self.dstMOffset = dstMOffset
            self.blocks = blocks
        }
    }

    private let device: MTLDevice
    private var pipelines: [GGUFLUT4.SourceType: MTLComputePipelineState] = [:]
    private let iq3Grid: MTLBuffer
    private let iq2Grid: MTLBuffer
    private let signs: MTLBuffer

    init(context: MetalContext) throws {
        device = context.device
        let library = try MetalContext.moduleLibrary(device: context.device, module: "gguf_lut4")
        for type in GGUFLUT4.SourceType.allCases {
            let values = MTLFunctionConstantValues()
            var raw = type.rawValue
            values.setConstantValue(&raw, type: .uint, index: 0)
            let function = try library.makeFunction(name: "gguf_lut4_expand", constantValues: values)
            pipelines[type] = try context.device.makeComputePipelineState(function: function)
        }
        func upload<T>(_ array: [T]) throws -> MTLBuffer {
            guard let buffer = array.withUnsafeBytes({
                context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }) else { throw MetalError.commandBufferFailed("LUT4 table buffer allocation failed") }
            return buffer
        }
        iq3Grid = try upload(GGUFLUT4.iq3xxsIndexGrid)
        iq2Grid = try upload(GGUFLUT4.iq2sIndexGrid)
        signs = try upload(GGUFQuantTables.ksignsIQ2XS)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       type: GGUFLUT4.SourceType,
                       src: MTLBuffer,
                       dst: MTLBuffer,
                       jobs: [Job]) {
        guard !jobs.isEmpty, let pipeline = pipelines[type],
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        for job in jobs { precondition(job.dstIndexOffset % 16 == 0, "LUT4 index section must be 16-byte aligned") }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(src, offset: 0, index: 0)
        encoder.setBuffer(dst, offset: 0, index: 1)
        let jobBytes = jobs.count * MemoryLayout<Job>.stride
        if jobBytes <= 4096 {
            jobs.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: jobBytes, index: 2) }
        } else {
            // A retained-references command buffer keeps this alive until it completes.
            let jobBuffer = jobs.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: jobBytes, options: .storageModeShared)
            }
            encoder.setBuffer(jobBuffer, offset: 0, index: 2)
        }
        encoder.setBuffer(type == .iq2_s ? iq2Grid : iq3Grid, offset: 0, index: 3)
        encoder.setBuffer(signs, offset: 0, index: 4)
        let maxBlocks = Int(jobs.map(\.blocks).max()!)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 128)
        encoder.dispatchThreads(MTLSize(width: maxBlocks, height: jobs.count, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
