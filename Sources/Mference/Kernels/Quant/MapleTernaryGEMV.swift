import Metal

/// Maple's exact BF16 resident projections and full INT4 vocabulary head.
final class MapleTernaryGEMV {
    static let hiddenSize = 2048
    static let vocabularySize = 151_936
    private static let groupSize = Quantization.groupSize

    private let ternaryPipeline: MTLComputePipelineState
    private let int4Pipeline: MTLComputePipelineState
    private let embeddingPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "maple_ternary",
                                                     safeMath: true)
        self.ternaryPipeline = try Self.makePipeline("maple_ternary_qmv_d2048",
                                                      library: library,
                                                      device: context.device)
        self.int4Pipeline = try Self.makePipeline("maple_int4_qmv_d2048",
                                                   library: library,
                                                   device: context.device)
        self.embeddingPipeline = try Self.makePipeline("maple_embed_lookup_int4_bf16",
                                                        library: library,
                                                        device: context.device)
    }

    /// Projects native-BF16 activations with widened INT4 ternary weights.
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                rows: UInt32) {
        encodeQMV(commandBuffer: commandBuffer,
                  pipeline: ternaryPipeline,
                  outputBytesPerRow: MemoryLayout<UInt16>.stride,
                  maximumRows: Self.hiddenSize,
                  weightAlignment: 1,
                  weights: weights, weightsOffset: weightsOffset,
                  scales: scales, scalesOffset: scalesOffset,
                  biases: biases, biasesOffset: biasesOffset,
                  x: x, xOffset: xOffset, y: y, yOffset: yOffset, rows: rows)
    }

    /// Projects native-BF16 activations with Maple's exact INT4/group-64 head.
    /// The kernel rounds logits to BF16 before the existing FP16 export; the
    /// later full-model parity gate verifies that retained logits stay within
    /// FP16's exactly representable range.
    func encodeInt4(commandBuffer: MTLCommandBuffer,
                    weights: MTLBuffer, weightsOffset: Int = 0,
                    scales: MTLBuffer, scalesOffset: Int = 0,
                    biases: MTLBuffer, biasesOffset: Int = 0,
                    x: MTLBuffer, xOffset: Int = 0,
                    y: MTLBuffer, yOffset: Int = 0,
                    rows: UInt32) {
        encodeQMV(commandBuffer: commandBuffer,
                  pipeline: int4Pipeline,
                  outputBytesPerRow: MemoryLayout<Float16>.stride,
                  maximumRows: Self.vocabularySize,
                  weightAlignment: MemoryLayout<UInt16>.alignment,
                  weights: weights, weightsOffset: weightsOffset,
                  scales: scales, scalesOffset: scalesOffset,
                  biases: biases, biasesOffset: biasesOffset,
                  x: x, xOffset: xOffset, y: y, yOffset: yOffset, rows: rows)
    }

    /// Dequantizes one INT4 embedding row directly into native-BF16 storage.
    func encodeEmbedding(commandBuffer: MTLCommandBuffer,
                         table: MTLBuffer, tableOffset: Int = 0,
                         scales: MTLBuffer, scalesOffset: Int = 0,
                         biases: MTLBuffer, biasesOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0,
                         tokenID: UInt32) {
        let rowBytes = Self.hiddenSize / 2
        let parameterBytes = Self.hiddenSize / Self.groupSize * MemoryLayout<UInt16>.stride
        let token = Int(tokenID)
        precondition(tableOffset >= 0 && scalesOffset >= 0 && biasesOffset >= 0 &&
                     outOffset >= 0,
                     "Maple embedding offsets must be non-negative")
        precondition(token < Self.vocabularySize,
                     "Maple embedding token ID is outside the vocabulary")
        precondition(token <= (Int.max - tableOffset) / rowBytes,
                     "Maple embedding token ID overflows its table offset")
        precondition(token <= (Int.max - scalesOffset) / parameterBytes &&
                     token <= (Int.max - biasesOffset) / parameterBytes,
                     "Maple embedding token ID overflows its parameter offset")
        Self.requireRange(table, offset: tableOffset + token * rowBytes,
                          bytes: rowBytes, named: "embedding table", alignment: 1)
        Self.requireRange(scales, offset: scalesOffset + token * parameterBytes,
                          bytes: parameterBytes, named: "embedding scales")
        Self.requireRange(biases, offset: biasesOffset + token * parameterBytes,
                          bytes: parameterBytes, named: "embedding biases")
        Self.requireRange(out, offset: outOffset,
                          bytes: Self.hiddenSize * MemoryLayout<UInt16>.stride,
                          named: "embedding output")
        let outputBytes = Self.hiddenSize * MemoryLayout<UInt16>.stride
        Self.requireDisjoint(out, offset: outOffset, bytes: outputBytes,
                             table, offset: tableOffset + token * rowBytes,
                             bytes: rowBytes, named: "embedding output and table")
        Self.requireDisjoint(out, offset: outOffset, bytes: outputBytes,
                             scales, offset: scalesOffset + token * parameterBytes,
                             bytes: parameterBytes, named: "embedding output and scales")
        Self.requireDisjoint(out, offset: outOffset, bytes: outputBytes,
                             biases, offset: biasesOffset + token * parameterBytes,
                             bytes: parameterBytes, named: "embedding output and biases")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(embeddingPipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var tokenValue = tokenID
        encoder.setBytes(&tokenValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreads(MTLSize(width: Self.hiddenSize, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func makePipeline(_ name: String,
                                     library: MTLLibrary,
                                     device: MTLDevice) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        return try device.makeComputePipelineState(function: function)
    }

    private func encodeQMV(commandBuffer: MTLCommandBuffer,
                           pipeline: MTLComputePipelineState,
                           outputBytesPerRow: Int,
                           maximumRows: Int,
                           weightAlignment: Int,
                           weights: MTLBuffer, weightsOffset: Int,
                           scales: MTLBuffer, scalesOffset: Int,
                           biases: MTLBuffer, biasesOffset: Int,
                           x: MTLBuffer, xOffset: Int,
                           y: MTLBuffer, yOffset: Int,
                           rows: UInt32) {
        precondition(rows > 0, "Maple quantized projection needs output rows")
        let rowCount = Int(rows)
        precondition(rowCount <= maximumRows,
                     "Maple quantized projection exceeds its supported row count")
        let weightBytes = Self.hiddenSize / 2
        let parameterBytes = Self.hiddenSize / Self.groupSize * MemoryLayout<UInt16>.stride
        precondition(rowCount <= Int.max / weightBytes && rowCount <= Int.max / parameterBytes &&
                     rowCount <= Int.max / outputBytesPerRow,
                     "Maple projection row count overflows its byte range")
        let weightsBytes = rowCount * weightBytes
        let parametersBytes = rowCount * parameterBytes
        let outputBytes = rowCount * outputBytesPerRow
        let inputBytes = Self.hiddenSize * MemoryLayout<UInt16>.stride
        Self.requireRange(weights, offset: weightsOffset, bytes: weightsBytes,
                          named: "weights", alignment: weightAlignment)
        Self.requireRange(scales, offset: scalesOffset, bytes: parametersBytes,
                          named: "scales")
        Self.requireRange(biases, offset: biasesOffset, bytes: parametersBytes,
                          named: "biases")
        Self.requireRange(x, offset: xOffset,
                          bytes: inputBytes,
                          named: "input")
        Self.requireRange(y, offset: yOffset, bytes: outputBytes,
                          named: "output", alignment: outputBytesPerRow)
        Self.requireDisjoint(y, offset: yOffset, bytes: outputBytes,
                             weights, offset: weightsOffset, bytes: weightsBytes,
                             named: "output and weights")
        Self.requireDisjoint(y, offset: yOffset, bytes: outputBytes,
                             scales, offset: scalesOffset, bytes: parametersBytes,
                             named: "output and scales")
        Self.requireDisjoint(y, offset: yOffset, bytes: outputBytes,
                             biases, offset: biasesOffset, bytes: parametersBytes,
                             named: "output and biases")
        Self.requireDisjoint(y, offset: yOffset, bytes: outputBytes,
                             x, offset: xOffset, bytes: inputBytes,
                             named: "output and input")

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var rowCountValue = rows
        encoder.setBytes(&rowCountValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: (rowCount + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func requireRange(_ buffer: MTLBuffer,
                                     offset: Int,
                                     bytes: Int,
                                     named: String,
                                     alignment: Int = MemoryLayout<UInt16>.stride) {
        precondition(offset >= 0 && bytes >= 0 && offset <= buffer.length &&
                     bytes <= buffer.length - offset,
                     "Maple \(named) buffer is too small")
        precondition(offset.isMultiple(of: alignment),
                     "Maple \(named) offset is misaligned")
    }

    private static func requireDisjoint(_ first: MTLBuffer,
                                        offset firstOffset: Int,
                                        bytes firstBytes: Int,
                                        _ second: MTLBuffer,
                                        offset secondOffset: Int,
                                        bytes secondBytes: Int,
                                        named: String) {
        guard first === second else { return }
        let firstEnd = firstOffset + firstBytes
        let secondEnd = secondOffset + secondBytes
        precondition(firstEnd <= secondOffset || secondEnd <= firstOffset,
                     "Maple \(named) ranges must not overlap")
    }
}
