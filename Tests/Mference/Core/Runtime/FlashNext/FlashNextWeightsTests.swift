import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct FlashNextWeightsTests {
    @Test(arguments: [4, 8], [64, 2560])
    func packedReferenceUsesActualWidthAndCompanionOffsets(bits: Int, columns: Int) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let rows = 3
        let count = rows * columns
        let groups = count / 64
        let bytes = count * bits / 8
        let weightOffset = 32
        let scaleOffset = weightOffset + bytes + 16
        let biasOffset = scaleOffset + groups * 2 + 16
        let buffer = try #require(device.makeBuffer(length: biasOffset + groups * 2 + 32,
                                                    options: .storageModeShared))
        memset(buffer.contents(), 0xa5, buffer.length)
        let packed = buffer.contents().advanced(by: weightOffset).assumingMemoryBound(to: UInt8.self)
        packed.update(repeating: 0, count: bytes)
        let scales = buffer.contents().advanced(by: scaleOffset).assumingMemoryBound(to: UInt16.self)
        let biases = buffer.contents().advanced(by: biasOffset).assumingMemoryBound(to: UInt16.self)
        for group in 0..<groups {
            scales[group] = Quantization.bf16Bits(Float(group % 7 + 1) / 32)
            biases[group] = Quantization.bf16Bits(Float(group % 9 - 4) / 8)
        }
        var expected: [Float] = []
        for index in 0..<count {
            let q = (index * 29 + 17) % (1 << bits)
            if bits == 8 { packed[index] = UInt8(q) }
            else { packed[index / 2] |= UInt8(q << ((index % 2) * 4)) }
            // Independent scalar affine rule, not the production dequantizer.
            expected.append(Float(q) * Quantization.bf16ToFloat(scales[index / 64])
                            + Quantization.bf16ToFloat(biases[index / 64]))
        }
        let view = TensorView(buffer: buffer, offset: UInt64(weightOffset), length: UInt64(bytes),
            scaleOffset: UInt64(scaleOffset), scaleLength: UInt64(groups * 2),
            biasOffset: UInt64(biasOffset), biasLength: UInt64(groups * 2),
            shape: (UInt32(rows), UInt32(columns), 0, 0), dtype: 0)
        let actual = FlashNextWeights.read(view)
        #expect(actual.count == count)
        #expect(actual.allSatisfy { $0.isFinite })
        #expect(actual == expected)
    }

    @Test(arguments: [UInt8(1), UInt8(3)])
    func denseReferencePreservesOffsets(dtype: UInt8) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let expected: [Float] = [-2, -0.5, 0, 0.25, 1, 4]
        let stride = dtype == 1 ? 2 : 4
        let buffer = try #require(device.makeBuffer(length: 128, options: .storageModeShared))
        memset(buffer.contents(), 0xa5, buffer.length)
        for (index, value) in expected.enumerated() {
            let pointer = buffer.contents().advanced(by: 16 + index * stride)
            if dtype == 1 { pointer.storeBytes(of: Quantization.bf16Bits(value), as: UInt16.self) }
            else { pointer.storeBytes(of: value, as: Float.self) }
        }
        let view = TensorView(buffer: buffer, offset: 16, length: UInt64(expected.count * stride),
            scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
            shape: (2, 3, 0, 0), dtype: dtype)
        #expect(FlashNextWeights.read(view) == expected)
    }

    /// An installed-weight numerical gate, not a full draft or acceptance test.
    /// The CPU reader above has an independent byte-level scalar oracle. Here
    /// scalar Double dots compare it with actual GPU projection dispatch.
    @Test func installedMTPProjectionsMatchScalarDots() throws {
        guard let path = ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_GTURBO"] else { return }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path),
            device: context.device, streamingMode: .pread(slotCount: 16))
        let weights = try FlashNextMTPWeights(model: model)
        let columns = model.config.hiddenSize
        let x = (0..<columns).map { Float16(Float($0 % 31 - 15) / 32) }
        let input = try #require(context.device.makeBuffer(bytes: x, length: columns * 2,
                                                           options: .storageModeShared))
        let matvec = try FlashNextMatVec(context: context,
            int4: DequantInt4GEMV(context: context), int8Columns: columns)
        let projections: [(String, FlashNextWeightMatrix)] = [
            ("mtp.fc_embedding.weight", weights.embeddingProjection),
            ("mtp.fc_hidden.weight", weights.hiddenProjection),
            ("mtp.layers.0.mlp.gate.weight", weights.router),
            ("mtp.layers.0.mlp.shared_expert_gate.weight", weights.sharedGate),
        ]
        for (name, projection) in projections {
            let view = try model.resident(name: name)
            let rows = Int(view.shape.0)
            let values = FlashNextWeights.read(view)
            try #require(values.count == rows * columns)
            let output = try #require(context.device.makeBuffer(length: rows * 4, options: .storageModeShared))
            let actual = output.contents().assumingMemoryBound(to: Float.self)
            actual.update(repeating: .nan, count: rows)
            let cb = try #require(context.queue.makeCommandBuffer())
            matvec.encode(commandBuffer: cb, matrix: projection, x: input, y: output,
                          rows: rows, cols: columns, outputFloat32: true)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.error == nil)
            var maximumError: Float = 0
            var scale: Float = 0
            for row in 0..<rows {
                var sum: Double = 0
                for column in 0..<columns {
                    sum += Double(values[row * columns + column]) * Double(x[column])
                }
                let expected = Float(sum)
                try #require(expected.isFinite && actual[row].isFinite)
                maximumError = max(maximumError, abs(actual[row] - expected))
                scale = max(scale, abs(expected))
            }
            #expect(scale > 0)
            #expect(maximumError <= 0.002 * max(scale, 1))
            print("[MTP installed projection] \(name) rows=\(rows) maxAbs=\(maximumError) scale=\(scale)")
        }
    }
}
