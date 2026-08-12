import Metal
import Testing
@testable import Mference

@Suite("Maple BF16 residual and norms")
struct MapleAddRMSNormTests {
    private static let d = MapleAddRMSNorm.hiddenSize
    private static let eps: Float = 1e-6

    @Test("residual and final norm retain their native-BF16 boundaries and offsets")
    func residualAndFinalNormPreserveBF16BoundariesAndBounds() throws {
        let d = Self.d
        let hiddenPrefix = 3, hiddenSuffix = 2
        let deltaPrefix = 4, deltaSuffix = 3
        let residualWeightPrefix = 5, finalWeightPrefix = 6
        let residualPrefix = 7, residualSuffix = 2
        let finalPrefix = 8, finalSuffix = 3
        var hiddenPayload = (0..<Self.d).map { Quantization.bf16Bits(Float(($0 * 9) % 23 - 11) / 16) }
        var deltaPayload = (0..<Self.d).map { Quantization.bf16Bits(Float(($0 * 5) % 17 - 8) / 32) }
        hiddenPayload[0] = Quantization.bf16Bits(1e-30)
        deltaPayload[0] = Quantization.bf16Bits(2e-30)
        let residualWeight = (0..<Self.d).map { Quantization.bf16Bits(Float(($0 % 7) + 5) / 8) }
        let finalWeight = (0..<Self.d).map { Quantization.bf16Bits(Float(($0 % 11) + 3) / 8) }
        let hiddenSentinel = UInt16(0x7BAD)
        let outputSentinel = UInt16(0x6BAD)
        let hidden = [UInt16](repeating: hiddenSentinel, count: hiddenPrefix) + hiddenPayload
            + [UInt16](repeating: hiddenSentinel, count: hiddenSuffix)
        let delta = [UInt16](repeating: 0xA5A5, count: deltaPrefix) + deltaPayload
            + [UInt16](repeating: 0x5A5A, count: deltaSuffix)
        let residualWeights = [UInt16](repeating: 0xA5A5, count: residualWeightPrefix) + residualWeight
            + [UInt16](repeating: 0x5A5A, count: 2)
        let finalWeights = [UInt16](repeating: 0xA5A5, count: finalWeightPrefix) + finalWeight
            + [UInt16](repeating: 0x5A5A, count: 2)
        let residualOutput = [UInt16](repeating: outputSentinel, count: residualPrefix + Self.d + residualSuffix)
        let finalOutput = [UInt16](repeating: outputSentinel, count: finalPrefix + Self.d + finalSuffix)
        let zeroDelta = [UInt16](repeating: 0, count: Self.d)

        let expectedHidden = zip(hiddenPayload, deltaPayload).map { hidden, delta in
            Quantization.bf16Bits(Quantization.bf16ToFloat(hidden) + Quantization.bf16ToFloat(delta))
        }
        let expectedResidual = Self.normReference(hidden: expectedHidden, weight: residualWeight)
        let expectedFinal = Self.normReference(hidden: expectedHidden, weight: finalWeight)

        let context = try MetalContext()
        let kernel = try MapleAddRMSNorm(context: context)
        guard let hiddenBuffer = Self.buffer(context.device, hidden),
              let deltaBuffer = Self.buffer(context.device, delta),
              let residualWeightBuffer = Self.buffer(context.device, residualWeights),
              let finalWeightBuffer = Self.buffer(context.device, finalWeights),
              let residualBuffer = Self.buffer(context.device, residualOutput),
              let finalBuffer = Self.buffer(context.device, finalOutput),
              let zeroBuffer = Self.buffer(context.device, zeroDelta),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("Metal allocation failed")
            return
        }
        kernel.encode(
            commandBuffer: commandBuffer,
            hidden: hiddenBuffer, hiddenOffset: hiddenPrefix * MemoryLayout<UInt16>.stride,
            delta: deltaBuffer, deltaOffset: deltaPrefix * MemoryLayout<UInt16>.stride,
            weight: residualWeightBuffer, weightOffset: residualWeightPrefix * MemoryLayout<UInt16>.stride,
            normed: residualBuffer, normedOffset: residualPrefix * MemoryLayout<UInt16>.stride,
            eps: Self.eps)
        kernel.encode(
            commandBuffer: commandBuffer,
            hidden: hiddenBuffer, hiddenOffset: hiddenPrefix * MemoryLayout<UInt16>.stride,
            delta: zeroBuffer,
            weight: finalWeightBuffer, weightOffset: finalWeightPrefix * MemoryLayout<UInt16>.stride,
            normed: finalBuffer, normedOffset: finalPrefix * MemoryLayout<UInt16>.stride,
            eps: Self.eps)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        let actualHidden: [UInt16] = Self.read(hiddenBuffer, count: hidden.count)
        #expect(actualHidden[..<hiddenPrefix].allSatisfy { $0 == hiddenSentinel })
        #expect(Array(actualHidden[hiddenPrefix..<(hiddenPrefix + Self.d)]) == expectedHidden)
        #expect(actualHidden.suffix(hiddenSuffix).allSatisfy { $0 == hiddenSentinel })
        #expect(Quantization.bf16ToFloat(expectedHidden[0]) != 0)
        #expect(Float16(Quantization.bf16ToFloat(expectedHidden[0])) == 0)
        Self.expectNorm(residualBuffer, prefix: residualPrefix, suffix: residualSuffix,
                        expected: expectedResidual, sentinel: outputSentinel)
        Self.expectNorm(finalBuffer, prefix: finalPrefix, suffix: finalSuffix,
                        expected: expectedFinal, sentinel: outputSentinel)
    }

    private static func normReference(hidden: [UInt16], weight: [UInt16]) -> [UInt16] {
        let values = hidden.map(Quantization.bf16ToFloat)
        let sum = values.reduce(Float.zero) { $0 + $1 * $1 }
        let scale = 1 / sqrt(sum / Float(d) + eps)
        return zip(values, weight).map { value, weight in
            Quantization.bf16Bits(value * scale * Quantization.bf16ToFloat(weight))
        }
    }

    private static func expectNorm(_ buffer: MTLBuffer, prefix: Int, suffix: Int,
                                   expected: [UInt16], sentinel: UInt16) {
        let actual: [UInt16] = Self.read(buffer, count: prefix + d + suffix)
        #expect(actual[..<prefix].allSatisfy { $0 == sentinel })
        #expect(actual.suffix(suffix).allSatisfy { $0 == sentinel })
        let maximum = zip(actual[prefix..<(prefix + d)], expected).map { actual, expected in
            abs(Quantization.bf16ToFloat(actual) - Quantization.bf16ToFloat(expected))
        }.max() ?? 0
        #expect(maximum < 0.02, "BF16 norm maxAbsDiff=\(maximum)")
    }

    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
    }

    private static func read<T>(_ buffer: MTLBuffer, count: Int) -> [T] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: T.self, capacity: count), count: count))
    }
}
