import Metal
import Testing
@testable import Mference

@Suite("Maple ternary primitives")
struct MapleTernaryGEMVTests {
    private static let d = MapleTernaryGEMV.hiddenSize
    private static let groups = d / Quantization.groupSize
    private static let sourceGroups = d / 128

    @Test("embedding writes BF16-only values at its selected output offset")
    func embeddingPreservesNativeBF16AndOutputBounds() throws {
        let tinyBits = Quantization.bf16Bits(1e-30)
        #expect(Quantization.bf16ToFloat(tinyBits) != 0)
        #expect(Float16(Quantization.bf16ToFloat(tinyBits)) == 0)

        let rows = 3
        let tablePrefix = 2
        let parameterPrefix = 3
        let outPrefix = 5
        let outSuffix = 4
        let table = [UInt8](repeating: 0xA5, count: tablePrefix)
            + [UInt8](repeating: 0, count: rows * Self.d / 2)
            + [UInt8](repeating: 0x5A, count: 3)
        let scales = [UInt16](repeating: 0xA5A5, count: parameterPrefix)
            + [UInt16](repeating: 0, count: rows * Self.groups)
            + [UInt16](repeating: 0x5A5A, count: 2)
        let biases = [UInt16](repeating: 0xA5A5, count: parameterPrefix)
            + [UInt16](repeating: tinyBits, count: rows * Self.groups)
            + [UInt16](repeating: 0x5A5A, count: 2)
        let outputSentinel = UInt16(0x7BAD)
        let output = [UInt16](repeating: outputSentinel, count: outPrefix + Self.d + outSuffix)

        let context = try MetalContext()
        let kernel = try MapleTernaryGEMV(context: context)
        guard let tableBuffer = Self.buffer(context.device, table),
              let scaleBuffer = Self.buffer(context.device, scales),
              let biasBuffer = Self.buffer(context.device, biases),
              let outBuffer = Self.buffer(context.device, output),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("Metal allocation failed")
            return
        }
        kernel.encodeEmbedding(
            commandBuffer: commandBuffer,
            table: tableBuffer, tableOffset: tablePrefix,
            scales: scaleBuffer, scalesOffset: parameterPrefix * MemoryLayout<UInt16>.stride,
            biases: biasBuffer, biasesOffset: parameterPrefix * MemoryLayout<UInt16>.stride,
            out: outBuffer, outOffset: outPrefix * MemoryLayout<UInt16>.stride,
            tokenID: 1)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        let actual: [UInt16] = Self.read(outBuffer, count: output.count)
        #expect(actual[..<outPrefix].allSatisfy { $0 == outputSentinel })
        #expect(actual[outPrefix..<(outPrefix + Self.d)].allSatisfy { $0 == tinyBits })
        #expect(actual.suffix(outSuffix).allSatisfy { $0 == outputSentinel })
    }

    @Test("ternary QMV uses widened INT4 codes and source groups of 128")
    func ternaryQMVMatchesExplicitSourceGroupReferenceIncludingNineRowTail() throws {
        let sourceGroups = Self.sourceGroups
        let rows = 9
        let weightPrefix = 1
        let parameterPrefix = 3
        let xPrefix = 4
        let yPrefix = 5
        let ySuffix = 3
        var payloadWeights = [UInt8](repeating: 0, count: rows * Self.d / 2)
        var payloadScales = [UInt16](repeating: 0, count: rows * Self.groups)
        var payloadBiases = [UInt16](repeating: 0, count: rows * Self.groups)
        let xBits = (0..<Self.d).map { index in
            Quantization.bf16Bits(Float((index * 7) % 13 - 6))
        }

        for row in 0..<rows {
            for index in 0..<Self.d {
                let code = UInt8((row * 5 + index * 3) & 3)
                let byte = row * Self.d / 2 + index / 2
                payloadWeights[byte] |= index.isMultiple(of: 2) ? code : code << 4
            }
            for source in 0..<sourceGroups {
                let alpha: Float = Float((row + 1) * (source + 1)) / 8
                let scale = Quantization.bf16Bits(alpha)
                let bias = Quantization.bf16Bits(-alpha)
                let group = source * 2
                payloadScales[row * Self.groups + group] = scale
                payloadScales[row * Self.groups + group + 1] = scale
                payloadBiases[row * Self.groups + group] = bias
                payloadBiases[row * Self.groups + group + 1] = bias
            }
        }
        let expected = Self.ternaryReference(
            weights: payloadWeights, scales: payloadScales, biases: payloadBiases, x: xBits, rows: rows)
        let weights = [UInt8](repeating: 0xA5, count: weightPrefix) + payloadWeights
            + [UInt8](repeating: 0x5A, count: 3)
        let scales = [UInt16](repeating: 0xA5A5, count: parameterPrefix) + payloadScales
            + [UInt16](repeating: 0x5A5A, count: 2)
        let biases = [UInt16](repeating: 0xA5A5, count: parameterPrefix) + payloadBiases
            + [UInt16](repeating: 0x5A5A, count: 2)
        let x = [UInt16](repeating: 0xA5A5, count: xPrefix) + xBits
            + [UInt16](repeating: 0x5A5A, count: 2)
        let ySentinel = UInt16(0x7BAD)
        let y = [UInt16](repeating: ySentinel, count: yPrefix + rows + ySuffix)

        let context = try MetalContext()
        let kernel = try MapleTernaryGEMV(context: context)
        guard let weightBuffer = Self.buffer(context.device, weights),
              let scaleBuffer = Self.buffer(context.device, scales),
              let biasBuffer = Self.buffer(context.device, biases),
              let xBuffer = Self.buffer(context.device, x),
              let yBuffer = Self.buffer(context.device, y),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("Metal allocation failed")
            return
        }
        kernel.encode(
            commandBuffer: commandBuffer,
            weights: weightBuffer, weightsOffset: weightPrefix,
            scales: scaleBuffer, scalesOffset: parameterPrefix * MemoryLayout<UInt16>.stride,
            biases: biasBuffer, biasesOffset: parameterPrefix * MemoryLayout<UInt16>.stride,
            x: xBuffer, xOffset: xPrefix * MemoryLayout<UInt16>.stride,
            y: yBuffer, yOffset: yPrefix * MemoryLayout<UInt16>.stride,
            rows: UInt32(rows))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        let actual: [UInt16] = Self.read(yBuffer, count: y.count)
        #expect(actual[..<yPrefix].allSatisfy { $0 == ySentinel })
        #expect(Array(actual[yPrefix..<(yPrefix + rows)]) == expected)
        #expect(actual.suffix(ySuffix).allSatisfy { $0 == ySentinel })
    }

    @Test("ternary QMV keeps a BF16-only activation out of FP16 storage")
    func ternaryQMVPersistsNativeBF16InputWithOffsets() throws {
        let d = Self.d
        let groups = Self.groups
        let tinyBits = Quantization.bf16Bits(1e-30)
        let prefix = 2
        let sentinel = UInt16(0x7BAD)
        var weights = [UInt8](repeating: 0xA5, count: prefix)
        var payload = [UInt8](repeating: 0, count: d / 2)
        payload[0] = 1
        weights += payload + [UInt8](repeating: 0x5A, count: 2)
        let scales = [UInt16](repeating: 0xA5A5, count: prefix)
            + [UInt16](repeating: Quantization.bf16Bits(1), count: groups)
            + [UInt16](repeating: 0x5A5A, count: 2)
        let biases = [UInt16](repeating: 0xA5A5, count: prefix)
            + [UInt16](repeating: 0, count: groups)
            + [UInt16](repeating: 0x5A5A, count: 2)
        let x = [UInt16](repeating: 0xA5A5, count: prefix)
            + [tinyBits] + [UInt16](repeating: 0, count: d - 1)
            + [UInt16](repeating: 0x5A5A, count: 2)
        let y = [UInt16](repeating: sentinel, count: prefix + 1 + 2)

        let context = try MetalContext()
        let kernel = try MapleTernaryGEMV(context: context)
        guard let weightBuffer = Self.buffer(context.device, weights),
              let scaleBuffer = Self.buffer(context.device, scales),
              let biasBuffer = Self.buffer(context.device, biases),
              let xBuffer = Self.buffer(context.device, x),
              let yBuffer = Self.buffer(context.device, y),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("Metal allocation failed")
            return
        }
        kernel.encode(
            commandBuffer: commandBuffer,
            weights: weightBuffer, weightsOffset: prefix,
            scales: scaleBuffer, scalesOffset: prefix * MemoryLayout<UInt16>.stride,
            biases: biasBuffer, biasesOffset: prefix * MemoryLayout<UInt16>.stride,
            x: xBuffer, xOffset: prefix * MemoryLayout<UInt16>.stride,
            y: yBuffer, yOffset: prefix * MemoryLayout<UInt16>.stride,
            rows: 1)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        let actual: [UInt16] = Self.read(yBuffer, count: y.count)
        #expect(actual[..<prefix].allSatisfy { $0 == sentinel })
        #expect(actual[prefix] == tinyBits)
        #expect(actual.suffix(2).allSatisfy { $0 == sentinel })
    }

    @Test("INT4 head rounds through BF16 before its FP16 export")
    func int4HeadMatchesExplicitGroup64ReferenceAcrossRowsAndOffsets() throws {
        let rows = 9
        let weightPrefix = 2
        let parameterPrefix = 4
        let xPrefix = 3
        let yPrefix = 6
        let ySuffix = 2
        var payloadWeights = [UInt8](repeating: 0, count: rows * Self.d / 2)
        var payloadScales = [UInt16](repeating: 0, count: rows * Self.groups)
        var payloadBiases = [UInt16](repeating: 0, count: rows * Self.groups)
        let xBits = (0..<Self.d).map { index in
            Quantization.bf16Bits(Float((index * 11) % 29 - 14) / 8)
        }
        for row in 0..<rows {
            for index in 0..<Self.d {
                let code = UInt8((row * 7 + index * 5) & 15)
                let byte = row * Self.d / 2 + index / 2
                payloadWeights[byte] |= index.isMultiple(of: 2) ? code : code << 4
            }
            for group in 0..<Self.groups {
                payloadScales[row * Self.groups + group] = Quantization.bf16Bits(
                    Float((row + 2) * (group + 1)) / 64)
                payloadBiases[row * Self.groups + group] = Quantization.bf16Bits(
                    Float((row - group) % 7) / 16)
            }
        }
        let expected = Self.int4Reference(
            weights: payloadWeights, scales: payloadScales, biases: payloadBiases, x: xBits, rows: rows)
        let weights = [UInt8](repeating: 0xA5, count: weightPrefix) + payloadWeights
            + [UInt8](repeating: 0x5A, count: 3)
        let scales = [UInt16](repeating: 0xA5A5, count: parameterPrefix) + payloadScales
            + [UInt16](repeating: 0x5A5A, count: 2)
        let biases = [UInt16](repeating: 0xA5A5, count: parameterPrefix) + payloadBiases
            + [UInt16](repeating: 0x5A5A, count: 2)
        let x = [UInt16](repeating: 0xA5A5, count: xPrefix) + xBits
            + [UInt16](repeating: 0x5A5A, count: 2)
        let ySentinel = Float16(-777)
        let y = [Float16](repeating: ySentinel, count: yPrefix + rows + ySuffix)

        let context = try MetalContext()
        let kernel = try MapleTernaryGEMV(context: context)
        guard let weightBuffer = Self.buffer(context.device, weights),
              let scaleBuffer = Self.buffer(context.device, scales),
              let biasBuffer = Self.buffer(context.device, biases),
              let xBuffer = Self.buffer(context.device, x),
              let yBuffer = Self.buffer(context.device, y),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("Metal allocation failed")
            return
        }
        kernel.encodeInt4(
            commandBuffer: commandBuffer,
            weights: weightBuffer, weightsOffset: weightPrefix,
            scales: scaleBuffer, scalesOffset: parameterPrefix * MemoryLayout<UInt16>.stride,
            biases: biasBuffer, biasesOffset: parameterPrefix * MemoryLayout<UInt16>.stride,
            x: xBuffer, xOffset: xPrefix * MemoryLayout<UInt16>.stride,
            y: yBuffer, yOffset: yPrefix * MemoryLayout<Float16>.stride,
            rows: UInt32(rows))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        let actual: [Float16] = Self.read(yBuffer, count: y.count)
        #expect(actual[..<yPrefix].allSatisfy { $0 == ySentinel })
        #expect(Array(actual[yPrefix..<(yPrefix + rows)]) == expected)
        #expect(actual.suffix(ySuffix).allSatisfy { $0 == ySentinel })
    }

    private static func ternaryReference(weights: [UInt8], scales: [UInt16],
                                         biases: [UInt16], x: [UInt16], rows: Int) -> [UInt16] {
        (0..<rows).map { row in
            var total: Float = 0
            for source in 0..<sourceGroups {
                var dot: Float = 0
                var sum: Float = 0
                for index in (source * 128)..<((source + 1) * 128) {
                    let code = index.isMultiple(of: 2)
                        ? weights[row * d / 2 + index / 2] & 15
                        : weights[row * d / 2 + index / 2] >> 4
                    let value = Quantization.bf16ToFloat(x[index])
                    dot += Float(code) * value
                    sum += value
                }
                let parameter = row * groups + source * 2
                total += Quantization.bf16ToFloat(scales[parameter]) * dot
                    + Quantization.bf16ToFloat(biases[parameter]) * sum
            }
            return Quantization.bf16Bits(total)
        }
    }

    private static func int4Reference(weights: [UInt8], scales: [UInt16],
                                      biases: [UInt16], x: [UInt16], rows: Int) -> [Float16] {
        (0..<rows).map { row in
            var total: Float = 0
            for group in 0..<groups {
                var dot: Float = 0
                var sum: Float = 0
                for index in (group * Quantization.groupSize)..<((group + 1) * Quantization.groupSize) {
                    let code = index.isMultiple(of: 2)
                        ? weights[row * d / 2 + index / 2] & 15
                        : weights[row * d / 2 + index / 2] >> 4
                    let value = Quantization.bf16ToFloat(x[index])
                    dot += Float(code) * value
                    sum += value
                }
                let parameter = row * groups + group
                total += Quantization.bf16ToFloat(scales[parameter]) * dot
                    + Quantization.bf16ToFloat(biases[parameter]) * sum
            }
            return Float16(Quantization.bf16ToFloat(Quantization.bf16Bits(total)))
        }
    }

    private static func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
    }

    private static func read<T>(_ buffer: MTLBuffer, count: Int) -> [T] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: T.self, capacity: count), count: count))
    }
}
