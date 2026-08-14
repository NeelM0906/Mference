import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct MapleMoETests {
    private let d = MapleMoE.dimension
    private let f = MapleMoE.intermediate
    private let topK = MapleMoE.topK

    private struct ExpertFixture {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test("Maple router keeps BF16 inputs, stable ties, and full-softmax top-8 weights")
    func routerBF16FullSoftmaxAndOffsets() throws {
        let context = try MetalContext()
        let moe = try MapleMoE(context: context)

        var hidden = [Float](repeating: 0, count: d)
        hidden[0] = 1
        hidden[1] = bf16(1.0e-30) // Native BF16, outside FP16's finite range.
        let tiedScores: [Int: Float] = [3: 8, 55: 8, 7: 7, 63: 7,
                                        11: 6, 71: 6, 19: 5, 95: 5]
        let nativeRangeWeights: [Int: Float] = [
            3: 1.0e30, 55: 1.0e30,
            7: 8.0e29, 63: 8.0e29,
            11: 6.0e29, 71: 6.0e29,
            19: 4.0e29, 95: 4.0e29,
        ]
        var router = [UInt16](repeating: Quantization.bf16Bits(-4), count: 256 * d)
        let nativeRangeContribution = bf16(1.0e30) * hidden[1]
        for expert in 0..<256 {
            router[expert * d] = Quantization.bf16Bits(tiedScores[expert] ?? -4)
            // Pair-dependent products preserve the deliberate ties while
            // making BF16 values outside FP16's range observable in weights.
            router[expert * d + 1] = Quantization.bf16Bits(
                nativeRangeWeights[expert] ?? 2.0e29)
        }

        let weightPayload = bytes(router)
        let hiddenPayload = bytes(hidden.map(Quantization.bf16Bits))
        let weight = try paddedBuffer(context.device, prefix: 2, payload: weightPayload, suffix: 3)
        let x = try paddedBuffer(context.device, prefix: 4, payload: hiddenPayload, suffix: 2)
        let indices = try paddedBuffer(context.device, prefix: 4,
                                       payload: [UInt8](repeating: 0, count: topK * 4), suffix: 4)
        let weights = try paddedBuffer(context.device, prefix: 4,
                                       payload: [UInt8](repeating: 0, count: topK * 4), suffix: 4)
        let command = try #require(context.queue.makeCommandBuffer())
        moe.encodeRouterTop8(commandBuffer: command,
                             weights: weight.buffer, weightsOffset: 2,
                             hidden: x.buffer, hiddenOffset: 4,
                             indices: indices.buffer, indicesOffset: 4,
                             routingWeights: weights.buffer, routingWeightsOffset: 4)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.error == nil)

        let expected = routerTop8(router, hidden: hidden)

        #expect(nativeRangeContribution != 0)
        #expect(readUInt32(indices.buffer, offset: 4, count: topK) == expected.indices)
        let actualWeights = readFloat(weights.buffer, offset: 4, count: topK)
        let routerWeightDiff = maxAbs(actualWeights, expected.weights)
        #expect(routerWeightDiff <= 0.000_002,
                "Maple router max weight diff: \(routerWeightDiff)")
        #expect(abs(actualWeights.reduce(Float(0), +) - 1) <= 0.000_002)
        assertSentinels(weight)
        assertSentinels(x)
        assertSentinels(indices)
        assertSentinels(weights)
    }

    @Test("Maple router is deterministic across serial command-buffer reuse")
    func routerDeterminismAcrossSerialReuse() throws {
        let context = try MetalContext()
        let moe = try MapleMoE(context: context)
        let firstIDs = [5, 17, 29, 41, 53, 65, 77, 89]
        let secondIDs = [159, 171, 183, 195, 207, 219, 231, 243]
        var firstRouter = [UInt16](repeating: Quantization.bf16Bits(-16), count: 256 * d)
        var secondRouter = [UInt16](repeating: Quantization.bf16Bits(-16), count: 256 * d)
        for (expert, logit) in zip(firstIDs, [8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5] as [Float]) {
            firstRouter[expert * d] = Quantization.bf16Bits(logit)
        }
        for (expert, logit) in zip(secondIDs, [8, 4, 0, -4, -8, -10, -12, -14] as [Float]) {
            secondRouter[expert * d + 1] = Quantization.bf16Bits(logit)
        }
        var firstHidden = [Float](repeating: 0, count: d)
        var secondHidden = [Float](repeating: 0, count: d)
        firstHidden[0] = bf16(1)
        secondHidden[1] = bf16(1)
        let expected = [routerTop8(firstRouter, hidden: firstHidden),
                        routerTop8(secondRouter, hidden: secondHidden)]
        #expect(Set(expected[0].indices).isDisjoint(with: Set(expected[1].indices)))
        #expect(abs(expected[0].weights[0] - expected[1].weights[0]) > 0.2)

        let routers = try [firstRouter, secondRouter].map {
            try paddedBuffer(context.device, prefix: 2, payload: bytes($0), suffix: 3)
        }
        let hidden = try [firstHidden, secondHidden].map {
            try paddedBuffer(context.device, prefix: 4,
                             payload: bytes($0.map(Quantization.bf16Bits)), suffix: 2)
        }
        let indices = try paddedBuffer(context.device, prefix: 4,
                                       payload: [UInt8](repeating: 0, count: topK * 4), suffix: 4)
        let weights = try paddedBuffer(context.device, prefix: 4,
                                       payload: [UInt8](repeating: 0, count: topK * 4), suffix: 4)
        var baselineWeights = [[UInt32]?](repeating: nil, count: 2)

        for iteration in 0..<3_072 {
            let fixture = iteration & 1
            let command = try #require(context.queue.makeCommandBuffer())
            moe.encodeRouterTop8(commandBuffer: command,
                                 weights: routers[fixture].buffer, weightsOffset: 2,
                                 hidden: hidden[fixture].buffer, hiddenOffset: 4,
                                 indices: indices.buffer, indicesOffset: 4,
                                 routingWeights: weights.buffer, routingWeightsOffset: 4)
            command.commit()
            command.waitUntilCompleted()
            #expect(command.error == nil)

            let actualIndices = readUInt32(indices.buffer, offset: 4, count: topK)
            let actualWeights = readFloat(weights.buffer, offset: 4, count: topK)
            #expect(actualIndices == expected[fixture].indices)
            #expect(actualWeights.allSatisfy { $0.isFinite })
            #expect(abs(actualWeights.reduce(Float(0), +) - 1) <= 0.000_002)
            if let baseline = baselineWeights[fixture] {
                #expect(actualWeights.map(\.bitPattern) == baseline)
            } else {
                #expect(maxAbs(actualWeights, expected[fixture].weights) <= 0.000_002)
                baselineWeights[fixture] = actualWeights.map(\.bitPattern)
            }
        }
        routers.forEach(assertSentinels)
        hidden.forEach(assertSentinels)
        assertSentinels(indices)
        assertSentinels(weights)
    }

    @Test("Maple router emits sentinels for non-finite BF16 values")
    func routerNonFiniteEmitsSentinels() throws {
        let context = try MetalContext()
        let moe = try MapleMoE(context: context)
        var finiteHidden = [UInt16](repeating: 0, count: d)
        finiteHidden[0] = Quantization.bf16Bits(1)
        var nanHidden = finiteHidden
        nanHidden[0] = Quantization.bf16Bits(.nan)
        let finiteRouter = [UInt16](repeating: 0, count: 256 * d)
        var nanRouter = finiteRouter
        nanRouter[0] = Quantization.bf16Bits(.nan)
        var positiveInfinityRouter = finiteRouter
        positiveInfinityRouter[0] = Quantization.bf16Bits(.infinity)
        var negativeInfinityRouter = finiteRouter
        negativeInfinityRouter[0] = Quantization.bf16Bits(-Float.infinity)
        let routers = try [finiteRouter, nanRouter, positiveInfinityRouter, negativeInfinityRouter].map {
            try paddedBuffer(context.device, prefix: 2, payload: bytes($0), suffix: 3)
        }
        let hidden = try [nanHidden, finiteHidden, finiteHidden, finiteHidden].map {
            try paddedBuffer(context.device, prefix: 4, payload: bytes($0), suffix: 2)
        }
        let indices = try paddedBuffer(context.device, prefix: 4,
                                       payload: [UInt8](repeating: 0, count: topK * 4), suffix: 4)
        let weights = try paddedBuffer(context.device, prefix: 4,
                                       payload: [UInt8](repeating: 0, count: topK * 4), suffix: 4)

        for fixture in routers.indices {
            let command = try #require(context.queue.makeCommandBuffer())
            moe.encodeRouterTop8(commandBuffer: command,
                                 weights: routers[fixture].buffer, weightsOffset: 2,
                                 hidden: hidden[fixture].buffer, hiddenOffset: 4,
                                 indices: indices.buffer, indicesOffset: 4,
                                 routingWeights: weights.buffer, routingWeightsOffset: 4)
            command.commit()
            command.waitUntilCompleted()
            #expect(command.error == nil)
            #expect(readUInt32(indices.buffer, offset: 4, count: topK) == [UInt32](repeating: 256, count: topK))
            #expect(readFloat(weights.buffer, offset: 4, count: topK) == [Float](repeating: 0, count: topK))
        }
        routers.forEach(assertSentinels)
        hidden.forEach(assertSentinels)
        assertSentinels(indices)
        assertSentinels(weights)
    }

    @Test("Maple INT2 source-group128 experts preserve rank slots across full, subset, and permutation paths")
    func expertsSourceGroup128SubsetAndPermutation() throws {
        let context = try MetalContext()
        let moe = try MapleMoE(context: context)
        let fixtures = (0..<topK).map { makeExpert($0) }
        let offsets = fixtures[0].offsets
        // Nonzero prefixes model slot-slab slices: expert data at a nonzero
        // offset inside a larger cache buffer.
        let originalBlobs: [MapleMoE.RoutedBlob] = try fixtures.map { fixture in
            let padded = try paddedBuffer(context.device, prefix: 64,
                                          payload: fixture.bytes, suffix: 32)
            return (buffer: padded.buffer, offset: padded.prefix, length: fixture.bytes.count)
        }
        var x = (0..<d).map { index -> Float in
            index == 1 ? bf16(1.0e-30) : bf16(0.09 + Float(index % 23) * 0.006)
        }
        x[0] = bf16(0.31)
        let routing: [Float] = [0.25, 0.1875, 0.15625, 0.125, 0.109375, 0.078125, 0.0625, 0.03125]
        let expected = reference(fixtures: fixtures, x: x, routing: routing)

        let xBuffer = try paddedBuffer(context.device, prefix: 2,
                                       payload: bytes(x.map(Quantization.bf16Bits)), suffix: 3)
        let routingBuffer = try paddedBuffer(context.device, prefix: 4,
                                             payload: bytes(routing), suffix: 4)
        let fullActs = try paddedBuffer(context.device, prefix: 2,
                                        payload: [UInt8](repeating: 0, count: topK * f * 2), suffix: 2)
        let splitActs = try paddedBuffer(context.device, prefix: 2,
                                         payload: [UInt8](repeating: 0, count: topK * f * 2), suffix: 2)
        let fullOutput = try paddedBuffer(context.device, prefix: 2,
                                          payload: [UInt8](repeating: 0, count: d * 2), suffix: 3)
        let splitOutput = try paddedBuffer(context.device, prefix: 2,
                                           payload: [UInt8](repeating: 0, count: d * 2), suffix: 3)

        let fullArguments = moe.makeRoutedArgumentBuffer(routedBlobs: originalBlobs, offsets: offsets)
        let full = try #require(context.queue.makeCommandBuffer())
        moe.encodePhase1(commandBuffer: full, routedArgumentBuffer: fullArguments,
                         routedBlobs: originalBlobs, offsets: offsets,
                         x: xBuffer.buffer, xOffset: 2, acts: fullActs.buffer, actsOffset: 2)
        moe.encodePhase2(commandBuffer: full, routedArgumentBuffer: fullArguments,
                         routedBlobs: originalBlobs, offsets: offsets,
                         acts: fullActs.buffer, actsOffset: 2,
                         routingWeights: routingBuffer.buffer, routingWeightsOffset: 4,
                         output: fullOutput.buffer, outputOffset: 2)
        full.commit()
        full.waitUntilCompleted()
        #expect(full.error == nil)

        let splitArguments = moe.makeRoutedArgumentBuffer(routedBlobs: originalBlobs, offsets: offsets)
        let split = try #require(context.queue.makeCommandBuffer())
        for slots in [[UInt32(6), 1, 4], [UInt32(0), 7, 2, 5, 3]] {
            moe.encodePhase1Subset(commandBuffer: split, routedArgumentBuffer: splitArguments,
                                   routedBlobs: originalBlobs, offsets: offsets,
                                   x: xBuffer.buffer, xOffset: 2, acts: splitActs.buffer, actsOffset: 2,
                                   activeSlotIndices: slots)
        }
        moe.encodePhase2(commandBuffer: split, routedArgumentBuffer: splitArguments,
                         routedBlobs: originalBlobs, offsets: offsets,
                         acts: splitActs.buffer, actsOffset: 2,
                         routingWeights: routingBuffer.buffer, routingWeightsOffset: 4,
                         output: splitOutput.buffer, outputOffset: 2)
        split.commit()
        split.waitUntilCompleted()
        #expect(split.error == nil)

        let permutation = [3, 6, 0, 7, 2, 5, 1, 4]
        let permutedBlobs = permutation.map { originalBlobs[$0] }
        let permutedRouting = permutation.map { routing[$0] }
        let permutedRoutingBuffer = try paddedBuffer(context.device, prefix: 4,
                                                      payload: bytes(permutedRouting), suffix: 4)
        let permutedActs = try paddedBuffer(context.device, prefix: 2,
                                            payload: [UInt8](repeating: 0, count: topK * f * 2), suffix: 2)
        let permutedOutput = try paddedBuffer(context.device, prefix: 2,
                                              payload: [UInt8](repeating: 0, count: d * 2), suffix: 3)
        let permutedArguments = moe.makeRoutedArgumentBuffer(routedBlobs: permutedBlobs, offsets: offsets)
        let permuted = try #require(context.queue.makeCommandBuffer())
        moe.encodePhase1(commandBuffer: permuted, routedArgumentBuffer: permutedArguments,
                         routedBlobs: permutedBlobs, offsets: offsets,
                         x: xBuffer.buffer, xOffset: 2, acts: permutedActs.buffer, actsOffset: 2)
        moe.encodePhase2(commandBuffer: permuted, routedArgumentBuffer: permutedArguments,
                         routedBlobs: permutedBlobs, offsets: offsets,
                         acts: permutedActs.buffer, actsOffset: 2,
                         routingWeights: permutedRoutingBuffer.buffer, routingWeightsOffset: 4,
                         output: permutedOutput.buffer, outputOffset: 2)
        permuted.commit()
        permuted.waitUntilCompleted()
        #expect(permuted.error == nil)

        let fullBits = readUInt16(fullOutput.buffer, offset: 2, count: d)
        let splitBits = readUInt16(splitOutput.buffer, offset: 2, count: d)
        let permutedBits = readUInt16(permutedOutput.buffer, offset: 2, count: d)
        let fullActBits = readUInt16(fullActs.buffer, offset: 2, count: topK * f)
        let splitActBits = readUInt16(splitActs.buffer, offset: 2, count: topK * f)
        #expect(fullActBits == splitActBits)
        #expect(fullBits == splitBits)
        #expect(fullBits == permutedBits)
        #expect(maxAbs(fullBits.map(Quantization.bf16ToFloat), expected) <= 0.0078125)
        let firstActs = fullActBits.map(Quantization.bf16ToFloat)
        #expect(firstActs[0] > 45 && firstActs[1] < -45,
                "fixture must exercise both MoE ±7 clamp edges")
        [xBuffer, routingBuffer, fullActs, splitActs, fullOutput, splitOutput,
         permutedRoutingBuffer, permutedActs, permutedOutput].forEach(assertSentinels)
    }

    private func makeExpert(_ expert: Int) -> ExpertFixture {
        var bytes = [UInt8](repeating: 0xA7, count: 3) // weight offsets may be byte-aligned.
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func alignBF16() { if !bytes.count.isMultiple(of: 2) { bytes.append(0xD3) } }
        func append(_ values: [UInt16]) { values.forEach { value in
            bytes.append(UInt8(truncatingIfNeeded: value))
            bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        } }
        func projection(rows: Int, columns: Int, kind: Int) -> (UInt32, UInt32, UInt32) {
            let w = UInt32(bytes.count)
            var codes = [UInt8](repeating: 0, count: rows * columns / 4)
            for row in 0..<rows {
                for column in 0..<columns {
                    let code: UInt8
                    if kind == 0 && row < 2 {
                        code = 3
                    } else if kind == 1 && row == 0 {
                        code = 3
                    } else if kind == 1 && row == 1 {
                        code = 0
                    } else {
                        code = UInt8((row * 5 + column * 3 + expert * 7 + kind) & 3)
                    }
                    let index = row * columns + column
                    codes[index / 4] |= code << UInt8(2 * (index & 3))
                }
            }
            append(codes)
            alignBF16()
            let s = UInt32(bytes.count)
            let sourceGroups = columns / 128
            var scales = [UInt16]()
            for row in 0..<rows {
                for source in 0..<sourceGroups {
                    let alpha: Float = row < 2 && kind < 2
                        ? 0.08 : 0.0025 + Float((row + source + expert + kind) % 7) * 0.0005
                    let bits = Quantization.bf16Bits(alpha)
                    scales.append(bits)
                    scales.append(bits) // Source group-128 expands into two group-64 entries.
                }
            }
            append(scales)
            let b = UInt32(bytes.count)
            append(scales.map { $0 ^ 0x8000 })
            return (w, s, b)
        }
        let gate = projection(rows: f, columns: d, kind: 0)
        let up = projection(rows: f, columns: d, kind: 1)
        let down = projection(rows: d, columns: f, kind: 2)
        bytes.append(contentsOf: [0xB1, 0xB2, 0xB3])
        return ExpertFixture(bytes: bytes, offsets: MoEExpertOffsets(
            gateWOff: gate.0, gateSOff: gate.1, gateBOff: gate.2,
            upWOff: up.0, upSOff: up.1, upBOff: up.2,
            downWOff: down.0, downSOff: down.1, downBOff: down.2))
    }

    private func reference(fixtures: [ExpertFixture], x: [Float], routing: [Float]) -> [Float] {
        let outputs = fixtures.map { fixture -> [Float] in
            let gate = (0..<f).map { row in bf16(qmv(fixture, offset: fixture.offsets.gateWOff,
                                                      scale: fixture.offsets.gateSOff,
                                                      bias: fixture.offsets.gateBOff,
                                                      row: row, columns: d, x: x)) }
            let up = (0..<f).map { row in bf16(qmv(fixture, offset: fixture.offsets.upWOff,
                                                    scale: fixture.offsets.upSOff,
                                                    bias: fixture.offsets.upBOff,
                                                    row: row, columns: d, x: x)) }
            let acts = zip(gate, up).map(swiglu)
            return (0..<d).map { row in bf16(qmv(fixture, offset: fixture.offsets.downWOff,
                                                  scale: fixture.offsets.downSOff,
                                                  bias: fixture.offsets.downBOff,
                                                  row: row, columns: f, x: acts)) }
        }
        return (0..<d).map { row in
            var sum: Float = 0
            for rank in 0..<topK { sum += routing[rank] * outputs[rank][row] }
            return bf16(sum)
        }
    }

    private func qmv(_ fixture: ExpertFixture, offset: UInt32, scale: UInt32, bias: UInt32,
                            row: Int, columns: Int, x: [Float]) -> Float {
        let rowBytes = columns / 4
        let runtimeGroups = columns / 64
        var lanes = [Float](repeating: 0, count: 32)
        for lane in 0..<32 {
            var result: Float = 0
            for block in stride(from: 0, to: columns, by: 512) {
                let base = block + lane * 16
                var values = [Float](repeating: 0, count: 16)
                var inputSum: Float = 0
                for i in stride(from: 0, to: 16, by: 4) {
                    let x0 = bf16(x[base + i]), x1 = bf16(x[base + i + 1])
                    let x2 = bf16(x[base + i + 2]), x3 = bf16(x[base + i + 3])
                    inputSum += bf16(bf16(bf16(x0 + x1) + x2) + x3)
                    values[i] = x0; values[i + 1] = x1 / 4
                    values[i + 2] = x2 / 16; values[i + 3] = x3 / 64
                }
                var dot: Float = 0
                let packed = Int(offset) + row * rowBytes + block / 4 + lane * 4
                for i in 0..<4 {
                    let code = fixture.bytes[packed + i]
                    let j = i * 4
                    dot += values[j] * Float(code & 3)
                        + values[j + 1] * Float(code & 12)
                        + values[j + 2] * Float(code & 48)
                        + values[j + 3] * Float(code & 192)
                }
                let source = block / 128 + lane / 8
                let metadata = row * runtimeGroups + source * 2
                let s = bf16Bits(fixture.bytes, at: Int(scale) + metadata * 2)
                let b = bf16Bits(fixture.bytes, at: Int(bias) + metadata * 2)
                result += s * dot + inputSum * b
            }
            lanes[lane] = result
        }
        for offset in [16, 8, 4, 2, 1] {
            let prior = lanes
            for lane in 0..<(32 - offset) { lanes[lane] += prior[lane + offset] }
        }
        return lanes[0]
    }

    private func swiglu(_ gateValue: Float, _ upValue: Float) -> Float {
        let gate = bf16(min(bf16(gateValue), 7))
        let up = bf16(min(max(bf16(upValue), -7), 7))
        let exponent = bf16(Foundation.exp(abs(gate)))
        let reciprocal = bf16(1 / bf16(1 + exponent))
        let sigmoid = gate < 0 ? reciprocal : bf16(1 - reciprocal)
        return bf16(bf16(gate * sigmoid) * up)
    }

    private func routerLogit(_ weights: [UInt16], hidden: [Float], row: Int) -> Float {
        var lanes = [Float](repeating: 0, count: 32)
        for lane in 0..<32 {
            var result: Float = 0
            for block in 0..<(d / 128) {
                let column = block * 128 + lane * 4
                for n in 0..<4 {
                    result += Quantization.bf16ToFloat(weights[row * d + column + n]) * hidden[column + n]
                }
            }
            lanes[lane] = result
        }
        for offset in [16, 8, 4, 2, 1] {
            let prior = lanes
            for lane in 0..<(32 - offset) { lanes[lane] += prior[lane + offset] }
        }
        return lanes[0]
    }

    private func routerTop8(_ router: [UInt16], hidden: [Float]) -> (indices: [UInt32], weights: [Float]) {
        let logits = (0..<256).map { routerLogit(router, hidden: hidden, row: $0) }
        let maximum = logits.max()!
        let exponentials = logits.map { Foundation.exp($0 - maximum) }
        var simdSums = [Float](repeating: 0, count: 8)
        for simd in 0..<8 {
            var lanes = Array(exponentials[(simd * 32)..<((simd + 1) * 32)])
            for offset in [16, 8, 4, 2, 1] {
                let prior = lanes
                for lane in 0..<(32 - offset) { lanes[lane] += prior[lane + offset] }
            }
            simdSums[simd] = lanes[0]
        }
        let fullSum = simdSums.dropFirst().reduce(simdSums[0], +)
        let softmax = exponentials.map { $0 * (1 / (fullSum + 1.0e-20)) }
        let indices = softmax.indices.sorted {
            softmax[$0] == softmax[$1] ? $0 < $1 : softmax[$0] > softmax[$1]
        }.prefix(topK).map(UInt32.init)
        let selected = indices.map { softmax[Int($0)] }
        let normalizer = selected.reduce(Float(0), +) + 1.0e-20
        return (indices, selected.map { $0 / normalizer })
    }

    private struct PaddedBuffer {
        let buffer: MTLBuffer
        let original: [UInt8]
        let prefix: Int
        let suffix: Int
    }

    private func paddedBuffer(_ device: MTLDevice, prefix: Int, payload: [UInt8], suffix: Int) throws -> PaddedBuffer {
        let sentinel: UInt8 = 0xCD
        let original = [UInt8](repeating: sentinel, count: prefix) + payload
            + [UInt8](repeating: sentinel, count: suffix)
        let buffer = try #require(device.makeBuffer(bytes: original,
                                                    length: original.count,
                                                    options: .storageModeShared))
        return PaddedBuffer(buffer: buffer, original: original, prefix: prefix, suffix: suffix)
    }

    private func assertSentinels(_ padded: PaddedBuffer) {
        let actual = readBytes(padded.buffer, count: padded.original.count)
        #expect(Array(actual.prefix(padded.prefix)) == Array(padded.original.prefix(padded.prefix)))
        #expect(Array(actual.suffix(padded.suffix)) == Array(padded.original.suffix(padded.suffix)))
    }

    private func bytes<T>(_ values: [T]) -> [UInt8] {
        values.withUnsafeBytes { Array($0) }
    }

    private func readBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self), count: count))
    }

    private func readUInt16(_ buffer: MTLBuffer, offset: Int, count: Int) -> [UInt16] {
        let pointer = buffer.contents().advanced(by: offset).assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readUInt32(_ buffer: MTLBuffer, offset: Int, count: Int) -> [UInt32] {
        let pointer = buffer.contents().advanced(by: offset).assumingMemoryBound(to: UInt32.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readFloat(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float] {
        let pointer = buffer.contents().advanced(by: offset).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func bf16Bits(_ bytes: [UInt8], at offset: Int) -> Float {
        Quantization.bf16ToFloat(UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8)
    }

    private func bf16(_ value: Float) -> Float {
        Quantization.bf16ToFloat(Quantization.bf16Bits(value))
    }

    private func maxAbs(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
    }
}
