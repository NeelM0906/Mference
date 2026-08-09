import Foundation
import Metal
import Testing
@testable import Mference

@Suite("Maple native-BF16 decode attention")
struct MapleDecodeAttentionTests {
    private static let qCount = MapleDecodeAttention.numQHeads * MapleDecodeAttention.headDim
    private static let kvStride = MapleDecodeAttention.numKVHeads * MapleDecodeAttention.headDim
    private static let sentinel = UInt16(0x7BAD)

    @Test func maximumSequenceLengthIs128K() {
        #expect(MapleDecodeAttention.maximumSequenceLength == 128_000)
    }

    @Test("single-pass BF16 SDPA preserves exponent range and maps 16 Q heads onto 4 KV heads")
    func singlePassNativeBF16AndGQAMappingWithOffsets() throws {
        let qCount = Self.qCount
        let kvStride = Self.kvStride
        let tiny: [UInt16] = [
            Quantization.bf16Bits(1e-30), Quantization.bf16Bits(-1e-30),
            Quantization.bf16Bits(2e-30), Quantization.bf16Bits(-2e-30),
            Quantization.bf16Bits(1), Quantization.bf16Bits(-0.5),
        ]
        let q = [UInt16](repeating: 0, count: qCount)
        let k = [UInt16](repeating: 0, count: kvStride)
        let v = (0..<kvStride).map { tiny[$0 % tiny.count] }
        let context = try MetalContext()
        let attention = try MapleDecodeAttention(context: context)
        let actual = try Self.run(context: context, attention: attention, q: q, k: k, v: v,
                                  count: 1, sliding: true)
        var expected = [UInt16]()
        for head in 0..<MapleDecodeAttention.numQHeads {
            let source = (head / 4) * MapleDecodeAttention.headDim
            expected.append(contentsOf: v[source..<(source + MapleDecodeAttention.headDim)])
        }
        #expect(actual == expected)
        let rangeValue = Quantization.bf16ToFloat(actual[0])
        #expect(rangeValue != 0 && Float16(rangeValue) == 0)
    }

    @Test("mid-cycle 512-row ring follows physical rather than chronological cache order")
    func slidingPhysicalOrderWrapMatchesNonuniformGolden() throws {
        let count = MapleDecodeAttention.slidingCapacity
        let wrappedSlots = 100
        let q = Self.queryPayload(multiplier: 11)
        let k = Self.wrappedRingPayload(wrappedSlots: wrappedSlots, multiplier: 13)
        let v = Self.wrappedRingPayload(wrappedSlots: wrappedSlots, multiplier: 29)
        let context = try MetalContext()
        let attention = try MapleDecodeAttention(context: context)
        let actual = try Self.run(context: context, attention: attention, q: q, k: k, v: v,
                                  count: count, sliding: true)
        let physicalGolden = Self.reference(q: q, k: k, v: v, count: count)
        Self.expectClose(actual, physicalGolden)
        #expect(Self.cacheOrderGolden(k) != Self.cacheOrderGolden(
            Self.chronologicalRing(k, wrappedSlots: wrappedSlots)),
            "the deliberately rotated ring must distinguish physical slot traversal")
    }

    @Test("full NoPE attention grows from its complete linear prefix")
    func fullPrefixGrowthMatchesNonuniformGolden() throws {
        let q = Self.queryPayload(multiplier: 7)
        let shortK = Self.kvPayload(count: 3, multiplier: 17)
        let shortV = Self.kvPayload(count: 3, multiplier: 31)
        let longK = Self.kvPayload(count: 9, multiplier: 17)
        let longV = Self.kvPayload(count: 9, multiplier: 31)
        let context = try MetalContext()
        let attention = try MapleDecodeAttention(context: context)
        let short = try Self.run(context: context, attention: attention, q: q, k: shortK, v: shortV,
                                 count: 3, sliding: false)
        let long = try Self.run(context: context, attention: attention, q: q, k: longK, v: longV,
                                count: 9, sliding: false)
        Self.expectClose(short, Self.reference(q: q, k: shortK, v: shortV, count: 3))
        Self.expectClose(long, Self.reference(q: q, k: longK, v: longV, count: 9))
        #expect(short != long, "full attention must include appended prefix rows")
    }

    @Test("actual architecture threshold switches from single to two pass without changing GQA output")
    func fullAttentionCrossesActualTwoPassBoundary() throws {
        let qCount = Self.qCount
        let kvStride = Self.kvStride
        #expect(MapleDecodeAttention.dispatchPlan(architectureName: "applegpu_family_d")
                    == .init(twoPassThreshold: 1024, blocks: 128))
        #expect(MapleDecodeAttention.dispatchPlan(architectureName: "applegpu_family_s")
                    == .init(twoPassThreshold: 1024, blocks: 64))
        #expect(MapleDecodeAttention.dispatchPlan(architectureName: "applegpu_g16g")
                    == .init(twoPassThreshold: 4096, blocks: 64))

        let context = try MetalContext()
        let attention = try MapleDecodeAttention(context: context)
        let threshold = Int(MapleDecodeAttention.dispatchPlan(
            architectureName: context.device.architecture.name).twoPassThreshold)
        let q = [UInt16](repeating: 0, count: qCount)
        let expected = Self.constantHeadValues()
        for count in [threshold - 1, threshold] {
            let k = [UInt16](repeating: 0, count: count * kvStride)
            var v = [UInt16](repeating: 0, count: count * kvStride)
            for position in 0..<count {
                for kvHead in 0..<MapleDecodeAttention.numKVHeads {
                    let row = position * kvStride + kvHead * MapleDecodeAttention.headDim
                    v.replaceSubrange(row..<(row + MapleDecodeAttention.headDim), with: expected[
                        kvHead * MapleDecodeAttention.headDim..<(kvHead + 1) * MapleDecodeAttention.headDim])
                }
            }
            let actual = try Self.run(context: context, attention: attention, q: q, k: k, v: v,
                                      count: count, sliding: false)
            var gqaExpected = [UInt16]()
            for head in 0..<MapleDecodeAttention.numQHeads {
                let source = (head / 4) * MapleDecodeAttention.headDim
                gqaExpected.append(contentsOf: expected[source..<(source + MapleDecodeAttention.headDim)])
            }
            #expect(actual == gqaExpected, "N=\(count) should preserve constant values across dispatches")
        }
        let qNonuniform = Self.queryPayload(multiplier: 19)
        let kNonuniform = Self.kvPayload(count: threshold, multiplier: 23)
        let vNonuniform = Self.kvPayload(count: threshold, multiplier: 37)
        let nonuniform = try Self.run(context: context, attention: attention,
                                      q: qNonuniform, k: kNonuniform, v: vNonuniform,
                                      count: threshold, sliding: false)
        let twoPassMaximum = Self.maxAbs(nonuniform,
                                         Self.reference(q: qNonuniform, k: kNonuniform,
                                                        v: vNonuniform, count: threshold))
        #expect(twoPassMaximum <= 0.03125,
                "two-pass BF16 SDPA maxAbsDiff=\(twoPassMaximum)")
    }

    private static func run(context: MetalContext, attention: MapleDecodeAttention,
                            q: [UInt16], k: [UInt16], v: [UInt16], count: Int, sliding: Bool) throws -> [UInt16] {
        let inputPrefix = 3, outputPrefix = 5, suffix = 2
        let qStorage = [UInt16](repeating: sentinel, count: inputPrefix) + q
            + [UInt16](repeating: sentinel, count: suffix)
        let kStorage = [UInt16](repeating: sentinel, count: inputPrefix) + k
            + [UInt16](repeating: sentinel, count: suffix)
        let vStorage = [UInt16](repeating: sentinel, count: inputPrefix) + v
            + [UInt16](repeating: sentinel, count: suffix)
        let outStorage = [UInt16](repeating: sentinel, count: outputPrefix + qCount + suffix)
        guard let qBuffer = Self.buffer(context.device, qStorage),
              let kBuffer = Self.buffer(context.device, kStorage),
              let vBuffer = Self.buffer(context.device, vStorage),
              let outBuffer = Self.buffer(context.device, outStorage),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("Metal allocation failed")
            return []
        }
        if sliding {
            attention.encodeSliding(commandBuffer: commandBuffer,
                                    q: qBuffer, qOffset: inputPrefix * 2,
                                    k: kBuffer, kOffset: inputPrefix * 2,
                                    v: vBuffer, vOffset: inputPrefix * 2,
                                    out: outBuffer, outOffset: outputPrefix * 2,
                                    physicalCount: UInt32(count))
        } else {
            attention.encodeFull(commandBuffer: commandBuffer,
                                 q: qBuffer, qOffset: inputPrefix * 2,
                                 k: kBuffer, kOffset: inputPrefix * 2,
                                 v: vBuffer, vOffset: inputPrefix * 2,
                                 out: outBuffer, outOffset: outputPrefix * 2,
                                 sequenceLength: UInt32(count))
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        #expect(Self.read(qBuffer, count: qStorage.count) == qStorage)
        #expect(Self.read(kBuffer, count: kStorage.count) == kStorage)
        #expect(Self.read(vBuffer, count: vStorage.count) == vStorage)
        let output = Self.read(outBuffer, count: outStorage.count)
        #expect(output[..<outputPrefix].allSatisfy { $0 == sentinel })
        #expect(output.suffix(suffix).allSatisfy { $0 == sentinel })
        return Array(output[outputPrefix..<(outputPrefix + qCount)])
    }

    private static func reference(q: [UInt16], k: [UInt16], v: [UInt16], count: Int) -> [UInt16] {
        var output = [UInt16](repeating: 0, count: qCount)
        for qHead in 0..<MapleDecodeAttention.numQHeads {
            let kvHead = qHead / 4
            var scores = [Float](repeating: 0, count: count)
            for position in 0..<count {
                var score: Float = 0
                for dimension in 0..<MapleDecodeAttention.headDim {
                    score += Quantization.bf16ToFloat(q[qHead * MapleDecodeAttention.headDim + dimension])
                        * Quantization.bf16ToFloat(k[position * kvStride + kvHead * MapleDecodeAttention.headDim + dimension])
                }
                scores[position] = score * MapleDecodeAttention.attentionScale
            }
            let maximum = scores.max() ?? 0
            let weights = scores.map { exp($0 - maximum) }
            let normalizer = weights.reduce(0, +)
            for dimension in 0..<MapleDecodeAttention.headDim {
                var value: Float = 0
                for position in 0..<count {
                    value += weights[position] * Quantization.bf16ToFloat(
                        v[position * kvStride + kvHead * MapleDecodeAttention.headDim + dimension])
                }
                output[qHead * MapleDecodeAttention.headDim + dimension] = Quantization.bf16Bits(value / normalizer)
            }
        }
        return output
    }

    private static func queryPayload(multiplier: Int) -> [UInt16] {
        (0..<qCount).map { Quantization.bf16Bits(Float(($0 * multiplier) % 31 - 15) / 16) }
    }

    private static func kvPayload(count: Int, multiplier: Int) -> [UInt16] {
        (0..<(count * kvStride)).map { Quantization.bf16Bits(Float(($0 * multiplier) % 37 - 18) / 16) }
    }

    private static func wrappedRingPayload(wrappedSlots: Int, multiplier: Int) -> [UInt16] {
        var payload = [UInt16](repeating: 0, count: MapleDecodeAttention.slidingCapacity * kvStride)
        for slot in 0..<MapleDecodeAttention.slidingCapacity {
            let logicalPosition = slot < wrappedSlots ? MapleDecodeAttention.slidingCapacity + slot : slot
            for index in 0..<kvStride {
                payload[slot * kvStride + index] = Quantization.bf16Bits(
                    Float((logicalPosition * multiplier + index * 7) % 53 - 26) / 16)
            }
        }
        return payload
    }

    private static func chronologicalRing(_ physical: [UInt16], wrappedSlots: Int) -> [UInt16] {
        var chronological = [UInt16]()
        chronological.reserveCapacity(physical.count)
        chronological.append(contentsOf: physical[(wrappedSlots * kvStride)...])
        chronological.append(contentsOf: physical[..<(wrappedSlots * kvStride)])
        return chronological
    }

    private static func cacheOrderGolden(_ cache: [UInt16]) -> [UInt16] {
        (0..<MapleDecodeAttention.slidingCapacity).map { cache[$0 * kvStride] }
    }

    private static func constantHeadValues() -> [UInt16] {
        (0..<kvStride).map { Quantization.bf16Bits(Float(($0 % 11) - 5) / 8) }
    }

    private static func expectClose(_ actual: [UInt16], _ expected: [UInt16]) {
        let maximum = Self.maxAbs(actual, expected)
        #expect(maximum < 0.08, "BF16 SDPA maxAbsDiff=\(maximum)")
    }

    private static func maxAbs(_ actual: [UInt16], _ expected: [UInt16]) -> Float {
        zip(actual, expected).map {
            abs(Quantization.bf16ToFloat($0) - Quantization.bf16ToFloat($1))
        }.max() ?? 0
    }

    private static func buffer(_ device: MTLDevice, _ values: [UInt16]) -> MTLBuffer? {
        values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
    }

    private static func read(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: UInt16.self, capacity: count), count: count))
    }
}
