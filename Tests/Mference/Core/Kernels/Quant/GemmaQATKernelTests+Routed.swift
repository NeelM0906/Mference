import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

extension GemmaQATKernelTests {
    struct RoutedFixture {
        let d = 2816, f = 704
        let input: [Float16]
        let residual: [Float16]
        let routing: [Float16]
        let blobs: [[UInt8]]
        let offsets: MoEExpertOffsets
        let expected: [Float]
        let expectedPartials: [[Float]]
        let expectedActivations: [[Float16]]

        init(sourceFP16: Bool = false) {
            let gate = Matrix(rows: f, columns: d, groupSize: 32)
            let down = Matrix(rows: d, columns: f, groupSize: 32)
            input = (0..<d).map { Float16(Float(($0 * 3) % 17 - 8) / 32) }
            residual = (0..<d).map { Float16(Float($0 % 9 - 4) / 32) }
            routing = (0..<8).map { Float16(Float($0 + 1) / 64) }
            let inputs = input
            let gateDots = (0..<f).map { r in
                (0..<gate.columns).reduce(0.0) { $0 + gate.value(row: r, column: $1) * Double(inputs[$1]) }
            }
            var outputs: [[Double]] = []
            var allActivations: [[Float16]] = []
            var resultBlobs: [[UInt8]] = []
            var resultOffsets: MoEExpertOffsets?
            for multiplier in [0.125, 0.25, 0.5, 1, 0.125, 0.25, 0.5, 1] {
                // Routed phase 1 retains FP32 gate/up results until the FP16
                // activation boundary (unlike the shared expert).
                let act = gateDots.map { dot in
                    let g = dot * multiplier, u = g * 0.5
                    if sourceFP16 { return Double(GemmaQATKernelTests.sourceGeGLU(Float16(g), Float16(u))) }
                    let gelu = 0.5 * g * (1 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
                    return Double(Float16(gelu * u))
                }
                allActivations.append(act.map(Float16.init))
                outputs.append((0..<d).map { r in
                    if sourceFP16 {
                        return GemmaQATKernelTests.sourceProjection(down, row: r, input: act.map(Float16.init)) * multiplier
                    }
                    return (0..<down.columns).reduce(0.0) { $0 + down.value(row: r, column: $1) * multiplier * act[$1] }
                })
                var bytes = [UInt8](repeating: 0xA5, count: 6)
                func append<T>(_ values: [T], alignment: Int) -> UInt32 {
                    while bytes.count % alignment != 0 { bytes.append(0xA5) }
                    let offset = UInt32(bytes.count)
                    values.withUnsafeBytes { bytes.append(contentsOf: $0) }
                    return offset
                }
                func project(_ matrix: Matrix, factor: Double, alignment: Int) -> (UInt32, UInt32, UInt32) {
                    func scaled(_ bits: [UInt16]) -> [UInt16] {
                        bits.map {
                            UInt16(truncatingIfNeeded:
                                (Float(bitPattern: UInt32($0) << 16) * Float(factor)).bitPattern >> 16)
                        }
                    }
                    return (append(matrix.packed, alignment: alignment),
                            append(scaled(matrix.scales), alignment: 2),
                            append(scaled(matrix.biases), alignment: 2))
                }
                let g = project(gate, factor: multiplier, alignment: 2)
                let u = project(gate, factor: multiplier * 0.5, alignment: 2)
                let v = project(down, factor: multiplier, alignment: 4)
                resultOffsets = MoEExpertOffsets(gateWOff: g.0, gateSOff: g.1, gateBOff: g.2,
                    upWOff: u.0, upSOff: u.1, upBOff: u.2, downWOff: v.0, downSOff: v.1, downBOff: v.2)
                resultBlobs.append(bytes)
            }
            blobs = resultBlobs
            offsets = resultOffsets!
            expectedPartials = outputs.map { $0.map { Float(Float16($0)) } }
            expectedActivations = allActivations
            let residuals = residual, routes = routing
            expected = (0..<d).map { col in
                if sourceFP16 {
                    var sum = Float16(0)
                    for rank in (0..<8).reversed() { sum += Float16(outputs[rank][col]) * routes[rank] }
                    return Float(sum + residuals[col])
                }
                return Float(Float16((0..<8).reduce(Double(residuals[col])) {
                    $0 + outputs[$1][col] * Double(routes[$1])
                }))
            }
        }
    }

    @Test(arguments: [false, true]) func nativeRoutedDecodeCoversStreamedSubsetAndSlotMap(sourceFP16: Bool) throws {
        let fixture = RoutedFixture(sourceFP16: sourceFP16)
        let context = try MetalContext()
        let d = fixture.d, f = fixture.f
        let x = try Self.buffer(fixture.input, context: context)
        let residual = try Self.buffer(fixture.residual, context: context)
        let routes = try Self.buffer(fixture.routing, context: context)
        let stride = (fixture.blobs[0].count + 4095) / 4096 * 4096
        let permutation = [5, 2, 7, 0, 3, 6, 1, 4]
        var slabBytes = [UInt8](repeating: 0xA5, count: stride * 8)
        for expert in 0..<8 {
            let start = permutation[expert] * stride
            slabBytes.replaceSubrange(start..<start + fixture.blobs[expert].count, with: fixture.blobs[expert])
        }
        let slab = try Self.buffer(slabBytes, context: context)
        let blobs = permutation.map { (buffer: slab, offset: $0 * stride) }
        var tableEntries = [Int16](repeating: -1, count: 128)
        for expert in 0..<8 { tableEntries[expert] = Int16(permutation[expert]) }
        let table = try Self.buffer(tableEntries, context: context)
        let indices = try Self.buffer((0..<8).map(UInt32.init), context: context)
        let slotOffsets = try Self.buffer([UInt32](repeating: .max, count: 8), context: context)
        let allHit = try Self.buffer([UInt32(0)], context: context)
        let subsets: [[UInt32]] = [[0, 2, 4, 6], [1, 3, 5, 7]]
        let subsetBuffers = try subsets.map { try Self.buffer($0, context: context) }
        for specialized in [false, true] {
            let kernel = try MoE(context: context,
                specializedD: UInt32(specialized ? d : 128), specializedF: UInt32(f), groupSize: 32, sourceFP16: sourceFP16)
            let arguments = try #require(kernel.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: 8))
            for mode in 0..<3 {
                let acts = try Self.buffer([Float16](repeating: .nan, count: 8 * f), context: context)
                let out = try Self.buffer([Float16](repeating: .nan, count: d), context: context)
                let hidden = try Self.buffer([Float16](repeating: 0, count: d), context: context)
                let cb = try #require(context.queue.makeCommandBuffer())
                if mode == 2 {
                    kernel.encodeSlotLookup(commandBuffer: cb, indices: indices, table: table,
                        slotStride: stride, slotOffsets: slotOffsets, allHit: allHit, numExperts: 128, topK: 8)
                    kernel.encodeSlotMapGuardedFFN(commandBuffer: cb, slab: slab,
                        slotOffsets: slotOffsets, allHit: allHit, routedOffsets: fixture.offsets,
                        x: x, acts: acts, routingWeights: routes, residual: residual, y: out, hidden: hidden,
                        d: UInt32(d), f: UInt32(f), topK: 8)
                } else {
                    if mode == 0 {
                        kernel.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                            routedArgBuffer: arguments, routedBlobs: blobs, routedOffsets: fixture.offsets,
                            x: x, acts: acts, d: UInt32(d), f: UInt32(f), topK: 8)
                    } else {
                        for (slots, active) in zip(subsets, subsetBuffers) {
                            kernel.encodeRoutedPersistentPhase1SubsetU16Load(commandBuffer: cb,
                                routedArgBuffer: arguments, routedBlobs: blobs, routedOffsets: fixture.offsets,
                                x: x, acts: acts, activeSlots: active, activeSlotIndices: slots,
                                activeCount: 4, d: UInt32(d), f: UInt32(f), topK: 8)
                        }
                    }
                    kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb,
                        routedArgBuffer: arguments, routedBlobs: blobs, routedOffsets: fixture.offsets,
                        acts: acts, routingWeights: routes, residual: residual, y: out,
                        d: UInt32(d), f: UInt32(f), topK: 8)
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed, "\(String(describing: cb.error))")
                if sourceFP16 {
                    let actualActs = Array(UnsafeBufferPointer(start: acts.contents().assumingMemoryBound(to: Float16.self), count: 8 * f))
                    #expect(actualActs == fixture.expectedActivations.flatMap { $0 },
                            "source routed activation mode=\(mode), specialized=\(specialized)")
                }
                let actual = out.contents().assumingMemoryBound(to: Float16.self)
                let values = (0..<d).map { Float(actual[$0]) }
                #expect(RelError.compute(actual: values, reference: fixture.expected) < 0.002,
                        "routed group-32 mode=\(mode), specialized=\(specialized)")
                if mode == 2 {
                    #expect(allHit.contents().load(as: UInt32.self) == 1)
                    let finalHidden = hidden.contents().assumingMemoryBound(to: Float16.self)
                    #expect(Array(UnsafeBufferPointer(start: finalHidden, count: d)) ==
                            Array(UnsafeBufferPointer(start: actual, count: d)))
                }
            }
        }
    }
}
