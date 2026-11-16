import Metal
import Testing
@testable import Mference

@Suite struct FlashNextMTPInputFusionTests {
    @Test(arguments: [64, 2560], [4, 8, 16])
    func fullBundleNormAndSharedProjectionMatchScalarOracle(hidden: Int, bits: Int) throws {
        let context = try MetalContext()
        let streams = 4
        let fusion = try FlashNextMTPInputFusion(context: context, hidden: hidden, streams: streams)
        let embedding = (0..<hidden).map { Float16(Float($0 % 13 - 6) / 8) }
        // Very different stream amplitudes expose an accidental grouped norm.
        var hc: [Float16] = []
        for index in 0..<(hidden * streams) {
            let amplitude: Float = Float(index / hidden + 1)
            let value: Float = Float(index % 11 - 5) / 4
            hc.append(Float16(amplitude * value))
        }
        let normE = (0..<hidden).map { Quantization.bf16Bits(0.5 + Float($0 % 7) / 16) }
        let normH = (0..<(hidden * streams)).map { Quantization.bf16Bits(0.75 + Float($0 % 5) / 8) }
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride,
                                                    options: .storageModeShared))
        }
        func projection(gain: Float, offset: Int) throws -> FlashNextWeightMatrix {
            if bits != 16 {
                let perByte = 8 / bits
                var bytes = [UInt8](repeating: 0, count: hidden * hidden / perByte)
                for row in 0..<hidden {
                    let column = (row * 17 + offset) % hidden
                    let flat = row * hidden + column
                    bytes[flat / perByte] |= UInt8(1 << ((flat % perByte) * bits))
                }
                let groups = hidden * hidden / 64
                let w = try buffer([UInt8](repeating: 0, count: 16) + bytes)
                let s = try buffer([UInt16(0)] + [UInt16](repeating: Quantization.bf16Bits(gain), count: groups))
                let b = try buffer([UInt16(0)] + [UInt16](repeating: 0, count: groups))
                if bits == 4 {
                    return .int4(weights: w, weightsOffset: 16, scales: s, scalesOffset: 2,
                                 biases: b, biasesOffset: 2)
                }
                return .int8(weights: w, weightsOffset: 16, scales: s, scalesOffset: 2,
                             biases: b, biasesOffset: 2)
            }
            var values = [UInt16](repeating: 0, count: hidden * hidden)
            for row in 0..<hidden {
                values[row * hidden + (row * 17 + offset) % hidden] = Quantization.bf16Bits(gain)
            }
            return .bf16(buffer: try buffer([UInt16(0)] + values), offset: 2)
        }
        let inputE = try buffer(embedding)
        let inputH = try buffer(hc)
        // Nonzero offsets ensure the norm slices, not the buffer base, are used.
        let weightsE = try buffer([UInt16(0)] + normE)
        let weightsH = try buffer([UInt16(0)] + normH)
        let output = try buffer([Float16](repeating: .nan, count: hc.count))
        let fcE = try projection(gain: 0.5, offset: 7)
        let fcH = try projection(gain: 0.25, offset: 3)
        let cb = try #require(context.queue.makeCommandBuffer())
        fusion.encode(commandBuffer: cb, embedding: inputE, targetHidden: inputH,
            embeddingNorm: weightsE, embeddingNormOffset: 2,
            hiddenNorm: weightsH, hiddenNormOffset: 2,
            embeddingProjection: fcE, hiddenProjection: fcH, output: output)
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.error == nil)

        // Independent scalar transcription: one RMS reduction over H*D,
        // then a shared projection per stream and a broadcast embedding add.
        // FP32 intermediates; only the final fused bundle is stored as FP16.
        func norm(_ values: [Float16], _ weights: [UInt16]) -> [Float] {
            var sum: Double = 0
            for value in values { sum += Double(value) * Double(value) }
            let inverse = Float(1 / (sum / Double(values.count) + 1e-6).squareRoot())
            return zip(values, weights).map { Float($0) * inverse * Quantization.bf16ToFloat($1) }
        }
        let normalizedE = norm(embedding, normE)
        let normalizedH = norm(hc, normH)
        let actual = output.contents().assumingMemoryBound(to: Float16.self)
        var maxError: Float = 0
        for stream in 0..<streams {
            for row in 0..<hidden {
                let eIndex: Int = (row * 17 + 7) % hidden
                let hColumn: Int = (row * 17 + 3) % hidden
                let hIndex: Int = stream * hidden + hColumn
                let e = normalizedE[eIndex] * Float(0.5)
                let h = normalizedH[hIndex] * Float(0.25)
                let expected = Float16(Float(e) + Float(h))
                let value = actual[stream * hidden + row]
                #expect(value.isFinite)
                maxError = max(maxError, abs(Float(value) - Float(expected)))
            }
        }
        // At this fixture's unit-scale outputs, 2e-3 covers FP16 store/reduction
        // rounding. No installed-model or end-to-end MTP parity is claimed.
        #expect(maxError <= 0.002)
        #expect(Array(UnsafeBufferPointer(start: inputE.contents().assumingMemoryBound(to: Float16.self),
                                          count: embedding.count)) == embedding)
        #expect(Array(UnsafeBufferPointer(start: inputH.contents().assumingMemoryBound(to: Float16.self),
                                          count: hc.count)) == hc)
        print("[Flash-Next MTP fusion] hidden=\(hidden) bits=\(bits) maxAbs=\(maxError)")
    }
}
