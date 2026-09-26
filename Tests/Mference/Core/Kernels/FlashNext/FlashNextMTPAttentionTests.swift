import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct FlashNextMTPAttentionTests {
    @Test func float32CacheAndSparseSelectionTrackScalarAttention() throws {
        let directory = try FlashNextToySynthetic.write(includeMTP: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        let cfg = model.config, d = cfg.hiddenSize
        let weights = try FlashNextMTPWeights(model: model)
        let rotary = Int(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor)
        let gpu = try FlashNextMTPAttention(context: context,
            geometry: .init(hidden: d, numHeads: cfg.numHeads, numKVHeads: cfg.numFullKVHeads,
                headDim: cfg.fullHeadDim, rotaryDim: rotary, theta: Float(cfg.fullRopeTheta),
                eps: 1e-6, scale: 1 / Float(cfg.fullHeadDim).squareRoot()), maxContext: 12)
        func matrix(_ name: String) throws -> [Float] {
            FlashNextWeights.read(try model.resident(name: "mtp.layers.0.self_attn." + name + ".weight"))
        }
        func norm(_ buffer: MTLBuffer, offset: Int) -> [Float] {
            Array(UnsafeBufferPointer(start: buffer.contents().advanced(by: offset).assumingMemoryBound(to: Float.self), count: cfg.fullHeadDim))
        }
        let referenceWeights = try FlashNextAttentionReference.Weights(
            q: matrix("q_proj"), k: matrix("k_proj"), v: matrix("v_proj"), o: matrix("o_proj"),
            qNorm: norm(weights.attention.qNorm, offset: weights.attention.qNormOffset),
            kNorm: norm(weights.attention.kNorm, offset: weights.attention.kNormOffset))
        var keys: [Float] = [], values: [Float] = []
        for position in 0..<12 {
            let x = (0..<d).map { Float16(Float(($0 * 7 + position * 3) % 29 - 14) / 16) }
            let input = try #require(context.device.makeBuffer(bytes: x, length: d * 2, options: .storageModeShared))
            let output = try #require(context.device.makeBuffer(length: d * 2, options: .storageModeShared))
            let contiguous = position.isMultiple(of: 2)
            let selected = contiguous ? Array(0...position) : [0, position]
            let indices = selected.map(UInt32.init)
            let selection = try #require(context.device.makeBuffer(bytes: indices, length: indices.count * 4, options: .storageModeShared))
            let cb = try #require(context.queue.makeCommandBuffer())
            gpu.encode(commandBuffer: cb, weights: weights.attention, x: input, output: output,
                position: position, selected: selection, selectedCount: selected.count, contiguous: contiguous)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.error == nil)
            let expected = FlashNextAttentionReference.run(x: x.map(Float.init), hidden: d,
                rows: 1, startPosition: position, w: referenceWeights, selected: [selected],
                keys: keys, values: values, scale: 1 / Float(cfg.fullHeadDim).squareRoot(),
                g: .init(numHeads: cfg.numHeads, numKVHeads: cfg.numFullKVHeads,
                    headDim: cfg.fullHeadDim, rotaryDim: rotary, theta: Float(cfg.fullRopeTheta), eps: 1e-6))
            keys = expected.keys
            values = expected.values
            let actual = UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float16.self), count: d).map(Float.init)
            let error = zip(actual, expected.out).map { abs($0 - $1) }.max()!
            let scale = expected.out.map(abs).max()!
            #expect(actual.allSatisfy(\.isFinite) && scale > 0 && error <= scale * 0.002)
        }
    }
}
