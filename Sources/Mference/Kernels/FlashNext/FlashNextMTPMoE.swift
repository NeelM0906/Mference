import Metal

/// Correctness-oriented FP32 native draft experts. Uses only the caller's
/// already-selected bounded/resident expert blobs; it never loads the trunk.
final class FlashNextMTPMoE {
    private let projection: FlashNextMTPFloat32Projection
    private let activation: MTLComputePipelineState
    private let sum: MTLComputePipelineState
    private let gate: MTLBuffer
    private let up: MTLBuffer
    private let acts: MTLBuffer
    private let expertOutputs: MTLBuffer
    private let shared: MTLBuffer
    private let sharedGate: MTLBuffer
    private let d: Int, f: Int, sharedF: Int, topK: Int

    init(context: MetalContext, hidden: Int, intermediate: Int, sharedIntermediate: Int, topK: Int) throws {
        d = hidden; f = intermediate; sharedF = sharedIntermediate; self.topK = topK
        projection = try FlashNextMTPFloat32Projection(context: context)
        activation = try context.pipeline("flashnext_mtp_silu_mul_f32")
        sum = try context.pipeline("flashnext_mtp_moe_sum_f32")
        func buffer(_ count: Int) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(length: count * 4, options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return b
        }
        gate = try buffer(max(f, sharedF)); up = try buffer(max(f, sharedF)); acts = try buffer(max(f, sharedF))
        expertOutputs = try buffer(d * topK); shared = try buffer(d); sharedGate = try buffer(1)
    }

    func encode(commandBuffer cb: MTLCommandBuffer, weights: FlashNextMTPWeights,
                blobs: [(buffer: MTLBuffer, offset: Int)], x: MTLBuffer,
                routerLogits: MTLBuffer, routeIDs: MTLBuffer, output: MTLBuffer) {
        precondition(blobs.count == topK)
        func feedForward(_ gateMatrix: FlashNextWeightMatrix, _ upMatrix: FlashNextWeightMatrix,
                         _ downMatrix: FlashNextWeightMatrix, width: Int, out: MTLBuffer, offset: Int) {
            projection.encode(commandBuffer: cb, matrix: gateMatrix, x: x, out: gate, rows: width, columns: d)
            projection.encode(commandBuffer: cb, matrix: upMatrix, x: x, out: up, rows: width, columns: d)
            let e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(activation)
            e.setBuffer(gate, offset: 0, index: 0)
            e.setBuffer(up, offset: 0, index: 1)
            e.setBuffer(acts, offset: 0, index: 2)
            e.dispatchThreads(.init(width: width, height: 1, depth: 1),
                threadsPerThreadgroup: .init(width: min(32, width), height: 1, depth: 1))
            e.endEncoding()
            projection.encode(commandBuffer: cb, matrix: downMatrix, x: acts,
                out: out, outOffset: offset, rows: d, columns: width)
        }
        let o = weights.expertOffsets
        for (index, blob) in blobs.enumerated() {
            func matrix(_ w: UInt32, _ s: UInt32, _ b: UInt32) -> FlashNextWeightMatrix {
                .int4(weights: blob.buffer, weightsOffset: blob.offset + Int(w),
                      scales: blob.buffer, scalesOffset: blob.offset + Int(s),
                      biases: blob.buffer, biasesOffset: blob.offset + Int(b))
            }
            feedForward(matrix(o.gateWOff, o.gateSOff, o.gateBOff),
                matrix(o.upWOff, o.upSOff, o.upBOff), matrix(o.downWOff, o.downSOff, o.downBOff),
                width: f, out: expertOutputs, offset: index * d * 4)
        }
        feedForward(weights.sharedGateProjection, weights.sharedUp, weights.sharedDown,
            width: sharedF, out: shared, offset: 0)
        projection.encode(commandBuffer: cb, matrix: weights.sharedGate, x: x,
            out: sharedGate, rows: 1, columns: d)
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(sum)
        for (index, buffer) in [expertOutputs, routerLogits, routeIDs, shared, sharedGate, output].enumerated() {
            e.setBuffer(buffer, offset: 0, index: index)
        }
        var width = UInt32(d), k = UInt32(topK)
        e.setBytes(&width, length: 4, index: 6)
        e.setBytes(&k, length: 4, index: 7)
        e.dispatchThreads(.init(width: d, height: 1, depth: 1),
            threadsPerThreadgroup: .init(width: min(32, d), height: 1, depth: 1))
        e.endEncoding()
    }
}
