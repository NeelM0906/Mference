import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

private struct QATRouterCase: Decodable {
    struct Span: Decodable { let offset: Int; let count: Int }
    let position: Int
    let layer: Int
    let indices: [UInt32]
    let values: [String: Span]
}

private struct QATRouterWitness {
    let root: URL
    let data: Data
    let cases: [QATRouterCase]
    let context: MetalContext
    let model: Model

    init() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_ROUTER_REFERENCE"]))
        func frozen(_ name: String, _ hash: String) throws -> Data {
            let bytes = try Data(contentsOf: root.appendingPathComponent(name))
            try #require(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() == hash)
            return bytes
        }
        let data = try frozen("router-reference.f16", "6098644944cce7df8521560f41c125383088ed248177f024f2d50da47cfdae07")
        let cases = try JSONDecoder().decode([QATRouterCase].self, from:
            frozen("router-cases.json", "85066f7b8d9238e74ee265a608c7fda556650968e310e60e357813a72857efd7"))
        try #require(cases.count == 1140)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])),
                               device: context.device)
        try #require(model.modelID == CheckpointIdentity.gemma4QAT)
        self.root = root
        self.data = data
        self.cases = cases
        self.context = context
        self.model = model
    }

    func values(_ item: QATRouterCase, _ name: String) throws -> [Float16] {
        let span = try #require(item.values[name])
        return data.subdata(in: span.offset..<span.offset + span.count * 2)
            .withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }

    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        try #require(values.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
    }

    func effectiveScale(_ layer: Int) throws -> MTLBuffer {
        let view = try model.routerScale(layer: layer)
        let src = view.buffer.contents().advanced(by: Int(view.offset)).assumingMemoryBound(to: UInt16.self)
        let scale = (0..<2816).map { Float16(Quantization.bf16ToFloat(src[$0])) * Float16(1 / Float(2816).squareRoot()) }
        return try buffer(scale)
    }

    func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed, "\(String(describing: cb.error))")
    }
}

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ROUTER_REFERENCE"] != nil))
    func installedRouterMatchesFrozenSourceValues() throws {
        let witness = try QATRouterWitness()
        let ids = try witness.buffer([UInt32](repeating: .max, count: 8))
        let weights = try witness.buffer([Float16](repeating: .nan, count: 8))
        var differences: [String: Int] = [:]
        for specialized in [false, true] {
            let router = try MoE(context: witness.context, specializedD: specialized ? 2816 : 128,
                                 groupSize: 32, routerBF16: true, sourceFP16: true)
            for layer in 0..<30 {
                let scale = try witness.effectiveScale(layer)
                let projection = try witness.model.router(layer: layer)
                let gain = try witness.model.routerPerExpertScale(layer: layer)
                for item in witness.cases where item.layer == layer {
                    let input = try witness.buffer(witness.values(item, "input"))
                    let cb = try #require(witness.context.queue.makeCommandBuffer())
                    router.encodeRouterGemma4(commandBuffer: cb,
                        weights: projection.buffer, weightsOffset: Int(projection.offset),
                        scales: projection.buffer, biases: projection.buffer, hidden: input,
                        effectiveScale: scale, perExpertScale: gain.buffer, perExpertScaleOffset: Int(gain.offset),
                        outIndices: ids, outWeights: weights, numExperts: 128, d: 2816, topK: 8)
                    try witness.finish(cb)
                    let expectedLogits = try witness.values(item, "logits").map(Float.init)
                    let expectedWeights = try witness.values(item, "weights")
                    let actualIDs = Array(UnsafeBufferPointer(start: ids.contents().assumingMemoryBound(to: UInt32.self), count: 8))
                    let actualWeights = Array(UnsafeBufferPointer(start: weights.contents().assumingMemoryBound(to: Float16.self), count: 8))
                    differences["\(specialized)/logits", default: 0] += zip(router.routerLogitSnapshot, expectedLogits).filter { $0 != $1 }.count
                    differences["\(specialized)/ids", default: 0] += zip(actualIDs, item.indices).filter { $0 != $1 }.count
                    differences["\(specialized)/weights", default: 0] += zip(actualWeights, expectedWeights).filter { $0 != $1 }.count
                }
            }
        }
        try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
            .write(to: witness.root.appendingPathComponent("router-native-decode.json"))
        for (stage, count) in differences { #expect(count == 0, "\(stage): \(count) frozen source mismatches") }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ROUTER_REFERENCE"] != nil))
    func installedPrefillRouterMatchesFrozenSourceValues() throws {
        let witness = try QATRouterWitness()
        let router = try PrefillRouter(context: witness.context, routerBF16: true, sourceFP16: true)
        var differences = ["ids": 0, "weights": 0]
        for layer in 0..<30 {
            let cases = witness.cases.filter { $0.layer == layer }
            let scale = try witness.effectiveScale(layer)
            let projection = try witness.model.router(layer: layer)
            let gain = try witness.model.routerPerExpertScale(layer: layer)
            let input = try witness.buffer(cases.flatMap { try witness.values($0, "input") + [Float16](repeating: .nan, count: 8) })
            let ids = try witness.buffer([UInt32](repeating: .max, count: cases.count * 8))
            let weights = try witness.buffer([Float16](repeating: .nan, count: cases.count * 8))
            let cb = try #require(witness.context.queue.makeCommandBuffer())
            router.encodeGemma4Block(commandBuffer: cb,
                weights: projection.buffer, weightsOffset: Int(projection.offset),
                scales: projection.buffer, biases: projection.buffer, hidden: input,
                effectiveScale: scale, perExpertScale: gain.buffer, perExpertScaleOffset: Int(gain.offset),
                outIndices: ids, outWeights: weights, queryCount: UInt32(cases.count),
                numExperts: 128, d: 2816, topK: 8, hiddenStrideElements: 2824)
            try witness.finish(cb)
            let expectedIDs = cases.flatMap(\.indices)
            let expectedWeights = try cases.flatMap { try witness.values($0, "weights") }
            let actualIDs = Array(UnsafeBufferPointer(start: ids.contents().assumingMemoryBound(to: UInt32.self), count: expectedIDs.count))
            let actualWeights = Array(UnsafeBufferPointer(start: weights.contents().assumingMemoryBound(to: Float16.self), count: expectedWeights.count))
            differences["ids", default: 0] += zip(actualIDs, expectedIDs).filter { $0 != $1 }.count
            differences["weights", default: 0] += zip(actualWeights, expectedWeights).filter { $0 != $1 }.count
        }
        try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
            .write(to: witness.root.appendingPathComponent("router-native-prefill.json"))
        for (stage, count) in differences { #expect(count == 0, "\(stage): \(count) frozen source mismatches") }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ROUTER_PROBE"] == "1"))
    func probeInstalledRouterReductionOrder() throws {
        let witness = try QATRouterWitness()
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void probe_router(device const bfloat* matrix [[buffer(0)]],
            device const half* input [[buffer(1)]], device half* output [[buffer(2)]],
            constant uint& groups [[buffer(3)]], uint2 grid [[threadgroup_position_in_grid]],
            uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            threadgroup float partial[32];
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint base = group * 128u + lane * 4u; base < 2816u; base += groups * 128u) {
                for (uint j = 0; j < 4u; ++j) {
                    const float x = float(input[grid.y * 2816u + base + j]);
                    for (uint row = 0; row < 4u; ++row) {
                        sum[row] += float(half(matrix[(grid.x * 4u + row) * 2816u + base + j])) * x;
                    }
                }
            }
            for (uint row = 0; row < 4u; ++row) {
                for (ushort delta = 16; delta > 0; delta >>= 1) sum[row] += simd_shuffle_down(sum[row], delta);
                if (lane == 0) partial[group * 4u + row] = sum[row];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0 && lane == 0) {
                for (uint row = 0; row < 4u; ++row) {
                    float value = partial[row];
                    for (uint g = 1; g < groups; ++g) value += partial[g * 4u + row];
                    output[grid.y * 128u + grid.x * 4u + row] = half(value);
                }
            }
        }
        kernel void probe_router_emulated(device const bfloat* matrix [[buffer(0)]],
            device const half* input [[buffer(1)]], device half* output [[buffer(2)]],
            constant uint& groups [[buffer(3)]], uint2 grid [[threadgroup_position_in_grid]],
            uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            const uint expert = grid.x * 4u + group;
            float total = 0.0f;
            for (uint g = 0; g < groups; ++g) {
                float sum = 0.0f;
                for (uint base = g * 128u + lane * 4u; base < 2816u; base += groups * 128u) {
                    for (uint j = 0; j < 4u; ++j) {
                        sum += float(half(matrix[expert * 2816u + base + j]))
                            * float(input[grid.y * 2816u + base + j]);
                    }
                }
                for (ushort delta = 16; delta > 0; delta >>= 1) sum += simd_shuffle_down(sum, delta);
                total = g == 0 ? sum : total + sum;
            }
            if (lane == 0) output[grid.y * 128u + expert] = half(total);
        }
        """
        var differences: [String: Int] = [:]
        for safe in [false, true] {
            let options = MTLCompileOptions()
            options.languageVersion = MetalContext.shaderLanguageVersion
            options.mathMode = safe ? .safe : .fast
            let library = try witness.context.device.makeLibrary(source: source, options: options)
            for functionName in ["probe_router", "probe_router_emulated"] {
                let function = try #require(library.makeFunction(name: functionName))
                let pipeline = try witness.context.device.makeComputePipelineState(function: function)
                for groups in [UInt32(1), UInt32(8)] {
                    let name = "\(functionName)/\(safe ? "safe" : "fast")/groups\(groups)"
                    differences[name] = 0
                    for layer in 0..<30 {
                        let cases = witness.cases.filter { $0.layer == layer }
                        let projection = try witness.model.router(layer: layer)
                        let input = try witness.buffer(cases.flatMap { try witness.values($0, "scaled_input") })
                        let output = try witness.buffer([Float16](repeating: .nan, count: cases.count * 128))
                        let expected = try cases.flatMap { try witness.values($0, "logits") }
                        let cb = try #require(witness.context.queue.makeCommandBuffer())
                        let encoder = try #require(cb.makeComputeCommandEncoder())
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBuffer(projection.buffer, offset: Int(projection.offset), index: 0)
                        encoder.setBuffer(input, offset: 0, index: 1)
                        encoder.setBuffer(output, offset: 0, index: 2)
                        var count = groups
                        encoder.setBytes(&count, length: 4, index: 3)
                        encoder.dispatchThreadgroups(MTLSize(width: 32, height: cases.count, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: functionName == "probe_router" ? Int(groups) * 32 : 128, height: 1, depth: 1))
                        encoder.endEncoding()
                        try witness.finish(cb)
                        let actual = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float16.self), count: expected.count))
                        differences[name, default: 0] += zip(actual, expected).filter { $0 != $1 }.count
                    }
                }
            }
        }
        try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
            .write(to: witness.root.appendingPathComponent("router-reduction-probe.json"))
    }
}
