import Foundation
import CryptoKit
import Metal
import Testing
@testable import Mference

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_PROJECTION_REFERENCE"] != nil))
    func installedProjectionsMatchFrozenSourceValues() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_PROJECTION_REFERENCE"]))
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name))
            let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            try #require(actual == digest)
            return data
        }
        let raw = try frozen("native.f32", "7fc6529405622476b498d7d18f95d648800f2baf37ea1539e36c37ccd93f5a55")
        let metadata = try frozen("trace.json", "a7e55277847ef691bac84c9bfead2ba22a52a00e717aae183be0b719526d54d4")
        let caseData = try frozen("projection-cases.json", "b7755626bcaefc39981059c9076a1685697d22bba70faa4e320cfe21229e6495")
        let expected = try frozen("projection-reference.f16", "704ff6a7fe889d5dfed7a63e6b1aee61169107425b9d0a5ddb920e4c3e1eebb3")
        let meta = try #require(JSONSerialization.jsonObject(with: metadata) as? [String: Any])
        let entries = try #require(meta["entries"] as? [[String: Any]])
        let cases = try #require(JSONSerialization.jsonObject(with: caseData) as? [[String: Any]])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), device: context.device)
        let native = try DequantInt4GEMV(context: context,
            additionalShapes: [(2112, 2816), (2816, 2112)], groupSize: 32, sourceFP16: true)
        var mismatches = 0, compared = 0
        for item in cases {
            let position = try #require(item["position"] as? Int), layer = try #require(item["layer"] as? Int)
            let stage = try #require(item["stage"] as? String), role = try #require(item["role"] as? String)
            let outputOffset = try #require(item["offset"] as? Int), rows = try #require(item["count"] as? Int)
            let entry = try #require(entries.first { $0["position"] as? Int == position && $0["layer"] as? Int == layer && $0["stage"] as? String == stage })
            let offset = try #require(entry["offset"] as? Int), count = try #require(entry["count"] as? Int)
            let input: [Float16] = raw.withUnsafeBytes { bytes in
                (0..<count).map { Float16(bytes.loadUnaligned(fromByteOffset: offset + $0 * 4, as: Float.self)) }
            }
            let suffix = role == "o" ? "self_attn.o_proj.weight" : "mlp.\(role)_proj.weight"
            let view = try model.resident(name: "language_model.model.layers.\(layer).\(suffix)")
            let x = try #require(input.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
            let y = try #require(context.device.makeBuffer(length: rows * 2, options: .storageModeShared))
            let cb = try #require(context.queue.makeCommandBuffer())
            native.encode(commandBuffer: cb, weights: view.buffer, weightsOffset: Int(view.offset),
                scales: view.buffer, scalesOffset: Int(view.scaleOffset), biases: view.buffer,
                biasesOffset: Int(view.biasOffset), x: x, y: y, m: UInt32(rows), n: UInt32(count))
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            let actual = y.contents().assumingMemoryBound(to: UInt16.self)
            expected.withUnsafeBytes { bytes in
                for row in 0..<rows {
                    if actual[row] != bytes.loadUnaligned(fromByteOffset: outputOffset + row * 2, as: UInt16.self) { mismatches += 1 }
                }
            }
            compared += rows
        }
        #expect(compared == 887040)
        #expect(mismatches == 0)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_PROJECTION_PROBE"] != nil))
    func probeInstalledProjectionReductionOrder() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_PROJECTION_PROBE"]))
        let meta = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("trace.json"))) as? [String: Any])
        let entries = try #require(meta["entries"] as? [[String: Any]])
        let raw = try Data(contentsOf: root.appendingPathComponent("native.f32"))
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), device: context.device)
        let native = try DequantInt4GEMV(context: context,
            additionalShapes: [(2112, 2816), (2816, 2112)], groupSize: 32, sourceFP16: true)
        // Diagnostic candidate only: preserve the pinned source's quad dot,
        // affine sub-result and 8/16-value lane geometry. No runtime uses it.
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void source_order(
            device const ushort* weights [[buffer(0)]],
            device const bfloat* scales [[buffer(1)]],
            device const bfloat* biases [[buffer(2)]],
            device const half* x [[buffer(3)]], device half* y [[buffer(4)]],
            constant uint& M [[buffer(5)]], constant uint& N [[buffer(6)]],
            uint tg [[threadgroup_position_in_grid]],
            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            const uint row = tg * 8u + sg;
            if (row >= M) return;
            const uint values = N % 512u == 0u ? 16u : 8u;
            float result = 0.0f;
            for (uint base = lane * values; base < N; base += 32u * values) {
                float sum = 0.0f, dot = 0.0f;
                for (uint i = 0; i < values; i += 4u) {
                    const uint k = base + i;
                    const ushort packed = weights[row * (N / 4u) + k / 4u];
                    sum += x[k] + x[k + 1u] + x[k + 2u] + x[k + 3u];
                    dot += float(x[k]) * (packed & 15u)
                        + (float(x[k + 1u]) / 16.0f) * (packed & 240u)
                        + (float(x[k + 2u]) / 256.0f) * (packed & 3840u)
                        + (float(x[k + 3u]) / 4096.0f) * (packed & 61440u);
                }
                const uint g = row * (N / 32u) + base / 32u;
                result += float(half(scales[g])) * dot + sum * float(half(biases[g]));
            }
            result = simd_sum(result);
            if (lane == 0u) y[row] = half(result);
        }
        """
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        let controlledSource = source.replacingOccurrences(of: "void source_order(", with: "void controlled_order(")
            .replacingOccurrences(of: "const uint row =", with: "#pragma clang fp reassociate(off) contract(off)\n            const uint row =")
        let volatileSource = controlledSource.replacingOccurrences(of: "controlled_order", with: "volatile_order")
            .replacingOccurrences(of: "float result =", with: "volatile float result =")
            .replacingOccurrences(of: "float sum = 0.0f, dot = 0.0f;", with: "volatile float sum = 0.0f, dot = 0.0f;")
            .replacingOccurrences(of: "sum += x[k] + x[k + 1u] + x[k + 2u] + x[k + 3u];", with: """
                volatile half h0 = x[k] + x[k + 1u];
                volatile half h1 = h0 + x[k + 2u];
                volatile half h2 = h1 + x[k + 3u];
                sum += float(h2);
                """)
            .replacingOccurrences(of: "dot += float(x[k]) * (packed & 15u)\n                + (float(x[k + 1u]) / 16.0f) * (packed & 240u)\n                + (float(x[k + 2u]) / 256.0f) * (packed & 3840u)\n                + (float(x[k + 3u]) / 4096.0f) * (packed & 61440u);", with: """
                volatile float q = float(x[k]) * (packed & 15u);
                q += (float(x[k + 1u]) / 16.0f) * (packed & 240u);
                q += (float(x[k + 2u]) / 256.0f) * (packed & 3840u);
                q += (float(x[k + 3u]) / 4096.0f) * (packed & 61440u);
                dot += q;
                """)
            .replacingOccurrences(of: "result += float(half(scales[g])) * dot + sum * float(half(biases[g]));", with: """
                volatile float scaled = float(half(scales[g])) * dot;
                volatile float shifted = sum * float(half(biases[g]));
                volatile float affine = scaled + shifted;
                result += affine;
                """)
        let library = try context.device.makeLibrary(source: [source, controlledSource, volatileSource].joined(separator: "\n"), options: options)
        let pipeline = try context.device.makeComputePipelineState(function: try #require(library.makeFunction(name: "source_order")))
        let controlledPipeline = try context.device.makeComputePipelineState(function: try #require(library.makeFunction(name: "controlled_order")))
        let volatilePipeline = try context.device.makeComputePipelineState(function: try #require(library.makeFunction(name: "volatile_order")))
        options.mathMode = .safe
        let safeLibrary = try context.device.makeLibrary(source: source, options: options)
        let safePipeline = try context.device.makeComputePipelineState(function: try #require(safeLibrary.makeFunction(name: "source_order")))
        var nativeData = Data(), candidateData = Data(), safeData = Data(), controlledData = Data(), volatileData = Data()
        var cases: [[String: Any]] = []
        func values(position: Int, layer: Int, stage: String) throws -> [Float16] {
            let entry = try #require(entries.first { $0["position"] as? Int == position && $0["layer"] as? Int == layer && $0["stage"] as? String == stage })
            let offset = try #require(entry["offset"] as? Int), count = try #require(entry["count"] as? Int)
            return raw.withUnsafeBytes { bytes in
                (0..<count).map { Float16(bytes.loadUnaligned(fromByteOffset: offset + $0 * 4, as: Float.self)) }
            }
        }
        for position in [0, 2, 4] {
            for layer in 0..<30 {
                for (role, stage) in [("gate", "dense_input"), ("up", "dense_input"), ("down", "shared_activations"), ("o", "attention")] {
                    let path = role == "o" ? "self_attn.o_proj.weight" : "mlp.\(role)_proj.weight"
                    let view = try model.resident(name: "language_model.model.layers.\(layer).\(path)")
                    let input = try values(position: position, layer: layer, stage: stage)
                    let rows = role == "gate" || role == "up" ? 2112 : 2816
                    let x = try #require(input.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
                    let y = try #require(context.device.makeBuffer(length: rows * 2, options: .storageModeShared))
                    let candidate = try #require(context.device.makeBuffer(length: rows * 2, options: .storageModeShared))
                    let safe = try #require(context.device.makeBuffer(length: rows * 2, options: .storageModeShared))
                    let controlled = try #require(context.device.makeBuffer(length: rows * 2, options: .storageModeShared))
                    let volatile = try #require(context.device.makeBuffer(length: rows * 2, options: .storageModeShared))
                    let cb = try #require(context.queue.makeCommandBuffer())
                    native.encode(commandBuffer: cb, weights: view.buffer, weightsOffset: Int(view.offset),
                        scales: view.buffer, scalesOffset: Int(view.scaleOffset), biases: view.buffer,
                        biasesOffset: Int(view.biasOffset), x: x, y: y, m: UInt32(rows), n: UInt32(input.count))
                    for (pso, output) in [(pipeline, candidate), (safePipeline, safe), (controlledPipeline, controlled), (volatilePipeline, volatile)] {
                        let enc = try #require(cb.makeComputeCommandEncoder())
                        enc.setComputePipelineState(pso)
                        for (i, buffer, offset) in [(0, view.buffer, Int(view.offset)), (1, view.buffer, Int(view.scaleOffset)),
                                                   (2, view.buffer, Int(view.biasOffset)), (3, x, 0), (4, output, 0)] {
                            enc.setBuffer(buffer, offset: offset, index: i)
                        }
                        var m = UInt32(rows), n = UInt32(input.count)
                        enc.setBytes(&m, length: 4, index: 5)
                        enc.setBytes(&n, length: 4, index: 6)
                        enc.dispatchThreadgroups(MTLSize(width: (rows + 7) / 8, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        enc.endEncoding()
                    }
                    cb.commit()
                    cb.waitUntilCompleted()
                    try #require(cb.status == .completed)
                    if role == "o" {
                        let original = try values(position: position, layer: layer, stage: "attention_projection")
                        #expect(Array(UnsafeBufferPointer(start: y.contents().assumingMemoryBound(to: Float16.self), count: rows)) == original)
                    }
                    cases.append(["position": position, "layer": layer, "role": role, "stage": stage,
                                  "offset": nativeData.count, "count": rows])
                    nativeData.append(y.contents().assumingMemoryBound(to: UInt8.self), count: y.length)
                    candidateData.append(candidate.contents().assumingMemoryBound(to: UInt8.self), count: candidate.length)
                    safeData.append(safe.contents().assumingMemoryBound(to: UInt8.self), count: safe.length)
                    controlledData.append(controlled.contents().assumingMemoryBound(to: UInt8.self), count: controlled.length)
                    volatileData.append(volatile.contents().assumingMemoryBound(to: UInt8.self), count: volatile.length)
                }
            }
        }
        try nativeData.write(to: root.appendingPathComponent("projection-native.f16"))
        try candidateData.write(to: root.appendingPathComponent("projection-source-order.f16"))
        try safeData.write(to: root.appendingPathComponent("projection-safe-order.f16"))
        try controlledData.write(to: root.appendingPathComponent("projection-controlled-order.f16"))
        try volatileData.write(to: root.appendingPathComponent("projection-volatile-order.f16"))
        try JSONSerialization.data(withJSONObject: cases, options: [.sortedKeys]).write(to: root.appendingPathComponent("projection-cases.json"))
    }
}
