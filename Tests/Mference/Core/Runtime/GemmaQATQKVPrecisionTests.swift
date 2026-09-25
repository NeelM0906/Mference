import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

private struct QATQKVCase: Decodable {
    struct Span: Decodable { let offset: Int, count: Int }
    let position: Int, inputPosition: Int, layer: Int, headDim: Int, kvHeads: Int, rotatedPairs: Int
    let theta: Float
    let values: [String: Span]
}

private struct QATQKVWitness {
    let root: URL
    let data: Data
    let cases: [QATQKVCase]
    let context: MetalContext
    let model: Model

    init() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_QKV_REFERENCE"]))
        self.root = root
        let data = try Data(contentsOf: root.appendingPathComponent("qkv-reference.f16"))
        let meta = try Data(contentsOf: root.appendingPathComponent("qkv-cases.json"))
        let dataHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let metaHash = SHA256.hash(data: meta).map { String(format: "%02x", $0) }.joined()
        try #require(dataHash == "a19d03a0c28b7a40694ed814fb3db8519c7cf5d2261c2c033b7f09bc80e73ad1")
        try #require(metaHash == "91526150d700a13bbd74d403e1f7456ec2b77b8c9fe6cbb759de4f4074b1b925")
        self.data = data
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        cases = try decoder.decode([QATQKVCase].self, from: meta)
        try #require(cases.count == 160)
        context = try MetalContext()
        model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), device: context.device)
        try #require(model.modelID == CheckpointIdentity.gemma4QAT)
    }

    func bytes(_ item: QATQKVCase, _ name: String) throws -> Data {
        let span = try #require(item.values[name])
        return data.subdata(in: span.offset..<span.offset + span.count * 2)
    }

    func buffer(_ data: Data) throws -> MTLBuffer {
        try #require(data.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
    }

    func differences(_ actual: MTLBuffer, _ expected: Data) -> Int {
        let values = actual.contents().assumingMemoryBound(to: UInt16.self)
        return expected.withUnsafeBytes { bytes in
            (0..<expected.count / 2).reduce(0) {
                $0 + (values[$1] == bytes.loadUnaligned(fromByteOffset: $1 * 2, as: UInt16.self) ? 0 : 1)
            }
        }
    }
}

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_QKV_PROBE"] != nil))
    func probeInstalledRotaryArithmetic() throws {
        let witness = try QATQKVWitness()
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void rotate(device const half* input [[buffer(0)]], device half* output [[buffer(1)]],
            constant uint& position [[buffer(2)]], constant uint& dimension [[buffer(3)]],
            constant float& theta [[buffer(4)]], constant float& logBase [[buffer(5)]],
            constant uint& rotated [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
            const uint lower = gid.y * dimension + gid.x, upper = lower + dimension / 2u;
            const float x0 = float(input[lower]), x1 = float(input[upper]);
            if (gid.x >= rotated) { output[lower] = half(x0); output[upper] = half(x1); return; }
            const float d = float(gid.x) / float(dimension / 2u);
            const float frequency = FREQUENCY;
            const float angle = float(position) * frequency;
            const float c = fast::cos(angle), s = fast::sin(angle);
            ROTATION
        }
        """
        let frequencies = [
            "pow(theta, -d)",
            "dimension == 256u ? exp2(-d * logBase) : (1.0f / pow(theta, d))",
            "dimension == 256u ? precise::exp2(-d * logBase) : precise::divide(1.0f, precise::pow(theta, d))",
        ]
        let rotations = [
            "output[lower] = half(x0 * c - x1 * s); output[upper] = half(x0 * s + x1 * c);",
            "volatile float a = x0 * c, b = x1 * s, e = x0 * s, f = x1 * c; output[lower] = half(a - b); output[upper] = half(e + f);",
        ]
        var differences: [String: Int] = [:]
        for safe in [false, true] {
            for frequency in frequencies.indices {
                for rotation in rotations.indices {
                    let options = MTLCompileOptions()
                    options.languageVersion = MetalContext.shaderLanguageVersion
                    if safe { options.mathMode = .safe }
                    let code = source.replacingOccurrences(of: "FREQUENCY", with: frequencies[frequency])
                        .replacingOccurrences(of: "ROTATION", with: rotations[rotation])
                    let library = try witness.context.device.makeLibrary(source: code, options: options)
                    let function = try #require(library.makeFunction(name: "rotate"))
                    let pipeline = try witness.context.device.makeComputePipelineState(function: function)
                    for item in witness.cases {
                        for (normalized, expected, heads) in [("q_norm", "query", 16), ("k_norm", "key", item.kvHeads)] {
                            let input = try witness.buffer(witness.bytes(item, normalized))
                            let output = try witness.buffer(Data(repeating: 0, count: input.length))
                            let cb = try #require(witness.context.queue.makeCommandBuffer())
                            let enc = try #require(cb.makeComputeCommandEncoder())
                            enc.setComputePipelineState(pipeline)
                            enc.setBuffer(input, offset: 0, index: 0)
                            enc.setBuffer(output, offset: 0, index: 1)
                            var position = UInt32(item.position), dimension = UInt32(item.headDim)
                            var theta = item.theta, logBase = Float(log2(Double(item.theta)))
                            var rotated = UInt32(item.rotatedPairs)
                            enc.setBytes(&position, length: 4, index: 2)
                            enc.setBytes(&dimension, length: 4, index: 3)
                            enc.setBytes(&theta, length: 4, index: 4)
                            enc.setBytes(&logBase, length: 4, index: 5)
                            enc.setBytes(&rotated, length: 4, index: 6)
                            enc.dispatchThreads(MTLSize(width: item.headDim / 2, height: heads, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: item.headDim / 2, height: 1, depth: 1))
                            enc.endEncoding()
                            cb.commit()
                            cb.waitUntilCompleted()
                            try #require(cb.status == .completed)
                            let key = "\(safe ? "safe" : "fast")/freq\(frequency)/rotation\(rotation)/\(item.headDim)"
                            differences[key, default: 0] += try witness.differences(output, witness.bytes(item, expected))
                        }
                    }
                }
            }
        }
        try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
            .write(to: witness.root.appendingPathComponent("qkv-rotary-probe.json"))
        print("[qat-rotary-probe] \(differences)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_QKV_REFERENCE"] != nil))
    func installedQKVStagesMatchFrozenSourceValues() throws {
        let witness = try QATQKVWitness()
        let projection = try FusedQKVGEMV(context: witness.context, groupSize: 32, sourceFP16: true)
        let epilogue = try FusedQKVEpilogue(context: witness.context, sourceFP16: true)
        var differences: [String: Int] = [:]
        for item in witness.cases {
            let qWeight = try witness.model.qProj(layer: item.layer)
            let kWeight = try witness.model.kProj(layer: item.layer)
            let vWeight = item.headDim == 512 ? kWeight : try witness.model.vProj(layer: item.layer)
            let input = try witness.buffer(witness.bytes(item, "input"))
            let q = try witness.buffer(Data(repeating: 0, count: item.headDim * 16 * 2))
            let k = try witness.buffer(Data(repeating: 0, count: item.headDim * item.kvHeads * 2))
            let v = try witness.buffer(Data(repeating: 0, count: item.headDim * item.kvHeads * 2))
            let cb = try #require(witness.context.queue.makeCommandBuffer())
            projection.encode(commandBuffer: cb,
                qWeights: qWeight.buffer, qWeightsOffset: Int(qWeight.offset),
                qScales: qWeight.buffer, qScalesOffset: Int(qWeight.scaleOffset),
                qBiases: qWeight.buffer, qBiasesOffset: Int(qWeight.biasOffset),
                kWeights: kWeight.buffer, kWeightsOffset: Int(kWeight.offset),
                kScales: kWeight.buffer, kScalesOffset: Int(kWeight.scaleOffset),
                kBiases: kWeight.buffer, kBiasesOffset: Int(kWeight.biasOffset),
                vWeights: vWeight.buffer, vWeightsOffset: Int(vWeight.offset),
                vScales: vWeight.buffer, vScalesOffset: Int(vWeight.scaleOffset),
                vBiases: vWeight.buffer, vBiasesOffset: Int(vWeight.biasOffset),
                x: input, qOut: q, kOut: k, vOut: v,
                qRows: UInt32(item.headDim * 16), kvRows: UInt32(item.headDim * item.kvHeads), n: 2816)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            for (name, buffer) in [("q", q), ("k", k), ("v", v)] {
                differences["projection/\(name)", default: 0] += try witness.differences(buffer, witness.bytes(item, name + "_raw"))
            }
            let qNorm = try witness.model.qNorm(layer: item.layer)
            let kNorm = try witness.model.kNorm(layer: item.layer)
            for normalizeOnly in [true, false] {
                let q = try witness.buffer(witness.bytes(item, "q_raw"))
                let k = try witness.buffer(witness.bytes(item, "k_raw"))
                let v = try witness.buffer(witness.bytes(item, "v_raw"))
                let cb = try #require(witness.context.queue.makeCommandBuffer())
                epilogue.encode(commandBuffer: cb, q: q, k: k, v: v,
                    qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                    kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                    headDim: UInt32(item.headDim), numQHeads: 16, numKVHeads: UInt32(item.kvHeads),
                    position: normalizeOnly ? 0 : UInt32(item.position), theta: item.theta,
                    rotatedPairs: UInt32(item.rotatedPairs), eps: 1e-6)
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                for (role, name, buffer) in [("q", normalizeOnly ? "q_norm" : "query", q),
                    ("k", normalizeOnly ? "k_norm" : "key", k), ("v", "v_norm", v)] {
                    let key = "\(normalizeOnly ? "normalization" : "rotated")/\(item.headDim)/\(role)"
                    differences[key, default: 0] += try witness.differences(buffer, witness.bytes(item, name))
                }
            }
        }
        try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
            .write(to: witness.root.appendingPathComponent("qkv-native-stages.json"))
        print("[qat-qkv-stages] \(differences)")
        for (stage, count) in differences { #expect(count == 0, "\(stage): \(count) frozen source values differ") }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_QKV_REFERENCE"] != nil))
    func installedPrefillQKVStagesMatchFrozenSourceValues() throws {
        let witness = try QATQKVWitness()
        let epilogue = try PrefillQKVEpilogue(context: witness.context, sourceFP16: true)
        var batches = (0..<30).map { layer in witness.cases.filter { $0.layer == layer && $0.position < 5 } }
        batches += witness.cases.filter { $0.position >= 5 }.map { [$0] }
        var differences: [String: Int] = [:]
        for batch in batches {
            let item = try #require(batch.first)
            func joined(_ key: String) throws -> Data {
                var data = Data()
                for row in batch { data.append(try witness.bytes(row, key)) }
                return data
            }
            let q = try witness.buffer(joined("q_raw"))
            let k = try witness.buffer(joined("k_raw"))
            let v = try witness.buffer(joined("v_raw"))
            let qNorm = try witness.model.qNorm(layer: item.layer)
            let kNorm = try witness.model.kNorm(layer: item.layer)
            let cb = try #require(witness.context.queue.makeCommandBuffer())
            epilogue.encode(commandBuffer: cb, q: q, k: k, v: v,
                qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                startPosition: UInt32(item.position), queryCount: UInt32(batch.count),
                headDim: UInt32(item.headDim), numQHeads: 16, numKVHeads: UInt32(item.kvHeads),
                qTokenStrideElements: UInt32(item.headDim * 16), kvTokenStrideElements: UInt32(item.headDim * item.kvHeads),
                theta: item.theta, rotatedPairs: UInt32(item.rotatedPairs), eps: 1e-6)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            for (name, buffer) in [("query", q), ("key", k), ("v_norm", v)] {
                differences["\(item.headDim)/\(name)", default: 0] += try witness.differences(buffer, joined(name))
            }
        }
        try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
            .write(to: witness.root.appendingPathComponent("qkv-native-prefill.json"))
        print("[qat-qkv-prefill] \(differences)")
        for (stage, count) in differences { #expect(count == 0, "\(stage): \(count) frozen source values differ") }
    }
}
