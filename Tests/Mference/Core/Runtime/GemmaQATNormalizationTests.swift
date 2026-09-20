import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

private struct QATNormalizationWitness {
    let root: URL
    let inputs: Data
    let expected: Data
    let cases: [[String: Any]]
    let context: MetalContext
    let model: Model

    init() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_NORM_REFERENCE"]))
        self.root = root
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name))
            try #require(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == digest)
            return data
        }
        inputs = try frozen("norm-inputs.f16", "035289863ac4947371c481715fcaa4e22495e8859fe16b4beffa4f43ac698d59")
        expected = try frozen("norm-reference.f16", "e2cd36b8dfd238a50c72ae61468ef6ac609393656ead4f5cee7ca40d57c6b534")
        let meta = try frozen("norm-cases.json", "4ca50ce9ec56e206b946a32812beeca6d66458724266f7752c9e6e69dc53ce95")
        cases = try #require(JSONSerialization.jsonObject(with: meta) as? [[String: Any]])
        context = try MetalContext()
        model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), device: context.device)
        try #require(model.modelID == CheckpointIdentity.gemma4QAT)
    }

    func bytes(_ item: [String: Any], _ key: String, _ data: Data) throws -> Data {
        let spec = try #require(item[key] as? [String: Int])
        let offset = try #require(spec["offset"]), count = try #require(spec["count"])
        return data.subdata(in: offset..<offset + count * 2)
    }

    func buffer(_ data: Data) throws -> MTLBuffer {
        try #require(data.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
    }

    func empty() throws -> MTLBuffer { try buffer(Data(repeating: 0, count: 2816 * 2)) }

    func differences(_ actual: MTLBuffer, _ expected: Data) -> Int {
        let a = actual.contents().assumingMemoryBound(to: UInt16.self)
        return expected.withUnsafeBytes { bytes in
            (0..<expected.count / 2).reduce(0) { $0 + (a[$1] == bytes.loadUnaligned(fromByteOffset: $1 * 2, as: UInt16.self) ? 0 : 1) }
        }
    }
}

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_NORM_PROBE"] != nil))
    func probeNormalizationFusionRopeMath() throws {
        let witness = try QATNormalizationWitness()
        let context = witness.context
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        constant uint HD [[function_constant(82)]];
        kernel void rotation(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
            constant uint& position [[buffer(2)]], constant float& theta [[buffer(3)]],
            constant uint& rotated [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
            const uint pair = gid.x, base = gid.y * HD;
            const float x0 = float(x[base + pair]), x1 = float(x[base + HD / 2u + pair]);
            if (pair < rotated) {
                const float exponent = -float(2u * pair) / float(HD);
                const float angle = float(position) * POW(theta, exponent);
                const float c = COS(angle), s = SIN(angle);
                y[base + pair] = half(LOWER);
                y[base + HD / 2u + pair] = half(UPPER);
            } else {
                y[base + pair] = half(x0);
                y[base + HD / 2u + pair] = half(x1);
            }
        }
        """
        let expressions = [("x0 * c - x1 * s", "x0 * s + x1 * c"),
            ("fma(x0, c, -(x1 * s))", "fma(x0, s, x1 * c)"),
            ("fma(-x1, s, x0 * c)", "fma(x0, s, x1 * c)"),
            ("fma(x0, c, -(x1 * s))", "fma(x1, c, x0 * s)"),
            ("fma(-x1, s, x0 * c)", "fma(x1, c, x0 * s)")]
        var results: [String: Int] = [:]
        for hd in [256, 512] {
            let input = try Data(contentsOf: witness.root.appendingPathComponent("rotary-\(hd)-normalized.f16"))
            let expected = try Data(contentsOf: witness.root.appendingPathComponent("rotary-\(hd)-legacy.f16"))
            let x = try witness.buffer(input)
            for profile in ["baseline", "default-safe", "fast-safe"] {
                for index in 0..<(profile == "baseline" ? 1 : expressions.count) {
                    let options = MTLCompileOptions()
                    options.languageVersion = MetalContext.shaderLanguageVersion
                    if profile != "baseline" { options.mathMode = .safe }
                    let (lower, upper) = expressions[index]
                    var code = source.replacingOccurrences(of: "LOWER", with: lower).replacingOccurrences(of: "UPPER", with: upper)
                    for name in ["POW", "COS", "SIN"] {
                        code = code.replacingOccurrences(of: name, with: (profile == "fast-safe" ? "fast::" : "") + name.lowercased())
                    }
                    let library = try context.device.makeLibrary(source: code, options: options)
                    let values = MTLFunctionConstantValues()
                    var dimension = UInt32(hd)
                    values.setConstantValue(&dimension, type: .uint, index: 82)
                    let function = try library.makeFunction(name: "rotation", constantValues: values)
                    let pipeline = try context.device.makeComputePipelineState(function: function)
                    let y = try witness.buffer(input)
                    let cb = try #require(context.queue.makeCommandBuffer())
                    let enc = try #require(cb.makeComputeCommandEncoder())
                    enc.setComputePipelineState(pipeline)
                    enc.setBuffer(x, offset: 0, index: 0)
                    enc.setBuffer(y, offset: 0, index: 1)
                    var position: UInt32 = hd == 256 ? 17 : 23
                    var theta: Float = hd == 256 ? 10000 : 1000000
                    var rotated: UInt32 = hd == 256 ? 128 : 64
                    enc.setBytes(&position, length: 4, index: 2)
                    enc.setBytes(&theta, length: 4, index: 3)
                    enc.setBytes(&rotated, length: 4, index: 4)
                    enc.dispatchThreads(MTLSize(width: hd / 2, height: 16, depth: 1), threadsPerThreadgroup: MTLSize(width: hd / 2, height: 1, depth: 1))
                    enc.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    try #require(cb.status == .completed)
                    let count = witness.differences(y, expected)
                    results["\(hd)/\(profile)/\(index)"] = count
                    if profile == "baseline" { #expect(count == 0) }
                }
            }
        }
        let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: witness.root.appendingPathComponent("norm-rotary-exact-probe.json"))
        print("[qat-normalization-rope] \(results)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_NORM_REFERENCE"] != nil))
    func installedPostAttentionNormalizationMatchesFrozenSourceValues() throws {
        let witness = try QATNormalizationWitness()
        let context = witness.context
        let fused = try FusedPostAttentionSetup(context: context, sourceFP16: true)
        let rms = try RMSNorm(context: context, sourceFP16: true)
        var differences: [String: Int] = [:]
        for item in witness.cases {
            let layer = try #require(item["layer"] as? Int)
            let prefix = "language_model.model.layers.\(layer)."
            let post = try witness.model.resident(name: prefix + "post_attention_layernorm.weight")
            let pre = try witness.model.resident(name: prefix + "pre_feedforward_layernorm.weight")
            let pre2 = try witness.model.resident(name: prefix + "pre_feedforward_layernorm_2.weight")
            let hidden = try witness.buffer(witness.bytes(item, "hidden_input", witness.inputs))
            let attention = try witness.buffer(witness.bytes(item, "attention_input", witness.inputs))
            let sourceHidden = try witness.buffer(witness.bytes(item, "hidden", witness.expected))
            let dense = try witness.empty(), routed = try witness.empty(), router = try witness.empty()
            let directDense = try witness.empty(), directRouted = try witness.empty(), directRouter = try witness.empty()
            let cb = try #require(context.queue.makeCommandBuffer())
            fused.encode(commandBuffer: cb, hidden: hidden, attn: attention,
                denseX: dense, routedX: routed, routerX: router,
                postAttentionWeight: post.buffer, postAttentionWeightOffset: Int(post.offset),
                preFFNWeight: pre.buffer, preFFNWeightOffset: Int(pre.offset),
                preFFN2Weight: pre2.buffer, preFFN2WeightOffset: Int(pre2.offset), d: 2816, eps: 1e-6)
            for (weight, output) in [(pre, directDense), (pre2, directRouted)] {
                rms.encodeBF16W(commandBuffer: cb, x: sourceHidden, weight: weight.buffer,
                    weightOffset: Int(weight.offset), out: output, d: 2816, eps: 1e-6)
            }
            rms.encodeNoScale(commandBuffer: cb, x: sourceHidden, out: directRouter, d: 2816, eps: 1e-6)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            for (path, key, actual) in [("fused", "hidden", hidden), ("fused", "dense", dense),
                ("fused", "routed", routed), ("fused", "router", router), ("primitive", "dense", directDense),
                ("primitive", "routed", directRouted), ("primitive", "router", directRouter)] {
                differences[path + "/" + key, default: 0] += try witness.differences(actual, witness.bytes(item, key, witness.expected))
            }
        }
        print("[qat-normalization] \(differences)")
        #expect(witness.cases.count == 90)
        for (stage, count) in differences { #expect(count == 0, "\(stage)") }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_NORM_PROBE"] != nil))
    func probeInstalledNormalizationReductionOrder() throws {
        let witness = try QATNormalizationWitness()
        let context = witness.context
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void norm_order(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
            constant uint& mode [[buffer(2)]], uint lid [[thread_position_in_threadgroup]],
            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            constexpr uint D = 2816u;
            threadgroup float partial[32];
            if (mode & 1u) {
                for (uint group = sg; group < 22u; group += 8u) {
                    float acc = 0.0f;
                    volatile float strict_acc = 0.0f;
                    for (uint j = 0; j < 4u; ++j) {
                        const float v = float(x[(group * 32u + lane) * 4u + j]);
                        if (mode & 4u) strict_acc = strict_acc + v * v;
                        else acc += v * v;
                    }
                    acc = simd_sum((mode & 4u) ? strict_acc : acc);
                    if (lane == 0u) partial[group] = acc;
                }
            } else {
                float acc = 0.0f;
                for (uint i = lid; i < D; i += 256u) { const float v = float(x[i]); acc = fma(v, v, acc); }
                acc = simd_sum(acc);
                if (lane == 0u) partial[sg] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                const float total = simd_sum(lane < ((mode & 1u) ? 22u : 8u) ? partial[lane] : 0.0f);
                if (lane == 0u) {
                    if (mode & 4u) {
                        volatile float mean = precise::divide(total, float(D));
                        volatile float regularized = mean + 1e-6f;
                        partial[0] = precise::rsqrt(regularized);
                    } else partial[0] = (mode & 2u)
                        ? precise::rsqrt(total / float(D) + 1e-6f) : rsqrt(total / float(D) + 1e-6f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lid; i < D; i += 256u) y[i] = half(float(x[i]) * partial[0]);
        }
        """
        var pipelines: [(String, MTLComputePipelineState)] = []
        for safe in [false, true] {
            let options = MTLCompileOptions()
            options.languageVersion = MetalContext.shaderLanguageVersion
            if safe { options.mathMode = .safe }
            let library = try context.device.makeLibrary(source: source, options: options)
            pipelines.append((safe ? "safe" : "fast", try context.device.makeComputePipelineState(function: try #require(library.makeFunction(name: "norm_order")))))
        }
        var results: [String: Int] = [:]
        for item in witness.cases {
            let x = try witness.buffer(witness.bytes(item, "hidden", witness.expected))
            let expected = try witness.bytes(item, "router", witness.expected)
            for (name, pipeline) in pipelines {
                for mode in UInt32(0)..<8 {
                    let y = try witness.empty()
                    let cb = try #require(context.queue.makeCommandBuffer())
                    let enc = try #require(cb.makeComputeCommandEncoder())
                    enc.setComputePipelineState(pipeline)
                    enc.setBuffer(x, offset: 0, index: 0)
                    enc.setBuffer(y, offset: 0, index: 1)
                    var value = mode
                    enc.setBytes(&value, length: 4, index: 2)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.endEncoding()
                    cb.commit()
                    cb.waitUntilCompleted()
                    try #require(cb.status == .completed)
                    results["\(name)/\(mode)", default: 0] += witness.differences(y, expected)
                }
            }
        }
        let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: witness.root.appendingPathComponent("norm-controlled-order-probe.json"))
        print("[qat-normalization-order] \(results)")
    }
}
