import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

private struct QATAttentionEntry: Decodable {
    let position: Int
    let layer: Int
    let stage: String
    let offset: Int
    let count: Int
}

private struct QATAttentionWitness {
    let root: URL
    let native: Data
    let expected: Data
    let entries: [QATAttentionEntry]
    let cases: [QATAttentionEntry]
    let context: MetalContext

    init() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_ATTENTION_REFERENCE"]))
        self.root = root
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name), options: .mappedIfSafe)
            try #require(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == digest)
            return data
        }
        struct Trace: Decodable { let entries: [QATAttentionEntry] }
        let trace = try frozen("trace.json", "f7f848996c3672a5f2396df62c2a0f735096acafe16a573d4a57b091c9d3b375")
        entries = try JSONDecoder().decode(Trace.self, from: trace).entries
        native = try frozen("native.f32", "4285c0af675a3b28874bc3196824675f25e38258a990e4ff7c72ebf0afac1469")
        expected = try frozen("attention-reference.f16", "aeb5f0d449e3cfd13f55cbdc12bebfbc4874094b3062e0eb5271c6dda7fa89ec")
        let meta = try frozen("attention-cases.json", "1e31903aa64cdb2ec92039a4928d7cb2306ddb15d2acac440fcc142ecb5a9a2a")
        cases = try JSONDecoder().decode([QATAttentionEntry].self, from: meta)
            .filter { $0.stage == "attention_native_qkv" }
        try #require(cases.count == 150)
        let install = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"]))
        let manifest = try ManifestReader.load(directoryURL: install, expecting: .gemma4_26B_A4B)
        try #require(manifest.modelID == CheckpointIdentity.gemma4QAT)
        context = try MetalContext()
    }

    func input(_ item: QATAttentionEntry, _ stage: String, position: Int? = nil) throws -> Data {
        let entry = try #require(entries.first {
            $0.layer == item.layer && $0.position == (position ?? item.position) && $0.stage == stage
        })
        var halves = [Float16]()
        halves.reserveCapacity(entry.count)
        try native.withUnsafeBytes { bytes in
            for i in 0..<entry.count {
                let value = bytes.loadUnaligned(fromByteOffset: entry.offset + 4 * i, as: Float.self)
                let half = Float16(value)
                try #require(Float(half) == value)
                halves.append(half)
            }
        }
        return halves.withUnsafeBytes { Data($0) }
    }

    func history(_ item: QATAttentionEntry, _ stage: String) throws -> Data {
        var data = Data()
        for position in 0...item.position {
            data.append(try input(item, stage, position: position))
        }
        return data
    }

    func buffer(_ data: Data) throws -> MTLBuffer {
        try #require(data.withUnsafeBytes {
            context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
    }

    func differences(_ actual: MTLBuffer, _ item: QATAttentionEntry) -> Int {
        let values = actual.contents().assumingMemoryBound(to: UInt16.self)
        return expected.withUnsafeBytes { bytes in
            (0..<item.count).reduce(0) {
                $0 + (values[$1] == bytes.loadUnaligned(fromByteOffset: item.offset + $1 * 2, as: UInt16.self) ? 0 : 1)
            }
        }
    }
}

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ATTENTION_HISTORY"] != nil))
    func attentionHistoryBoundariesMatchFrozenSourceValues() throws {
        let witness = try QATAttentionWitness()
        let expected = try Data(contentsOf: witness.root.appendingPathComponent("attention-history.f16"))
        let meta = try Data(contentsOf: witness.root.appendingPathComponent("attention-history.json"))
        try #require(SHA256.hash(data: expected).map { String(format: "%02x", $0) }.joined() == "e543ff94d5052f3e780dfe25afa3278c32e561634c230c280c39769b2da5cf78")
        try #require(SHA256.hash(data: meta).map { String(format: "%02x", $0) }.joined() == "488d84453f238106896e674f608b4afd206320c618cc0584216f576b42e42a7a")
        struct History: Decodable {
            let layer: Int, headDim: Int, length: Int, window: Int, offset: Int, count: Int
            let inputSha256: [String]
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let cases = try decoder.decode([History].self, from: meta)
        try #require(cases.count == 20)
        let attention = try Attention(context: witness.context, gemmaQATMaxContext: 4097)
        let guardBytes = Data(repeating: 0xa5, count: 64)
        var results: [String: Int] = [:]
        for item in cases {
            let source = try #require(witness.cases.first { $0.position == 4 && $0.layer == item.layer })
            let query = try witness.input(source, "query")
            var histories: [Data] = []
            for stage in ["key", "value"] {
                let rows = try (0..<5).map { try witness.input(source, stage, position: $0) }
                var history = Data()
                for position in 0..<item.length { history.append(rows[position % 5]) }
                histories.append(history)
            }
            for (data, digest) in zip([query] + histories, item.inputSha256) {
                try #require(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == digest)
            }
            for ring in (item.headDim == 256 ? [false, true] : [false]) {
                let capacity = ring ? min(item.length, item.window + 128) : 0
                var inputs = histories
                if ring {
                    let stride = 8 * 256 * 2
                    for index in inputs.indices {
                        var physical = Data(repeating: 0, count: capacity * stride)
                        for position in 0..<item.length {
                            let offset = (position % capacity) * stride
                            physical.replaceSubrange(offset..<offset + stride,
                                with: histories[index][position * stride..<(position + 1) * stride])
                        }
                        inputs[index] = physical
                    }
                }
                let q = try witness.buffer(guardBytes + query + guardBytes)
                let k = try witness.buffer(guardBytes + inputs[0] + guardBytes)
                let v = try witness.buffer(guardBytes + inputs[1] + guardBytes)
                let out = try witness.buffer(guardBytes + Data(repeating: 0, count: item.count * 2) + guardBytes)
                let cb = try #require(witness.context.queue.makeCommandBuffer())
                if item.headDim == 256 {
                    attention.encodeSWA(commandBuffer: cb, q: q, qOffset: 64, k: k, kOffset: 64,
                        v: v, vOffset: 64, out: out, outOffset: 64,
                        headDim: 256, numQHeads: 16, numKVHeads: 8, seqLen: UInt32(item.length),
                        window: UInt32(item.window), scale: 1, ringCapacity: UInt32(capacity))
                } else {
                    attention.encodeFull(commandBuffer: cb, q: q, qOffset: 64, k: k, kOffset: 64,
                        v: v, vOffset: 64, out: out, outOffset: 64,
                        headDim: 512, numQHeads: 16, numKVHeads: 2, seqLen: UInt32(item.length), scale: 1)
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                let actual = out.contents().advanced(by: 64).assumingMemoryBound(to: UInt16.self)
                let count = expected.withUnsafeBytes { bytes in
                    (0..<item.count).reduce(0) {
                        $0 + (actual[$1] == bytes.loadUnaligned(fromByteOffset: item.offset + $1 * 2, as: UInt16.self) ? 0 : 1)
                    }
                }
                let key = "\(item.headDim)/\(item.length)/\(ring ? "ring" : "linear")"
                results[key] = count
                #expect(count == 0, "\(key): \(count) frozen source values differ")
                #expect(Data(bytes: out.contents(), count: 64) == guardBytes)
                #expect(Data(bytes: out.contents().advanced(by: 64 + item.count * 2), count: 64) == guardBytes)
            }
        }
        try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
            .write(to: witness.root.appendingPathComponent("attention-history-native.json"))
        print("[qat-attention-history] \(results)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ATTENTION_REFERENCE"] != nil))
    func installedAttentionMatchesFrozenSourceValues() throws {
        let witness = try QATAttentionWitness()
        let attention = try Attention(context: witness.context, gemmaQATMaxContext: 5)
        var differences: [String: Int] = [:]
        var total = 0
        for item in witness.cases {
            let q = try witness.buffer(witness.input(item, "query"))
            let k = try witness.buffer(witness.history(item, "key"))
            let v = try witness.buffer(witness.history(item, "value"))
            let out = try witness.buffer(Data(repeating: 0, count: item.count * 2))
            let hd = UInt32(item.count / 16)
            let kvHeads: UInt32 = hd == 256 ? 8 : 2
            let cb = try #require(witness.context.queue.makeCommandBuffer())
            if hd == 256 {
                attention.encodeSWA(commandBuffer: cb, q: q, k: k, v: v, out: out,
                    headDim: hd, numQHeads: 16, numKVHeads: kvHeads,
                    seqLen: UInt32(item.position + 1), window: 1024, scale: 1)
            } else {
                attention.encodeFull(commandBuffer: cb, q: q, k: k, v: v, out: out,
                    headDim: hd, numQHeads: 16, numKVHeads: kvHeads,
                    seqLen: UInt32(item.position + 1), scale: 1)
            }
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            differences["\(hd)", default: 0] += witness.differences(out, item)
            total += item.count
        }
        #expect(total == 716_800)
        print("[qat-attention-frozen] count=\(total) differences=\(differences)")
        for (shape, count) in differences { #expect(count == 0, "headDim=\(shape): \(count) frozen source values differ") }
    }

    /// Diagnostic only: compare source-order arithmetic against the unchanged
    /// independently generated witness before changing production dispatch.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ATTENTION_PROBE"] != nil))
    func probeInstalledAttentionReductionOrder() throws {
        let witness = try QATAttentionWitness()
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        constant uint HD [[function_constant(82)]];
        kernel void source_order(
            device const half* query [[buffer(0)]], device const half* keys [[buffer(1)]],
            device const half* values [[buffer(2)]], device half* output [[buffer(3)]],
            constant uint& length [[buffer(4)]], constant uint& kvHeads [[buffer(5)]],
            uint head [[threadgroup_position_in_grid]],
            uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            threadgroup float maxima[32], denominators[32], transpose[1024];
            float q[16], accumulator[16];
            const uint perLane = HD / 32u, kvHead = head / (16u / kvHeads);
            for (uint j = 0; j < perLane; ++j) {
                q[j] = float(query[head * HD + lane * perLane + j]);
                accumulator[j] = 0.0f;
            }
            float maximum = -FLT_MAX, denominator = 0.0f;
            for (uint position = group; position < length; position += 32u) {
                const uint base = (position * kvHeads + kvHead) * HD + lane * perLane;
                float score = 0.0f;
                for (uint j = 0; j < perLane; ++j) score += q[j] * float(keys[base + j]);
                score = simd_sum(score);
                const float updated = max(maximum, score);
                const float factor = fast::exp(maximum - updated);
                const float probability = fast::exp(score - updated);
                denominator = denominator * factor + probability;
                maximum = updated;
                for (uint j = 0; j < perLane; ++j) {
                    accumulator[j] = accumulator[j] * factor + probability * float(values[base + j]);
                }
            }
            if (lane == 0) { maxima[group] = maximum; denominators[group] = denominator; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maximum = maxima[lane];
            const float updated = simd_max(maximum);
            const float factor = fast::exp(maximum - updated);
            denominator = simd_sum(denominators[lane] * factor);
            for (uint j = 0; j < perLane; ++j) {
                transpose[lane * 32u + group] = accumulator[j];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float sum = simd_sum(transpose[group * 32u + lane] * factor);
                if (lane == 0) output[head * HD + group * perLane + j] = half(sum / denominator);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        """
        var differences: [String: Int] = [:]
        for safe in [false, true] {
            let options = MTLCompileOptions()
            options.languageVersion = MetalContext.shaderLanguageVersion
            if safe { options.mathMode = .safe }
            let library = try witness.context.device.makeLibrary(source: source, options: options)
            var pipelines: [Int: MTLComputePipelineState] = [:]
            for dimension in [256, 512] {
                let constants = MTLFunctionConstantValues()
                var hd = UInt32(dimension)
                constants.setConstantValue(&hd, type: .uint, index: 82)
                let function = try library.makeFunction(name: "source_order", constantValues: constants)
                let descriptor = MTLComputePipelineDescriptor()
                descriptor.computeFunction = function
                descriptor.maxTotalThreadsPerThreadgroup = 1024
                let pipeline = try witness.context.device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
                try #require(pipeline.maxTotalThreadsPerThreadgroup >= 1024)
                pipelines[dimension] = pipeline
            }
            for item in witness.cases {
                let hd = item.count / 16
                let q = try witness.buffer(witness.input(item, "query"))
                let k = try witness.buffer(witness.history(item, "key"))
                let v = try witness.buffer(witness.history(item, "value"))
                let out = try witness.buffer(Data(repeating: 0, count: item.count * 2))
                let cb = try #require(witness.context.queue.makeCommandBuffer())
                let enc = try #require(cb.makeComputeCommandEncoder())
                enc.setComputePipelineState(try #require(pipelines[hd]))
                for (i, buffer) in [q, k, v, out].enumerated() { enc.setBuffer(buffer, offset: 0, index: i) }
                var length = UInt32(item.position + 1), kvHeads: UInt32 = hd == 256 ? 8 : 2
                enc.setBytes(&length, length: 4, index: 4)
                enc.setBytes(&kvHeads, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                differences["\(safe ? "safe" : "fast")/\(hd)", default: 0] += witness.differences(out, item)
            }
        }
        let data = try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: witness.root.appendingPathComponent("attention-order-probe.json"))
        print("[qat-attention-order] \(differences)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ATTENTION_PROBE"] != nil))
    func probeInstalledFullAttentionStages() throws {
        let witness = try QATAttentionWitness()
        let stageData = try Data(contentsOf: witness.root.appendingPathComponent("attention-stages.f16"))
        let stageJSON = try Data(contentsOf: witness.root.appendingPathComponent("attention-stages.json"))
        try #require(SHA256.hash(data: stageData).map { String(format: "%02x", $0) }.joined() == "97fb9cb1788af552fceea745a1152daa07590f2b256363fe65b7d41ec249001e")
        try #require(SHA256.hash(data: stageJSON).map { String(format: "%02x", $0) }.joined() == "a7d9d4fe94e5176b96d8e750ed315aea7c78c47eacd8ad1cbd6f623a89f734a5")
        let stages = try JSONDecoder().decode([QATAttentionEntry].self, from: stageJSON)
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        constant uint MODE [[function_constant(81)]];
        kernel void full_stages(
            device const half* query [[buffer(0)]], device const half* keys [[buffer(1)]],
            device const half* values [[buffer(2)]], device half* scores [[buffer(3)]],
            device half* probabilities [[buffer(4)]], device half* output [[buffer(5)]],
            constant uint& length [[buffer(6)]], uint2 grid [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]], uint group [[simdgroup_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]]) {
            const uint head = grid.x, kvHead = head / 8u;
            threadgroup float partial[8];
            if (MODE == 0) {
                float sum = 0.0f;
                for (uint j = 0; j < 4u; ++j) {
                    const uint d = tid * 4u + j;
                    if (d < 512u) sum += float(query[head * 512u + d]) * float(keys[(grid.y * 2u + kvHead) * 512u + d]);
                }
                for (ushort delta = 16; delta > 0; delta >>= 1) sum += simd_shuffle_down(sum, delta);
                if (lane == 0) partial[group] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    sum = partial[0];
                    for (uint g = 1; g < 8u; ++g) sum += partial[g];
                    scores[head * length + grid.y] = half(sum);
                }
            } else if (MODE == 1) {
                // The frozen cases have at most five keys: one source SIMD
                // group with four consecutive scores per lane.
                if (group != 0) return;
                float value[4], maximum = -FLT_MAX;
                for (uint j = 0; j < 4u; ++j) {
                    const uint position = lane * 4u + j;
                    value[j] = position < length ? float(scores[head * length + position]) : -INFINITY;
                    maximum = max(maximum, value[j]);
                }
                maximum = simd_max(maximum);
                float sum = 0.0f;
                for (uint j = 0; j < 4u; ++j) { value[j] = fast::exp(value[j] - maximum); sum += value[j]; }
                const float inverse = 1.0f / simd_sum(sum);
                for (uint j = 0; j < 4u; ++j) {
                    const uint position = lane * 4u + j;
                    if (position < length) probabilities[head * length + position] = half(value[j] * inverse);
                }
            } else {
                for (uint d = tid; d < 512u; d += 256u) {
                    float sum = 0.0f;
                    for (uint position = 0; position < length; ++position) {
                        sum += float(probabilities[head * length + position]) * float(values[(position * 2u + kvHead) * 512u + d]);
                    }
                    output[head * 512u + d] = half(sum);
                }
            }
        }
        """
        var differences: [String: Int] = [:]
        for safe in [false, true] {
            let options = MTLCompileOptions()
            options.languageVersion = MetalContext.shaderLanguageVersion
            if safe { options.mathMode = .safe }
            let library = try witness.context.device.makeLibrary(source: source, options: options)
            let pipelines = try (0..<3).map { mode in
                let constants = MTLFunctionConstantValues()
                var value = UInt32(mode)
                constants.setConstantValue(&value, type: .uint, index: 81)
                let function = try library.makeFunction(name: "full_stages", constantValues: constants)
                return try witness.context.device.makeComputePipelineState(function: function)
            }
            for item in witness.cases where item.count == 8192 {
                let q = try witness.buffer(witness.input(item, "query"))
                let k = try witness.buffer(witness.history(item, "key"))
                let v = try witness.buffer(witness.history(item, "value"))
                let scores = try witness.buffer(Data(repeating: 0, count: (item.position + 1) * 16 * 2))
                let probabilities = try witness.buffer(Data(repeating: 0, count: scores.length))
                let output = try witness.buffer(Data(repeating: 0, count: item.count * 2))
                let cb = try #require(witness.context.queue.makeCommandBuffer())
                for mode in 0..<3 {
                    let enc = try #require(cb.makeComputeCommandEncoder())
                    enc.setComputePipelineState(pipelines[mode])
                    for (index, buffer) in [q, k, v, scores, probabilities, output].enumerated() {
                        enc.setBuffer(buffer, offset: 0, index: index)
                    }
                    var length = UInt32(item.position + 1)
                    enc.setBytes(&length, length: 4, index: 6)
                    enc.dispatchThreadgroups(MTLSize(width: 16, height: mode == 0 ? Int(length) : 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    enc.endEncoding()
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                for (name, buffer) in [("scores", scores), ("probabilities", probabilities), ("output", output)] {
                    let entry = try #require(stages.first { $0.position == item.position && $0.layer == item.layer && $0.stage == name })
                    let actual = buffer.contents().assumingMemoryBound(to: UInt16.self)
                    let count = stageData.withUnsafeBytes { bytes in
                        (0..<entry.count).reduce(0) {
                            $0 + (actual[$1] == bytes.loadUnaligned(fromByteOffset: entry.offset + $1 * 2, as: UInt16.self) ? 0 : 1)
                        }
                    }
                    differences["\(safe ? "safe" : "fast")/\(name)", default: 0] += count
                }
            }
        }
        let data = try JSONSerialization.data(withJSONObject: differences, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: witness.root.appendingPathComponent("attention-full-stage-probe.json"))
        print("[qat-full-attention-stages] \(differences)")
    }
}
