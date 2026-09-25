import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_PREFILL_PROJECTION_REFERENCE"] != nil))
    func installedPrefillProjectionsMatchFrozenSourceValues() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_PREFILL_PROJECTION_REFERENCE"]))
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name), options: .mappedIfSafe)
            try #require(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == digest)
            return data
        }
        struct Case: Decodable {
            let layer: Int, role: String, weight: String, positions: [Int]
            let rows: Int, columns: Int, inputOffset: Int, outputOffset: Int
        }
        let inputs = try frozen("prefill-projection-inputs.f16", "f22d8ef658a05ac3186c1c97c45b63b801cb91574d71daad7fd65c0d83d996c2")
        let expected = try frozen("prefill-projection-reference.f16", "c5c50abfeca81b19031bb48a4401b133fc680b9bc980e6fc4e605264d6fbe59f")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let cases = try decoder.decode([Case].self, from: frozen("prefill-projection-cases.json",
            "915934e01e21bebf51a48ed89b69a01ff9f11dde9efc706f83e8f63fe6f13404"))
        try #require(cases.count == 12)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), device: context.device)
        try #require(model.modelID == CheckpointIdentity.gemma4QAT)
        let mpp = MPPPrefillInt4QMM(context: context, groupSize: 32, sourceFP16: true)
        try #require(mpp.isAvailable, "This installed-M2 witness requires the active MPP path")
        let qmm = try PrefillInt4QMM(context: context, groupSize: 32, sourceFP16: true)
        // Diagnostic candidate: exactly the same MPP shader, compiled safely.
        let safe = try context.pipeline("mpp_prefill_affine_threadgroup_f16",
            constants: Quantization.int4Constants(groupSize: 32) + Quantization.gemmaSourceConstants(enabled: true),
            maxTotalThreadsPerThreadgroup: nil, safeMathModule: "tensorops")
        func buffer(_ data: Data) throws -> MTLBuffer {
            try #require(data.withUnsafeBytes {
                context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            })
        }
        var results: [String: Int] = [:]
        var compared = 0, productionDifferences = 0
        var bosOutputs: [String: Data] = [:]
        for item in cases {
            let view = try model.resident(name: item.weight)
            for count in [32, 65, 128] {
                var xData = Data(), source = Data()
                for row in 0..<count {
                    let i = row % item.positions.count
                    let offset = item.inputOffset + i * item.columns * 2
                    xData.append(inputs.subdata(in: offset..<offset + item.columns * 2))
                    let out = item.outputOffset + i * item.rows * 2
                    source.append(expected.subdata(in: out..<out + item.rows * 2))
                }
                let x = try buffer(xData)
                for variant in ["mpp", "qmm", "safe-mpp"] {
                    let y = try buffer(Data(repeating: 0xa5, count: count * item.rows * 2))
                    let cb = try #require(context.queue.makeCommandBuffer())
                    if variant == "mpp" {
                        #expect(mpp.encode(commandBuffer: cb, weights: view.buffer, weightsOffset: Int(view.offset),
                            scales: view.buffer, scalesOffset: Int(view.scaleOffset), biases: view.buffer,
                            biasesOffset: Int(view.biasOffset), x: x, y: y,
                            m: count, n: item.rows, k: item.columns) == .sourceAffineF16)
                    } else if variant == "qmm" {
                        qmm.encode(commandBuffer: cb, weights: view.buffer, weightsOffset: Int(view.offset),
                            scales: view.buffer, scalesOffset: Int(view.scaleOffset), biases: view.buffer,
                            biasesOffset: Int(view.biasOffset), x: x, y: y,
                            t: count, n: item.rows, k: item.columns)
                    } else {
                        let enc = try #require(cb.makeComputeCommandEncoder())
                        enc.setComputePipelineState(safe)
                        enc.setBuffer(view.buffer, offset: Int(view.offset), index: 0)
                        enc.setBuffer(view.buffer, offset: Int(view.scaleOffset), index: 1)
                        enc.setBuffer(view.buffer, offset: Int(view.biasOffset), index: 2)
                        enc.setBuffer(x, offset: 0, index: 3)
                        enc.setBuffer(y, offset: 0, index: 4)
                        var m = UInt32(count), n = UInt32(item.rows), k = UInt32(item.columns)
                        enc.setBytes(&m, length: 4, index: 5)
                        enc.setBytes(&n, length: 4, index: 6)
                        enc.setBytes(&k, length: 4, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: (item.rows + 31) / 32, height: (count + 63) / 64, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: safe.threadExecutionWidth * 4, height: 1, depth: 1))
                        enc.endEncoding()
                    }
                    cb.commit()
                    cb.waitUntilCompleted()
                    try #require(cb.status == .completed, "\(String(describing: cb.error))")
                    let actual = y.contents().assumingMemoryBound(to: UInt16.self)
                    let differences = source.withUnsafeBytes { bytes in
                        (0..<count * item.rows).reduce(0) {
                            $0 + (actual[$1] == bytes.loadUnaligned(fromByteOffset: $1 * 2, as: UInt16.self) ? 0 : 1)
                        }
                    }
                    results["\(item.layer)/\(item.role)/\(count)/\(variant)"] = differences
                    if variant != "safe-mpp" { productionDifferences += differences; compared += count * item.rows }
                    if item.layer == 0 && item.role == "q" && count == 128 {
                        bosOutputs[variant] = Data(bytes: y.contents(), count: item.rows * 2)
                        bosOutputs["source"] = source.prefix(item.rows * 2)
                    }
                }
            }
        }
        // Apply the real query epilogue to BOS. MPP's red output must explain
        // the captured query mismatch, rather than merely a synthetic error.
        let norm = try PrefillPerHeadNorm(context: context, sourceFP16: true)
        let rope = try PrefillRoPE(context: context, sourceFP16: true)
        let weight = try model.resident(name: "language_model.model.layers.0.self_attn.q_norm.weight")
        var queries: [String: Data] = [:]
        for (name, data) in bosOutputs {
            let q = try buffer(data)
            let cb = try #require(context.queue.makeCommandBuffer())
            norm.encodeBF16W(commandBuffer: cb, x: q, weight: weight.buffer, weightOffset: Int(weight.offset),
                out: q, queryCount: 1, headDim: 256, numHeads: 16, tokenStrideElements: 4096,
                eps: 1e-6)
            rope.encodeDefaultNeox(commandBuffer: cb, data: q, startPosition: 0, queryCount: 1,
                headDim: 256, numHeads: 16, tokenStrideElements: 4096, theta: Float(model.config.ropeTheta))
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed)
            queries[name] = Data(bytes: q.contents(), count: q.length)
        }
        for (name, data) in queries { try data.write(to: root.appendingPathComponent("prefill-bos-query-\(name).f16")) }
        try JSONSerialization.data(withJSONObject: ["compared": compared, "different": productionDifferences,
            "cases": results], options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("prefill-projection-native.json"))
        print("[qat-prefill-projection] compared=\(compared) different=\(productionDifferences)")
        #expect(productionDifferences == 0)
    }
}
