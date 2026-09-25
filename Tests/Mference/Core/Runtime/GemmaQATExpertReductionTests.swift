import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_REDUCTION_REFERENCE"] != nil))
    func installedExpertReductionMatchesSourceHalfSum() throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_REDUCTION_REFERENCE"]))
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name))
            let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            try #require(actual == digest)
            return data
        }
        // Pinned MLX outputs from the actual BOS/layer-0 selected expert
        // matrices, on unchanged v7 native activations. No fabricated weights.
        let partials = try frozen("reduction-partials.f16", "d30472672f01e8eb543c9cfc297deb063943a7a2a8434bd4457120ab61248f83")
        let weights = try frozen("reduction-weights.f16", "eed92c7a019c523279320fabb93f653b5f1a025d6d1dc963e2fc379dab52528e")
        let expected = try frozen("reduction-reference.f16", "da7aa27a77d57e165fd3bc9eb78abc6bf372a8c80aa95eef2d6c9bd98d757113")
        try #require(partials.count == 8 * 2816 * 2 && weights.count == 16 && expected.count == 2816 * 2)
        let context = try MetalContext()
        func buffer(_ bytes: Data) throws -> MTLBuffer {
            try #require(bytes.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        let input = try buffer(partials), routes = try buffer(weights)
        let output = try #require(context.device.makeBuffer(length: expected.count, options: .storageModeShared))
        let kernel = try PrefillMoE(context: context, sourceFP16: true)
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encodeReduceTokenMajor(commandBuffer: cb, routePartials: input,
            routeWeights: routes, h2: output, queryCount: 1, topK: 8, d: 2816)
        cb.commit()
        cb.waitUntilCompleted()
        try #require(cb.status == .completed)
        let actual = Data(bytes: output.contents(), count: expected.count)
        #expect(actual == expected)
    }
}
