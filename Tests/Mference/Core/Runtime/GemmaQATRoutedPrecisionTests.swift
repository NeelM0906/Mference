import CryptoKit
import Foundation
import Metal
import Testing
@testable import Mference

extension GemmaQATInstalledExecutionTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ROUTED_REFERENCE"] != nil))
    func installedStreamedPrefillFFNMatchesFrozenSourceValues() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_ROUTED_REFERENCE"]))
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name))
            try #require(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == digest)
            return data
        }
        let inputs = try frozen("routed-inputs.f16", "bae4e53080026e9320009b55892907a7f9f8c7e9da1061e876b16ed06b2db592")
        let expected = try frozen("routed-reference.f16", "6f7f5a8931bc79385e8cdb980cf40a5bac3c42a3484eb6bb2356815c30291282")
        let meta = try frozen("routed-cases.json", "512e17a896c9a05e0c39913d51b323297823f0ab94a6d6f0b28b76f9305aa1ef")
        let cases = try #require(JSONSerialization.jsonObject(with: meta) as? [[String: Any]])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), device: context.device)
        try #require(model.modelID == CheckpointIdentity.gemma4QAT)
        func buffer(_ data: Data) throws -> MTLBuffer {
            try #require(data.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        func bytes(_ item: [String: Any], _ key: String, _ data: Data) throws -> Data {
            let spec = try #require(item[key] as? [String: Int])
            let offset = try #require(spec["offset"]), count = try #require(spec["count"])
            return data.subdata(in: offset..<offset + count * 2)
        }
        let kernel = try PrefillGroupedRoutedMoE(context: context, groupSize: 32, sourceFP16: true)
        let reduction = try PrefillMoE(context: context, sourceFP16: true)
        var results: [[String: Int]] = []
        for item in cases {
            let layer = try #require(item["layer"] as? Int)
            let ids = try #require(item["experts"] as? [Int])
            let routeBytes = try bytes(item, "weights", inputs)
            let routeWeights = routeBytes.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
            let pairs = ids.enumerated().map {
                PrefillTokenExpertPair(token: 0, expert: UInt32($0.element), rank: UInt32($0.offset), weight: routeWeights[$0.offset])
            }
            let routes = try PrefillMoEGrouping.groupTokenExpertPairs(pairs,
                queryCount: 1, topK: 8, numExperts: 128, tileExpertCount: 8,
                expertSortKeys: model.routedExpertPhysicalOffsets(layer: layer))
            let binding = try PrefillStreamedTileBinding(expertIDs: ids,
                views: ids.map { try model.routedExpert(layer: layer, expert: $0) })
            let arguments = try kernel.makeStreamedArgumentBuffer(device: context.device, binding: binding)
            let metadata = try kernel.makeStreamedMetadataBuffers(device: context.device, routes: routes)
            let params = PrefillGroupedRoutedMoEStreamedParams(pairStart: 0, pairCount: 8,
                d: 2816, routedIntermediate: 704, topK: 8, hiddenStrideElements: 2816,
                binding: binding, offsets: model.routedExpertOffsets(layer: layer))
            let x = try buffer(bytes(item, "input", inputs))
            let weights = try buffer(routeBytes)
            let scratch = try buffer(Data(repeating: 0, count: 3 * 8 * 704 * 2))
            let down = try buffer(Data(repeating: 0, count: 8 * 2816 * 2))
            let partials = try buffer(Data(repeating: 0, count: 8 * 2816 * 2))
            let output = try buffer(Data(repeating: 0, count: 2816 * 2))
            let cb = try #require(context.queue.makeCommandBuffer())
            let count = kernel.encodeStreamedBatched(commandBuffer: cb, hidden: x,
                sortedPairs: metadata.sortedPairs, routePartials: partials,
                gateUpActScratch: scratch, downScratch: down, argumentBuffer: arguments,
                binding: binding, params: params, pairMicrobatchRows: 8)
            #expect(count == 1)
            reduction.encodeReduceTokenMajor(commandBuffer: cb, routePartials: partials,
                routeWeights: weights, h2: output, queryCount: 1, topK: 8, d: 2816)
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.status == .completed, "\(String(describing: cb.error))")
            let sourceActs = try bytes(item, "expected_activations", expected)
                .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            let actualActs = scratch.contents().assumingMemoryBound(to: UInt16.self).advanced(by: 2 * 8 * 704)
            var actDifferences = 0
            for (row, pair) in routes.sortedPairs.enumerated() {
                for column in 0..<704 {
                    if actualActs[row * 704 + column] != sourceActs[Int(pair.rank) * 704 + column] { actDifferences += 1 }
                }
            }
            let sourceOutput = try bytes(item, "expected_full", expected)
                .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            let actualOutput = output.contents().assumingMemoryBound(to: UInt16.self)
            let outputDifferences = (0..<2816).filter { actualOutput[$0] != sourceOutput[$0] }.count
            results.append(["layer": layer, "activation_differences": actDifferences,
                            "output_differences": outputDifferences])
            #expect(actDifferences == 0, "layer=\(layer): source activation mismatch")
            #expect(outputDifferences == 0, "layer=\(layer): source streamed-prefill output mismatch")
        }
        try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("routed-prefill-native.json"))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_ROUTED_REFERENCE"] != nil))
    func installedRoutedFFNMatchesFrozenSourceValues() throws {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_ROUTED_REFERENCE"]))
        func frozen(_ name: String, _ digest: String) throws -> Data {
            let data = try Data(contentsOf: root.appendingPathComponent(name))
            try #require(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == digest)
            return data
        }
        let inputs = try frozen("routed-inputs.f16", "bae4e53080026e9320009b55892907a7f9f8c7e9da1061e876b16ed06b2db592")
        let expected = try frozen("routed-reference.f16", "6f7f5a8931bc79385e8cdb980cf40a5bac3c42a3484eb6bb2356815c30291282")
        let meta = try frozen("routed-cases.json", "512e17a896c9a05e0c39913d51b323297823f0ab94a6d6f0b28b76f9305aa1ef")
        let cases = try #require(JSONSerialization.jsonObject(with: meta) as? [[String: Any]])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_GTURBO"])), device: context.device)
        try #require(model.modelID == CheckpointIdentity.gemma4QAT)
        func buffer(_ data: Data) throws -> MTLBuffer {
            try #require(data.withUnsafeBytes { context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        func bytes(_ item: [String: Any], _ key: String, _ data: Data) throws -> Data {
            let spec = try #require(item[key] as? [String: Int])
            let offset = try #require(spec["offset"]), count = try #require(spec["count"])
            return data.subdata(in: offset..<offset + count * 2)
        }
        let residual = try buffer(Data(repeating: 0, count: 2816 * 2))
        for specialized in [false, true] {
            let kernel = try MoE(context: context, specializedD: specialized ? 2816 : 128,
                specializedF: 704, groupSize: 32, sourceFP16: true)
            for item in cases {
                let layer = try #require(item["layer"] as? Int)
                let ids = try #require(item["experts"] as? [Int])
                let blobs = try ids.map { id -> (buffer: MTLBuffer, offset: Int) in
                    let view = try model.routedExpert(layer: layer, expert: id)
                    return (view.buffer, Int(view.offset))
                }
                let arguments = try #require(kernel.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: 8))
                let offsets = model.routedExpertOffsets(layer: layer)
                let x = try buffer(bytes(item, "input", inputs))
                let frozenActs = try buffer(bytes(item, "native_activations", inputs))
                let weights = try buffer(bytes(item, "weights", inputs))
                let acts = try buffer(Data(repeating: 0, count: 8 * 704 * 2))
                let down = try buffer(Data(repeating: 0, count: 2816 * 2))
                let full = try buffer(Data(repeating: 0, count: 2816 * 2))
                let cb = try #require(context.queue.makeCommandBuffer())
                kernel.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                    routedArgBuffer: arguments, routedBlobs: blobs, routedOffsets: offsets,
                    x: x, acts: acts, d: 2816, f: 704, topK: 8)
                for (activation, output) in [(frozenActs, down), (acts, full)] {
                    kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb,
                        routedArgBuffer: arguments, routedBlobs: blobs, routedOffsets: offsets,
                        acts: activation, routingWeights: weights, residual: residual, y: output,
                        d: 2816, f: 704, topK: 8)
                }
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.status == .completed)
                for (key, actual) in [("expected_activations", acts), ("expected_down", down), ("expected_full", full)] {
                    let source = try bytes(item, key, expected)
                    #expect(Data(bytes: actual.contents(), count: actual.length) == source,
                            "layer=\(layer), specialized=\(specialized), stage=\(key)")
                }
            }
        }
    }
}
