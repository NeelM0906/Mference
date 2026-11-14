import Foundation
import Metal
import Testing
@testable import Mference

/// Component numerical/state gates only. These do not establish target-token
/// alignment, upstream native-draft parity, acceptance or a speed improvement.
@Suite(.serialized) struct FlashNextMTPDraftRunnerTests {
    private struct Harness {
        let directory: URL
        let context: MetalContext
        let model: Model
        let runner: FlashNextMTPDraftRunner
        let embedding: MTLBuffer
        let hidden: MTLBuffer
        let logits: MTLBuffer

        func append(_ row: Int, variant: Int = 0) throws -> FlashNextMTPDraftRunner.Output {
            let e = embedding.contents().assumingMemoryBound(to: Float16.self)
            let h = hidden.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<model.config.hiddenSize { e[i] = Float16(Float((i * 3 + row + variant) % 17 - 8) / 16) }
            for i in 0..<model.config.residualStreamWidth { h[i] = Float16(Float((i * 7 + row * 3 + variant) % 23 - 11) / 8) }
            logits.contents().assumingMemoryBound(to: Float16.self).update(repeating: .nan, count: model.config.vocabSize)
            return try runner.append(embedding: embedding, targetHidden: hidden, at: row, into: logits)
        }

        func bits(_ buffer: MTLBuffer) throws -> [UInt16] {
            let readback = try #require(context.device.makeBuffer(length: buffer.length, options: .storageModeShared))
            let cb = try #require(context.queue.makeCommandBuffer())
            let blit = try #require(cb.makeBlitCommandEncoder())
            blit.copy(from: buffer, sourceOffset: 0, to: readback, destinationOffset: 0, size: buffer.length)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            try #require(cb.error == nil)
            let values = Array(UnsafeBufferPointer(start: readback.contents().assumingMemoryBound(to: UInt16.self), count: buffer.length / 2))
            #expect(values.allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(values.contains { Float16(bitPattern: $0) != 0 })
            return values
        }
    }

    private func make(resident: Bool = false, maxContext: Int = 48) throws -> Harness {
        let directory = try FlashNextToySynthetic.write(includeMTP: true)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen38FlashNextToy(), streamingMode: .pread(slotCount: 16))
        let runner = try FlashNextMTPDraftRunner(model: model, context: context, maxContext: maxContext,
            policy: resident ? .resident : .bounded(slots: 6))
        func buffer(_ count: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: count * 2, options: .storageModeShared))
        }
        return Harness(directory: directory, context: context, model: model, runner: runner,
            embedding: try buffer(model.config.hiddenSize), hidden: try buffer(model.config.residualStreamWidth),
            logits: try buffer(model.config.vocabSize))
    }

    @Test func boundedAndResidentMatchAcrossSparseSelection() throws {
        func run(resident: Bool) throws -> [[UInt16]] {
            let h = try make(resident: resident)
            defer { try? FileManager.default.removeItem(at: h.directory) }
            var result: [[UInt16]] = []
            for row in 0..<40 {
                let out = try h.append(row)
                #expect(out.processedRows == row + 1)
                #expect(out.hidden.length == h.model.config.residualStreamWidth * 2)
                result.append(try h.bits(out.hidden))
                result.append(try h.bits(h.logits))
            }
            return result
        }
        let bounded = try run(resident: false)
        let resident = try run(resident: true)
        #expect(bounded == resident)
    }

    @Test func nativeLayerTracksIndependentFP32Composition() throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        let reference = try FlashNextMTPReference(model: h.model, device: h.context.device)
        let fusionRounded = try FlashNextMTPReference(model: h.model, device: h.context.device, roundFusionStores: true)
        var stages: [String: [Float]] = [:]
        var exportedRows: [[String: Any]] = []
        let exportPath = ProcessInfo.processInfo.environment["MFERENCE_MTP_REFERENCE_EXPORT"]
        h.runner.didCaptureStages = { stages = $0 }
        func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
            let p = buffer.contents().assumingMemoryBound(to: Float16.self)
            return (0..<count).map { Float(p[$0]) }
        }
        func compare(_ actual: [UInt16], _ expected: [Float], label: String) {
            #expect(actual.count == expected.count)
            #expect(expected.allSatisfy { $0.isFinite })
            let scale = expected.map { abs($0) }.max() ?? 0
            let error = zip(actual, expected).map { abs(Float(Float16(bitPattern: $0)) - $1) }.max() ?? .infinity
            #expect(scale > 0 && error.isFinite)
            // Same FP16-vs-FP32 semantic tier as family integration gates, not
            // bit parity or an exact speculative-verifier tolerance.
            #expect(error <= scale * 0.05)
            print("[MTP FP32 composition] \(label) maxAbs=\(error) scale=\(scale)")
        }
        for row in 0..<40 {
            let output = try h.append(row)
            let expected = try reference.append(embedding: floats(h.embedding, count: h.model.config.hiddenSize),
                hidden: floats(h.hidden, count: h.model.config.residualStreamWidth))
            let rounded = try fusionRounded.append(embedding: floats(h.embedding, count: h.model.config.hiddenSize),
                hidden: floats(h.hidden, count: h.model.config.residualStreamWidth))
            compare(try h.bits(output.hidden), rounded.hidden, label: "row=\(row) fusion-rounded hidden")
            compare(try h.bits(h.logits), rounded.logits, label: "row=\(row) fusion-rounded logits")
            let fusion = try #require(stages["fusion"])
            let expectedFusion = try #require(fusionRounded.stages["fusion"])
            let fusionError = zip(fusion, expectedFusion).map { abs($0 - $1) }.max()!
            #expect(fusion.allSatisfy { $0.isFinite })
            #expect(fusionError <= expectedFusion.map { abs($0) }.max()! * 0.002)
            if row == 0 {
                print("[MTP fusion storage oracle] maxAbs=\(fusionError)")
                for name in stages.keys.sorted() {
                    let desired = try #require(reference.stages[name])
                    let error = zip(stages[name]!, desired).map { abs($0 - $1) }.max()!
                    print("[MTP stage] \(name) maxAbs=\(error) scale=\(desired.map { abs($0) }.max()!)")
                }
                print("[MTP routes] gpu=\(FlashNextRouterReference.select(logits: stages["router"]!, k: h.model.config.topKExperts)) cpu=\(FlashNextRouterReference.select(logits: reference.stages["router"]!, k: h.model.config.topKExperts))")
            }
            compare(try h.bits(output.hidden), expected.hidden, label: "row=\(row) hidden")
            let actualLogits = try h.bits(h.logits)
            if exportPath != nil {
                exportedRows.append([
                    "embedding": floats(h.embedding, count: h.model.config.hiddenSize),
                    "hidden": floats(h.hidden, count: h.model.config.residualStreamWidth),
                    "output_hidden": try h.bits(output.hidden).map { Float(Float16(bitPattern: $0)) },
                    "logits": actualLogits.map { Float(Float16(bitPattern: $0)) },
                    "stages": stages
                ])
            }
            compare(actualLogits, expected.logits, label: "row=\(row) logits")
            let predicted = actualLogits.indices.max { Float16(bitPattern: actualLogits[$0]) < Float16(bitPattern: actualLogits[$1]) }
            let desired = expected.logits.indices.max { expected.logits[$0] < expected.logits[$1] }
            #expect(predicted == desired)
        }
        if let exportPath {
            let object: [String: Any] = ["weights": try reference.exportWeights(), "rows": exportedRows]
            let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            // Refuse to overwrite an earlier reference capture.
            try data.write(to: URL(fileURLWithPath: exportPath), options: .withoutOverwriting)
        }
    }

    @Test func nativeLayerTracksPinnedUpstreamComponents() throws {
        struct Golden: Decodable {
            struct Row: Decodable {
                let embedding: [Float]
                let hidden: [Float]
                let output_hidden: [Float]
                let logits: [Float]
            }
            let transformers_commit: String
            let indexer_tie_policy: String
            let rows: [Row]
        }
        let url = FlashNextParity.repoRoot.appendingPathComponent("Tests/Mference/Fixtures/flashnext-mtp-upstream.json")
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
        #expect(golden.transformers_commit == "4da05482135896a529d5536c3c003102d36528a2")
        // PyTorch topk does not promise stable tied indices. The fixture
        // explicitly uses our lowest-index QSA tie contract, not CPU topk's
        // incidental order. All arithmetic remains in upstream modules.
        #expect(golden.indexer_tie_policy == "lowest-index")
        try #require(golden.rows.count == 40)
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        for (position, row) in golden.rows.enumerated() {
            let output = try h.append(position)
            for (buffer, expected) in [(h.embedding, row.embedding), (h.hidden, row.hidden)] {
                let actual = try h.bits(buffer).map { Float(Float16(bitPattern: $0)) }
                #expect(actual == expected)
            }
            for (buffer, expected) in [(output.hidden, row.output_hidden), (h.logits, row.logits)] {
                let actual = try h.bits(buffer).map { Float(Float16(bitPattern: $0)) }
                try #require(actual.count == expected.count)
                let scale = try #require(expected.map(abs).max())
                let error = try #require(zip(actual, expected).map { abs($0 - $1) }.max())
                #expect(error.isFinite && scale > 0 && error <= scale * 0.05)
            }
            let actual = try h.bits(h.logits).map { Float(Float16(bitPattern: $0)) }
            #expect(actual.indices.max { actual[$0] < actual[$1] } == row.logits.indices.max { row.logits[$0] < row.logits[$1] })
        }
    }

    @Test(arguments: [false, true])
    func rejectedBranchRestoresCacheAndOwnedOutputs(resident: Bool) throws {
        let h = try make(resident: resident)
        defer { try? FileManager.default.removeItem(at: h.directory) }
        for row in 0..<31 { _ = try h.append(row) }
        let checkpoint = try h.runner.checkpoint()
        var expected: [[UInt16]] = []
        let owned = try h.append(31)
        let saved = try h.bits(owned.hidden)
        expected.append(saved)
        expected.append(try h.bits(h.logits))
        for row in 32..<37 {
            expected.append(try h.bits(h.append(row).hidden))
            expected.append(try h.bits(h.logits))
        }
        try h.runner.restore(checkpoint)
        let retry = try h.runner.checkpoint()
        for row in 31..<39 { _ = try h.append(row, variant: 9) }
        try h.runner.restore(retry)
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restore(retry) }
        var actual: [[UInt16]] = []
        for row in 31..<37 {
            actual.append(try h.bits(h.append(row).hidden))
            actual.append(try h.bits(h.logits))
        }
        #expect(actual == expected)
        #expect(try h.bits(owned.hidden) == saved, "owned HC row cannot alias mutable draft scratch")
        let stale = try h.runner.checkpoint()
        h.runner.reset()
        #expect(throws: FlashNextForwardRunnerError.self) { try h.runner.restore(stale) }
        _ = try h.append(0)
        #expect(try h.bits(owned.hidden) == saved)
    }

    @Test func partialGPUFailureRequiresRestoreOrReset() throws {
        let h = try make()
        defer { try? FileManager.default.removeItem(at: h.directory) }
        _ = try h.append(0)
        let parent = try h.runner.checkpoint()
        let expected = try h.bits(h.append(1).hidden)
        let expectedLogits = try h.bits(h.logits)
        try h.runner.restore(parent)
        let checkpoint = try h.runner.checkpoint()
        h.runner.didPrepareAttention = { throw CancellationError() }
        #expect(throws: CancellationError.self) { _ = try h.append(1, variant: 9) }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.runner.checkpoint() }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.append(1) }
        h.runner.didPrepareAttention = nil
        try h.runner.restore(checkpoint)
        #expect(try h.bits(h.append(1).hidden) == expected)
        #expect(try h.bits(h.logits) == expectedLogits)
    }

    @Test func missingPrefixContextExhaustionAndForeignCheckpointsReject() throws {
        let h = try make(maxContext: 2)
        defer { try? FileManager.default.removeItem(at: h.directory) }
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.append(1) }
        #expect(h.runner.position == 0)
        _ = try h.append(0)
        let checkpoint = try h.runner.checkpoint()
        let other = try FlashNextMTPDraftRunner(model: h.model, context: h.context, maxContext: 2, policy: .bounded(slots: 6))
        #expect(throws: FlashNextForwardRunnerError.self) { try other.restore(checkpoint) }
        _ = try h.append(1)
        #expect(throws: FlashNextForwardRunnerError.self) { _ = try h.append(2) }
        #expect(h.runner.position == 2)
        _ = try h.runner.checkpoint()
        #expect(throws: FlashNextForwardRunnerError.self) {
            _ = try FlashNextMTPDraftRunner(model: h.model, context: h.context, maxContext: 2, policy: .bounded(slots: 5))
        }
    }

    @Test func installedDraftExecutesAndRestoresItsOwnState() throws {
        guard let path = ProcessInfo.processInfo.environment["MFERENCE_FLASHNEXT_GTURBO"] else { return }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device,
            streamingMode: .pread(slotCount: 16))
        let cfg = model.config
        let eValues = (0..<cfg.hiddenSize).map { Float16(Float($0 % 19 - 9) / 16) }
        let hValues = (0..<cfg.residualStreamWidth).map { Float16(Float($0 % 23 - 11) / 8) }
        let embedding = try #require(context.device.makeBuffer(bytes: eValues, length: eValues.count * 2, options: .storageModeShared))
        let hidden = try #require(context.device.makeBuffer(bytes: hValues, length: hValues.count * 2, options: .storageModeShared))
        let logits = try #require(context.device.makeBuffer(length: cfg.vocabSize * 2, options: .storageModeShared))
        func run(resident: Bool) throws -> [[UInt16]] {
            let runner = try FlashNextMTPDraftRunner(model: model, context: context, maxContext: 8,
                policy: resident ? .resident : .bounded(slots: 16))
            func append(_ position: Int) throws -> [UInt16] {
                logits.contents().assumingMemoryBound(to: Float16.self).update(repeating: .nan, count: cfg.vocabSize)
                let output = try runner.append(embedding: embedding, targetHidden: hidden, at: position, into: logits)
                #expect(output.processedRows == position + 1)
                let readback = try #require(context.device.makeBuffer(length: output.hidden.length, options: .storageModeShared))
                let cb = try #require(context.queue.makeCommandBuffer())
                let blit = try #require(cb.makeBlitCommandEncoder())
                blit.copy(from: output.hidden, sourceOffset: 0, to: readback, destinationOffset: 0, size: readback.length)
                blit.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                try #require(cb.error == nil)
                let hc = Array(UnsafeBufferPointer(start: readback.contents().assumingMemoryBound(to: UInt16.self), count: cfg.residualStreamWidth))
                let head = Array(UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: UInt16.self), count: cfg.vocabSize))
                #expect(hc.allSatisfy { Float16(bitPattern: $0).isFinite })
                #expect(head.allSatisfy { Float16(bitPattern: $0).isFinite })
                #expect(hc.contains { Float16(bitPattern: $0) != 0 })
                #expect(head.contains { Float16(bitPattern: $0) != 0 })
                return hc + head
            }
            let first = try append(0)
            let checkpoint = try runner.checkpoint()
            let second = try append(1)
            _ = try append(2)
            try runner.restore(checkpoint)
            #expect(try append(1) == second)
            runner.reset()
            #expect(try append(0) == first)
            return [first, second]
        }
        let bounded = try run(resident: false)
        let resident = try run(resident: true)
        #expect(bounded == resident)
        print("[installed MTP draft] resident/16-slot finite HC + full logits, rollback and reset exact")
        if let path = ProcessInfo.processInfo.environment["MFERENCE_MTP_INSTALLED_REFERENCE_EXPORT"] {
            let rows: [[String: Any]] = bounded.map { values in
                ["embedding": eValues.map(Float.init), "hidden": hValues.map(Float.init),
                 "output_hidden": values.prefix(cfg.residualStreamWidth).map { Float(Float16(bitPattern: $0)) },
                 "logits": values.dropFirst(cfg.residualStreamWidth).map { Float(Float16(bitPattern: $0)) }]
            }
            let data = try JSONSerialization.data(withJSONObject: ["rows": rows], options: [.sortedKeys])
            try data.write(to: URL(fileURLWithPath: path), options: .withoutOverwriting)
        }
        // Inputs are deterministic probes, not target-aligned hidden states.
        // Do not report this as acceptance, end-to-end generation or speed.
    }
}
