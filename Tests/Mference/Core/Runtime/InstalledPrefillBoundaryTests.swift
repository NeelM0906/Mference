import Foundation
import Metal
import Testing
@testable import Mference

/// Opt-in real-checkpoint coverage. Run only after AGENTS.md preflight; each
/// arm releases its model before the next arm opens the same install.
@Suite(.serialized) struct InstalledPrefillBoundaryTests {
    private struct Profile {
        let gate: String
        let family: ModelFamily
        let boundaries: [Int]
        let chunk: Int
        let slots: Int

        static func named(_ name: String) -> Profile {
            switch name {
            case "gemma4":
                return Profile(gate: "MFERENCE_GEMMA4_GTURBO", family: .gemma4,
                    boundaries: [33, 1023, 1027, 1059], chunk: 128, slots: 8)
            case "qwen36":
                return Profile(gate: "MFERENCE_QWEN36_GTURBO", family: .qwen36,
                    boundaries: [33, 127, 131, 163], chunk: 64, slots: 8)
            case "maple":
                return Profile(gate: "MFERENCE_MAPLE_GTURBO", family: .maple,
                    boundaries: [33, 511, 515, 547], chunk: 64, slots: 8)
            case "inkling":
                return Profile(gate: "MFERENCE_INKLING_GTURBO", family: .inklingSmall,
                    boundaries: [33, 511, 515, 547], chunk: 128, slots: 16)
            default:
                return Profile(gate: "MFERENCE_FLASHNEXT_GTURBO", family: .qwen38flashnext,
                    boundaries: [33, 2047, 2051, 2083], chunk: 1024, slots: 16)
            }
        }
    }

    private struct Result {
        let heads: [[UInt16]]
        let tail: [[UInt16]]
        let continuation: [Int32]
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data(("[installed boundary] " + message + "\n").utf8))
    }

    private static func cancellationHook(_ producer: any ContinuableLogitProducer,
                                         enabled: Bool) throws {
        let hook: ((Int) throws -> Void)? = enabled ? { layer in
            if layer == 3 { throw CancellationError() }
        } : nil
        if let runner = producer as? MapleForwardRunner {
            runner.prefillWillEncodeLayer = hook
        } else if let runner = producer as? RealForwardRunner {
            runner.prefillWillEncodeLayer = hook
        } else if let runner = producer as? FlashNextForwardRunner {
            runner.prefillDidCompleteLayer = hook
        } else {
            throw CocoaError(.featureUnsupported)
        }
    }

    private static func runArm(path: String, profile: Profile, resident: Bool) async throws -> Result {
        let context = try MetalContext()
        let url = URL(fileURLWithPath: path)
        // Maple's runner requires bounded slots; it does not implement the
        // resident-expert binding. Compare its supported 16/8-slot profiles.
        let mode: ExpertStreamingMode = profile.family == .maple
            ? .pread(slotCount: resident ? 16 : profile.slots)
            : (resident ? .resident : .pread(slotCount: profile.slots))
        let model = try Model.load(directoryURL: url, device: context.device,
            streamingMode: mode)
        try #require(model.config.family == profile.family)
        let tokenizer = try await MFTokenizer.load(forModelDirectory: url)
        let count = try #require(profile.boundaries.last)
        let runtime = try ForwardRunnerFactory.make(model: model, context: context,
            maxContext: count + 32, runtimeConfiguration: RuntimeConfiguration(
                expertCacheSlots: profile.slots, prefillChunkTokens: profile.chunk,
                forceLogitsHead: true))
        let producer = runtime.producer
        let runner = try #require(producer as? any ChunkedPrefillRunner)
        let output = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        let source = tokenizer.encode(String(repeating:
            "The town records daily river levels, compares weekly averages, and preserves the original measurements.\n",
            count: 300), addBOS: true)
        try #require(source.count >= count)
        let tokens = Array(source.prefix(count))
        let validVocab = model.config.unpaddedVocabSize > 0
            ? model.config.unpaddedVocabSize : model.config.vocabSize
        func snapshot() -> [UInt16] {
            let bits = Array(UnsafeBufferPointer(
                start: output.contents().assumingMemoryBound(to: UInt16.self),
                count: model.config.vocabSize))
            #expect(bits.prefix(validVocab).allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(bits.dropFirst(validVocab).allSatisfy { Float16(bitPattern: $0) == -.infinity })
            return bits
        }
        func greedy() -> Int32 {
            let values = output.contents().assumingMemoryBound(to: Float16.self)
            var best = 0
            for i in 1..<validVocab where values[i] > values[best] { best = i }
            return Int32(best)
        }
        func targetHidden(_ flash: FlashNextForwardRunner) throws -> [UInt16] {
            let hidden = try flash.captureTargetHiddenBundle()
            #expect(hidden.processedTokenCount == flash.continuationPosition)
            #expect(hidden.buffer.length == model.config.residualStreamWidth * 2)
            let readback = try #require(context.device.makeBuffer(length: hidden.buffer.length,
                                                                 options: .storageModeShared))
            let command = try #require(context.queue.makeCommandBuffer())
            let blit = try #require(command.makeBlitCommandEncoder())
            blit.copy(from: hidden.buffer, sourceOffset: 0, to: readback,
                      destinationOffset: 0, size: readback.length)
            blit.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            #expect(command.error == nil)
            let bits = Array(UnsafeBufferPointer(start: readback.contents().assumingMemoryBound(to: UInt16.self),
                                                count: readback.length / 2))
            #expect(bits.allSatisfy { Float16(bitPattern: $0).isFinite })
            #expect(bits.contains { Float16(bitPattern: $0) != 0 })
            return bits
        }
        func append(_ start: Int, _ end: Int) async throws {
            if start > 0 { try producer.prepareForContinuation(expectedPosition: start) }
            let result = try await runner.prefillChunked(tokens: tokens[start..<end],
                startPosition: start, outputMode: .logits, config: runtime.prefillConfig,
                into: output, onProgress: { _ in })
            #expect(result.execution?.batchedTokens == end - start)
            #expect(result.execution?.replayedTokens == 0)
            #expect(result.execution?.batchedChunkSizes.reduce(0, +) == end - start)
            #expect(result.newPosition == end)
            log("\(profile.family) append=\(start)..<\(end) chunks=\(result.execution?.batchedChunkSizes ?? [])")
        }
        let modeLabel = profile.family == .maple && resident ? "slots=16"
            : (resident ? "resident" : "slots=\(profile.slots)")
        let label = "\(profile.family) \(modeLabel)"
        log("\(label) strict verified, chunk=\(profile.chunk)")
        var heads: [[UInt16]] = []
        var start = 0
        for end in profile.boundaries {
            try await append(start, end)
            heads.append(snapshot())
            log("\(label) position=\(end) replay=0")
            start = end
        }
        if let flash = producer as? FlashNextForwardRunner {
            #expect((flash.deviceGroupedPrefillLayers > 0) == resident)
            #expect((flash.tensorOpsPrefillEncodings > 0) == flash.tensorOpsPrefillAvailable)
            log("\(label) TensorOps available=\(flash.tensorOpsPrefillAvailable) encodings=\(flash.tensorOpsPrefillEncodings)")
        }
        var continuation: [Int32] = []
        var tail: [[UInt16]] = []
        let flash = producer as? FlashNextForwardRunner
        let committedHidden = try flash.map { try targetHidden($0) }
        let checkpoint = try flash?.captureDecodeCheckpoint()
        for step in 0..<8 {
            let token = greedy()
            continuation.append(token)
            try await producer.produce(token: token, position: count + step, into: output)
            tail.append(snapshot())
        }
        if let flash, let checkpoint {
            try flash.restoreDecodeCheckpoint(checkpoint)
            #expect(try targetHidden(flash) == committedHidden)
            let clean = try flash.captureDecodeCheckpoint()
            // A rejected draft includes EOS and crosses a pooled-indexer block
            // boundary above the sparse cutover. Its rows must become invisible.
            let wrong = [Int32(model.config.flashNext.pleEosTokenID), 7, 11, 19]
            for (step, token) in wrong.enumerated() {
                try await flash.produce(token: token, position: count + step, into: output)
            }
            try flash.restoreDecodeCheckpoint(clean)
            #expect(try targetHidden(flash) == committedHidden)
            try flash.prepareForContinuation(expectedPosition: count)
            for (step, token) in continuation.enumerated() {
                try await flash.produce(token: token, position: count + step, into: output)
                #expect(snapshot() == tail[step], "installed rollback full logits, step \(step)")
            }
            log("\(label) rejected-draft rollback: full HC bundle and eight full logit rows exact")
        }
        producer.reset()
        try await append(0, 33)
        #expect(snapshot() == heads[0], "clean reset prefix")
        try cancellationHook(producer, enabled: true)
        do {
            try await append(33, 49)
            Issue.record("expected cancellation after partially advanced GPU state")
        } catch is CancellationError { }
        #expect(throws: PrefillError.self) { try producer.prepareForContinuation(expectedPosition: 33) }
        do {
            try await producer.produce(token: tokens[33], position: 33, into: output)
            Issue.record("dirty state must reject decode")
        } catch let error as PrefillError {
            guard case .chunkedRunnerDirty = error else { throw error }
        }
        try cancellationHook(producer, enabled: false)
        producer.reset()
        try await append(0, 33)
        #expect(snapshot() == heads[0], "reset after cancellation")
        log("\(label) continuation, cancellation and exact reset complete")
        return Result(heads: heads, tail: tail, continuation: continuation)
    }

    @Test(arguments: ["gemma4", "qwen36", "maple", "inkling", "flashnext"])
    func memoryProfilesMatchAcrossBoundaries(name: String) async throws {
        let profile = Profile.named(name)
        guard let path = ProcessInfo.processInfo.environment[profile.gate] else { return }
        let resident = try await Self.runArm(path: path, profile: profile, resident: true)
        let streamed = try await Self.runArm(path: path, profile: profile, resident: false)
        #expect(resident.continuation == streamed.continuation, "eight greedy continuation tokens")
        for (index, pair) in zip(resident.heads + resident.tail, streamed.heads + streamed.tail).enumerated() {
            let mismatches = zip(pair.0, pair.1).filter { $0 != $1 }.count
            Self.log("\(name) row=\(index) memory-profile mismatches=\(mismatches)")
            #expect(mismatches == 0, "full logits across memory profiles, row \(index)")
        }
    }
}
