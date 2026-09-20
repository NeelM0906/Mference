import Foundation
import Metal
import Testing
@testable import Mference

/// Uses the existing installation read-only. An unset gate is not evidence
/// of installed-model support. Never stage or duplicate checkpoint weights.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_GTURBO"] != nil))
struct GemmaQATInstalledExecutionTests {
    private static func directory() throws -> URL {
        URL(fileURLWithPath: try #require(
            ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_GTURBO"]))
    }

    @Test func installedLayoutRejectsWrongKernelGeometry() throws {
        let directory = try Self.directory()
        let manifest = try ManifestReader.load(directoryURL: directory, expecting: .gemma4_26B_A4B)
        let index = try ResidentIndexReader.load(fileURL: directory.appendingPathComponent("model_weights.bin"))
        let layout = try PackedExpertsLayoutReader.load(directoryURL: directory)
        func validate(_ resident: ResidentIndex = index, _ experts: PackedExpertsLayout = layout) throws {
            try GemmaQATCheckpoint.validateRuntimeLayout(residentIndex: resident, layout: experts,
                                                        manifest: manifest, expected: .gemma4_26B_A4B)
        }
        try validate()
        #expect(index.entries.count == 597)
        #expect(layout.expertStride == 3_719_168)

        let embedding = try #require(index.entries["language_model.model.embed_tokens.weight"])
        let router = try #require(index.entries["language_model.model.layers.0.router.proj.weight"])
        func replaced(_ source: ResidentIndexEntry, dtype: UInt8? = nil,
                      offset: UInt64? = nil, size: UInt64? = nil,
                      scaleOffset: UInt64? = nil, scaleSize: UInt64? = nil) -> ResidentIndex {
            var entries = index.entries
            entries[source.name] = ResidentIndexEntry(name: source.name, dtype: dtype ?? source.dtype,
                fileOffset: offset ?? source.fileOffset, sizeBytes: size ?? source.sizeBytes,
                shape: source.shape, scaleOffset: scaleOffset ?? source.scaleOffset,
                scaleSize: scaleSize ?? source.scaleSize,
                biasOffset: source.biasOffset, biasSize: source.biasSize)
            return ResidentIndex(header: index.header, entries: entries)
        }
        // Mutations are in memory, never in the user's installed files.
        for corrupt in [
            replaced(embedding, scaleSize: embedding.scaleSize / 2), // group 64
            replaced(embedding, offset: index.header.indexSize - 2),
            replaced(embedding, offset: embedding.fileOffset + 1),
            replaced(embedding, offset: UInt64.max - 1),
            replaced(embedding, scaleOffset: embedding.fileOffset), // overlap
            replaced(router, dtype: 0),
            replaced(router, size: router.sizeBytes / 2),
            replaced(router, scaleOffset: router.fileOffset, scaleSize: 2),
        ] {
            #expect(throws: (any Error).self) { try validate(corrupt) }
        }

        let layer = layout.layers[0]
        let original = layer.experts[1]
        func changedExpert(_ role: String, _ replacement: SubTensorEntry,
                           uniform: Bool = true) -> PackedExpertsLayout {
            var tensors = original.subTensors
            tensors[role] = replacement
            var experts = layer.experts
            for id in (uniform ? Array(experts.indices) : [1]) {
                let prior = experts[id]
                experts[id] = ExpertEntry(expert: prior.expert, offset: prior.offset,
                                          size: prior.size, subTensors: tensors)
            }
            var layers = layout.layers
            layers[0] = LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
            return PackedExpertsLayout(expertStride: layout.expertStride, numLayers: layout.numLayers,
                                       expertsPerLayer: layout.expertsPerLayer, layers: layers)
        }
        let down = try #require(original.subTensors["down"])
        for corrupt in [
            SubTensorEntry(offset: down.offset + 2, size: down.size,
                           dtype: down.dtype, shape: down.shape, bits: down.bits),
            SubTensorEntry(offset: down.offset, size: down.size,
                           dtype: "BF16", shape: down.shape, bits: down.bits),
            SubTensorEntry(offset: down.offset, size: down.size,
                           dtype: down.dtype, shape: [704, 2816], bits: down.bits),
            SubTensorEntry(offset: down.offset, size: down.size,
                           dtype: down.dtype, shape: down.shape, bits: 8),
            SubTensorEntry(offset: UInt64.max - 1, size: down.size,
                           dtype: down.dtype, shape: down.shape, bits: down.bits),
        ] {
            #expect(throws: (any Error).self) { try validate(index, changedExpert("down", corrupt)) }
        }
        #expect(throws: (any Error).self) {
            try validate(index, changedExpert("down", SubTensorEntry(
                offset: down.offset + 4, size: down.size, dtype: down.dtype,
                shape: down.shape, bits: down.bits), uniform: false))
        }
    }

    @Test func installedModelSelectsNativeRuntimeFormat() throws {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.directory(), device: context.device,
                                   expecting: .gemma4_26B_A4B)
        #expect(model.modelID == CheckpointIdentity.gemma4QAT)
        #expect(model.affineInt4GroupSize == 32)
        #expect(model.hasBF16Router)
        #expect(model.sharedExpertWeightBits == 4)
        #expect(model.routedExpertWeightBits == 4)
        #expect(Quantization.groupSize == 64)
        let router = try model.router(layer: 0)
        #expect(router.dtype == 1)
        #expect(router.length == 128 * 2816 * 2)
        #expect(router.scaleLength == 0 && router.biasLength == 0)
        // Construct through the production factory; numerical execution is a
        // separate proof and is not implied by successful construction.
        _ = try ForwardRunnerFactory.make(model: model, context: context, maxContext: 256)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_DUMP"] != nil))
    func captureTeacherForcedNativeLogits() async throws {
        let directory = try Self.directory()
        let dump = URL(fileURLWithPath: try #require(
            ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_DUMP"]))
        try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   expecting: .gemma4_26B_A4B)
        // The raw source tokenizer is used directly for this numerical probe;
        // public CLI/chat capability remains gated until qualification passes.
        let tokenizer = try await MFTokenizer.load(from: directory.appendingPathComponent("tokenizer"),
                                                   family: .gemma4)
        let runtime = try ForwardRunnerFactory.make(model: model, context: context, maxContext: 256,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let runner = try #require(runtime.producer as? RealForwardRunner)
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        let longTokens = Array(tokenizer.encode(String(repeating: "The sky is blue. ", count: 40),
                                               addBOS: true).prefix(132))
        #expect(longTokens.count == 132)
        let corpus: [(String, [Int32])] = [
            ("capital", tokenizer.encode("The capital of France is Paris.", addBOS: true)),
            ("arithmetic", tokenizer.encode("2 + 2 = 4.", addBOS: true)),
            ("multi-chunk", longTokens),
        ]
        var items: [[String: Any]] = []
        func capture(_ bytes: inout Data) {
            bytes.append(logits.contents().assumingMemoryBound(to: UInt8.self), count: logits.length)
        }
        for (name, sequence) in corpus {
            runner.reset()
            var bytes = Data()
            var routes: [[[Int]]] = []
            for (position, token) in sequence.enumerated() {
                try await runner.produce(token: token, position: position, into: logits)
                let values = UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float16.self),
                                                 count: model.config.vocabSize)
                let finite = values.allSatisfy { $0.isFinite }
                #expect(finite)
                capture(&bytes)
                let selected = runner.lastDecodeRoutedExperts
                #expect(selected.count == 30 && selected.allSatisfy { $0.count == 8 })
                routes.append(selected)
            }
            let file = name + "-scalar.f16"
            try bytes.write(to: dump.appendingPathComponent(file))
            items.append(["name": name + "-scalar", "sequence": sequence,
                          "positions": Array(sequence.indices), "file": file, "routes": routes])
            print("[qat-reference] scalar \(name): \(sequence.count) positions captured")

            runner.reset()
            bytes = Data()
            let prefixCount = sequence.count - 2
            let prefill = try await runner.prefillChunked(tokens: sequence[..<prefixCount],
                startPosition: 0, outputMode: .logits, config: runtime.prefillConfig,
                into: logits, onProgress: { _ in })
            #expect(prefill.newPosition == prefixCount)
            let report = try #require(prefill.execution)
            if name == "multi-chunk" {
                #expect(report.batchedChunkSizes == [128, 2])
                #expect(report.replayedTokens == 0)
            }
            capture(&bytes)
            for position in prefixCount..<sequence.count {
                try await runner.produce(token: sequence[position], position: position, into: logits)
                capture(&bytes)
            }
            let chunkFile = name + "-chunked.f16"
            try bytes.write(to: dump.appendingPathComponent(chunkFile))
            let execution = try JSONSerialization.jsonObject(with: JSONEncoder().encode(report))
            items.append(["name": name + "-chunked", "sequence": sequence,
                          "positions": Array((prefixCount - 1)..<sequence.count),
                          "file": chunkFile, "prefill": execution])
            print("[qat-reference] chunked \(name): \(report)")
        }
        let manifest = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        let meta: [String: Any] = ["modelID": model.modelID,
            "manifest_sha256": Sha256Verifier.hashData(manifest),
            "logit_stage": "pre_softcap_fp16", "vocab": model.config.vocabSize,
            "items": items]
        try JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys, .prettyPrinted])
            .write(to: dump.appendingPathComponent("meta.json"))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_PREFILL_TRACE"] != nil))
    func traceChunkedNumericalDivergenceWithoutChangingLogits() async throws {
        let env = ProcessInfo.processInfo.environment
        let output = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_PREFILL_TRACE"]))
        try #require(!FileManager.default.fileExists(atPath: output.path), "Refusing to overwrite a trace")
        let baseline = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_BASELINE"]))
        let meta = try #require(JSONSerialization.jsonObject(with:
            Data(contentsOf: baseline.appendingPathComponent("meta.json"))) as? [String: Any])
        let items = try #require(meta["items"] as? [[String: Any]])
        let name = env["MFERENCE_GEMMA_QAT_TRACE_ITEM"] ?? "capital-chunked"
        try #require(name.hasSuffix("-chunked"))
        let item = try #require(items.first { $0["name"] as? String == name })
        let tokens = try #require(item["sequence"] as? [Int]).map(Int32.init)
        let prefixCount = tokens.count - 2
        try #require(prefixCount > 0)
        var observedPositions: Set<Int>?
        if let selected = env["MFERENCE_GEMMA_QAT_TRACE_POSITIONS"] {
            let positions = try selected.split(separator: ",").map { try #require(Int($0)) }
            try #require(!positions.isEmpty && positions.allSatisfy { $0 >= 0 && $0 < prefixCount })
            observedPositions = Set(positions)
        }
        let file = try #require(item["file"] as? String)
        let expected = try Data(contentsOf: baseline.appendingPathComponent(file))
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.directory(), device: context.device,
                                   expecting: .gemma4_26B_A4B)
        let runtime = try ForwardRunnerFactory.make(model: model, context: context, maxContext: 256,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let runner = try #require(runtime.producer as? RealForwardRunner)
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        var tensors = Data()
        var entries: [[String: Any]] = []
        runner.gemmaPrefillTrace = { position, layer, stage, values in
            if let observedPositions, !observedPositions.contains(position) { return }
            entries.append(["position": position, "layer": layer, "stage": stage,
                            "offset": tensors.count, "count": values.count])
            values.withUnsafeBytes { tensors.append(contentsOf: $0) }
        }
        var captured = Data()
        let prefill = try await runner.prefillChunked(tokens: tokens[..<prefixCount],
            startPosition: 0, outputMode: .logits, config: runtime.prefillConfig,
            into: logits, onProgress: { _ in })
        try #require(prefill.newPosition == prefixCount)
        let report = try #require(prefill.execution)
        #expect(report.replayedTokens == 0)
        let prior = try #require(item["prefill"] as? [String: Any])
        #expect(report.batchedChunkSizes == (prior["batchedChunkSizes"] as? [Int]))
        captured.append(logits.contents().assumingMemoryBound(to: UInt8.self), count: logits.length)
        for position in prefixCount..<tokens.count {
            try await runner.produce(token: tokens[position], position: position, into: logits)
            captured.append(logits.contents().assumingMemoryBound(to: UInt8.self), count: logits.length)
        }
        #expect(captured == expected, "Tracing must preserve the failing prefill and continuation logits byte-for-byte")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try tensors.write(to: output.appendingPathComponent("native.f32"))
        try JSONSerialization.data(withJSONObject: ["name": name,
            "sequence": Array(tokens.prefix(prefixCount)), "entries": entries], options: [.sortedKeys])
            .write(to: output.appendingPathComponent("trace.json"))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MFERENCE_GEMMA_QAT_TRACE"] != nil))
    func traceFirstNumericalDivergenceWithoutChangingLogits() async throws {
        let env = ProcessInfo.processInfo.environment
        let output = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_TRACE"]))
        let baseline = URL(fileURLWithPath: try #require(env["MFERENCE_GEMMA_QAT_BASELINE"]))
        let meta = try #require(JSONSerialization.jsonObject(with:
            Data(contentsOf: baseline.appendingPathComponent("meta.json"))) as? [String: Any])
        let items = try #require(meta["items"] as? [[String: Any]])
        let name = env["MFERENCE_GEMMA_QAT_TRACE_ITEM"] ?? "capital-scalar"
        try #require(name.hasSuffix("-scalar"), "This observer traces scalar execution")
        let item = try #require(items.first { $0["name"] as? String == name })
        let tokens = try #require(item["sequence"] as? [Int]).map(Int32.init)
        let limit = try #require(Int(env["MFERENCE_GEMMA_QAT_TRACE_LIMIT"] ?? String(tokens.count)))
        try #require(limit > 0 && limit <= tokens.count)
        let sequence = Array(tokens.prefix(limit))
        let file = try #require(item["file"] as? String)
        let vocabulary = try #require(meta["vocab"] as? Int)
        let baselineBytes = try Data(contentsOf: baseline.appendingPathComponent(file), options: .mappedIfSafe)
        try #require(baselineBytes.count == tokens.count * vocabulary * 2)
        let expected = Data(baselineBytes.prefix(limit * vocabulary * 2))
        let context = try MetalContext()
        let model = try Model.load(directoryURL: Self.directory(), device: context.device,
                                   expecting: .gemma4_26B_A4B)
        let runtime = try ForwardRunnerFactory.make(model: model, context: context, maxContext: 256,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let runner = try #require(runtime.producer as? RealForwardRunner)
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        var tensors = Data()
        var entries: [[String: Any]] = []
        runner.gemmaDecodeTrace = { position, layer, stage, values in
            entries.append(["position": position, "layer": layer, "stage": stage,
                            "offset": tensors.count, "count": values.count])
            values.withUnsafeBytes { tensors.append(contentsOf: $0) }
        }
        var captured = Data()
        for (position, token) in sequence.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
            captured.append(logits.contents().assumingMemoryBound(to: UInt8.self), count: logits.length)
        }
        #expect(captured == expected, "Tracing must preserve the original failing logits byte-for-byte")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try tensors.write(to: output.appendingPathComponent("native.f32"))
        try JSONSerialization.data(withJSONObject: ["name": name, "sequence": sequence, "entries": entries],
                                   options: [.sortedKeys])
            .write(to: output.appendingPathComponent("trace.json"))
    }
}
