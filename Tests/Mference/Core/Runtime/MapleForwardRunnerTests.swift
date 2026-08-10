import Foundation
import Metal
import Testing
@testable import Mference

@Suite struct MapleForwardRunnerTests {
    private static let hidden = 2_048
    private static let vocab = 151_936
    private static let layers = 24
    private static let experts = 256
    private static let intermediate = 512
    private static let expertStride: UInt64 = 983_040

    private struct Fixture {
        let directory: URL
        let context: MetalContext
        let residentBuffer: ResidentBuffer
        let residentIndex: ResidentIndex
        let layout: PackedExpertsLayout
        let manifest: Manifest

        func model(residentIndex: ResidentIndex? = nil,
                   layout: PackedExpertsLayout? = nil) -> Model {
            Model(device: context.device,
                  config: .maplePreview,
                  streamingMode: .pread(slotCount: 8),
                  expertCachePolicy: .lfu,
                  integrityPolicy: .sizeCheckTrustedReceipt,
                  residentBuffer: residentBuffer,
                  residentIndex: residentIndex ?? self.residentIndex,
                  packedExpertsLayout: layout ?? self.layout,
                  manifest: manifest,
                  directoryURL: directory)
        }
    }

    private func makeFixture(includeFlashHead: Bool = false) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("maple-runtime-\(UUID().uuidString)")
        let expertsDirectory = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: expertsDirectory,
                                                withIntermediateDirectories: true)

        let affineWeights = UInt64(Self.vocab * Self.hidden / 2)
        let affineCompanions = UInt64(Self.vocab * (Self.hidden / 64) * 2)
        let residentSize = affineWeights + 2 * affineCompanions
        let residentURL = directory.appendingPathComponent("model_weights.bin")
        FileManager.default.createFile(atPath: residentURL.path, contents: nil)
        let residentHandle = try FileHandle(forWritingTo: residentURL)
        try residentHandle.truncate(atOffset: residentSize)
        try residentHandle.close()

        func affine(_ name: String, rows: Int) -> ResidentIndexEntry {
            let weights = UInt64(rows * Self.hidden / 2)
            let companions = UInt64(rows * (Self.hidden / 64) * 2)
            return ResidentIndexEntry(name: name, dtype: 0, fileOffset: 0,
                                      sizeBytes: weights,
                                      shape: (UInt32(rows), UInt32(Self.hidden), 0, 0),
                                      scaleOffset: weights, scaleSize: companions,
                                      biasOffset: weights + companions, biasSize: companions)
        }
        func norm(_ name: String, rows: Int = Self.hidden) -> ResidentIndexEntry {
            ResidentIndexEntry(name: name, dtype: 1, fileOffset: 0,
                               sizeBytes: UInt64(rows * 2),
                               shape: (UInt32(rows), 0, 0, 0),
                               scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
        }
        func router(_ name: String) -> ResidentIndexEntry {
            ResidentIndexEntry(name: name, dtype: 1, fileOffset: 0,
                               sizeBytes: UInt64(Self.experts * Self.hidden * 2),
                               shape: (UInt32(Self.experts), UInt32(Self.hidden), 0, 0),
                               scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
        }

        var entries: [String: ResidentIndexEntry] = [:]
        for entry in [affine("model.word_embeddings.weight", rows: Self.vocab),
                      norm("model.norm.weight"),
                      affine("lm_head.weight", rows: Self.vocab)] {
            entries[entry.name] = entry
        }
        if includeFlashHead {
            let map = ResidentIndexEntry(
                name: "lm_head_flash.token_map", dtype: 5, fileOffset: 0,
                sizeBytes: UInt64(Self.vocab * MemoryLayout<Int32>.stride),
                shape: (4_748, 32, 0, 0),
                scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
            let centroids = affine("lm_head_flash.centroids.weight", rows: 4_748)
            entries[map.name] = map
            entries[centroids.name] = centroids
        }
        for layer in 0..<Self.layers {
            let prefix = "model.layers.\(layer)"
            for entry in [norm("\(prefix).input_layernorm.weight"),
                          norm("\(prefix).post_attention_layernorm.weight"),
                          affine("\(prefix).self_attn.q_proj.weight", rows: Self.hidden),
                          affine("\(prefix).self_attn.k_proj.weight", rows: 512),
                          affine("\(prefix).self_attn.v_proj.weight", rows: 512),
                          affine("\(prefix).self_attn.o_proj.weight", rows: Self.hidden),
                          norm("\(prefix).self_attn.q_norm.weight", rows: 128),
                          norm("\(prefix).self_attn.k_norm.weight", rows: 128),
                          router("\(prefix).mlp.gate.weight")] {
                entries[entry.name] = entry
            }
        }

        let raw = UInt64(Self.intermediate * Self.hidden / 4)
        let companion = UInt64(Self.intermediate * (Self.hidden / 64) * 2)
        let slices: [(String, UInt64, UInt64)] = [
            ("gate", 0, raw),
            ("gate_scales", raw, companion),
            ("gate_biases", raw + companion, companion),
            ("up", raw + 2 * companion, raw),
            ("up_scales", 2 * raw + 2 * companion, companion),
            ("up_biases", 2 * raw + 3 * companion, companion),
            ("down", 2 * (raw + 2 * companion), raw),
            ("down_scales", 3 * raw + 4 * companion, companion),
            ("down_biases", 3 * raw + 5 * companion, companion),
        ]
        let tensors = Dictionary(uniqueKeysWithValues: slices.map {
            ($0.0, SubTensorEntry(offset: $0.1, size: $0.2))
        })
        let layerSize = Self.expertStride * UInt64(Self.experts)
        var layoutLayers: [LayerLayout] = []
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": residentSize, "sha256": "0"],
        ]
        let layoutURL = expertsDirectory.appendingPathComponent("layout.json")
        let layoutData = Data("{}".utf8)
        try layoutData.write(to: layoutURL)
        files["packed_experts/layout.json"] = ["size": UInt64(layoutData.count), "sha256": "0"]
        for layer in 0..<Self.layers {
            let file = String(format: "layer_%02d.bin", layer)
            let url = expertsDirectory.appendingPathComponent(file)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            try handle.truncate(atOffset: layerSize)
            try handle.close()
            files["packed_experts/\(file)"] = ["size": layerSize, "sha256": "0"]
            let expertEntries = (0..<Self.experts).map {
                ExpertEntry(expert: $0, offset: UInt64($0) * Self.expertStride,
                            size: Self.expertStride, subTensors: tensors)
            }
            layoutLayers.append(LayerLayout(layer: layer, file: file, experts: expertEntries))
        }

        let config = ArchConfig.maplePreview
        let arch: [String: Any] = [
            "hiddenSize": config.hiddenSize,
            "ffnIntermediate": config.intermediateSize,
            "moeIntermediateSize": config.moeIntermediateSize,
            "numHeads": config.numHeads,
            "numKVHeads": config.numKVHeads,
            "numFullKVHeads": config.numFullKVHeads,
            "headDim": config.headDim,
            "fullHeadDim": config.fullHeadDim,
            "vocabSize": config.vocabSize,
            "slidingWindow": config.slidingWindow,
            "finalLogitSoftcap": config.finalLogitSoftcap,
            "ropeTheta": config.ropeTheta,
            "fullRopeTheta": config.fullRopeTheta,
            "partialRotaryFactor": config.partialRotaryFactor,
            "numLayers": config.numLayers,
            "numExperts": config.numExperts,
            "topKExperts": config.topKExperts,
            "tieWordEmbeddings": config.tieWordEmbeddings,
            "attentionKEqV": config.attentionKEqV,
            "hiddenActivation": config.hiddenActivation,
            "fullAttentionLayerMask": config.fullAttentionLayerMask.map(Int.init),
        ]
        func quant(bits: Int, scheme: String, scale: String, bias: String, group: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": scheme, "scaleType": scale,
             "biasType": bias, "groupSize": group]
        }
        var manifestObject: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": [:], "modelID": "maple-runtime-test", "arch": arch,
            "quant": [
                "embedding": quant(bits: 4, scheme: "affine", scale: "bf16", bias: "bf16", group: 64),
                "attention": quant(bits: 4, scheme: "affine", scale: "bf16", bias: "bf16", group: 64),
                "router": quant(bits: 16, scheme: "unquantized", scale: "none", bias: "none", group: 0),
                "sharedExpert": quant(bits: 0, scheme: "none", scale: "none", bias: "none", group: 0),
                "routedExpert": quant(bits: 2, scheme: "affine", scale: "bf16", bias: "bf16", group: 64),
            ],
            "files": files, "expertsPerLayer": Self.experts,
            "numLayers": Self.layers, "expertStride": Self.expertStride,
        ]
        if includeFlashHead {
            manifestObject["flashHead"] = [
                "nClusters": 4_748, "clusterSize": 32, "nProbes": 512,
                "groupSize": 64, "bits": 4, "headGroupSize": 64,
                "headBits": 4, "scaledCentroids": true,
                "forceTokens": [151_645, 151_668, 151_643],
            ]
        }
        let manifest = try JSONDecoder().decode(
            Manifest.self,
            from: JSONSerialization.data(withJSONObject: manifestObject, options: [.sortedKeys]))
        let context = try MetalContext()
        return Fixture(
            directory: directory,
            context: context,
            residentBuffer: try ResidentBuffer(fileURL: residentURL, fileOffset: 0,
                                                residentSize: residentSize, device: context.device),
            residentIndex: ResidentIndex(header: ResidentIndexHeader(indexSize: 0,
                                                                       residentSize: residentSize,
                                                                       entryCount: UInt64(entries.count)),
                                         entries: entries),
            layout: PackedExpertsLayout(expertStride: Self.expertStride,
                                        numLayers: Self.layers,
                                        expertsPerLayer: Self.experts,
                                        layers: layoutLayers),
            manifest: manifest)
    }

    private func makeLogits(_ context: MetalContext) throws -> MTLBuffer {
        guard let logits = context.device.makeBuffer(
            length: Self.vocab * MemoryLayout<UInt16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return logits
    }

    private func fillSentinel(_ logits: MTLBuffer) {
        let values = logits.contents().bindMemory(to: UInt16.self, capacity: Self.vocab)
        for index in 0..<Self.vocab { values[index] = 0x7BFF }
    }

    private func bits(_ logits: MTLBuffer) -> [UInt16] {
        Array(UnsafeBufferPointer(start: logits.contents().bindMemory(to: UInt16.self,
                                                                       capacity: Self.vocab),
                                  count: Self.vocab))
    }

    @Test func mapleFactory_replaysExactlyThroughMixedCacheRefills() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let requested = RuntimeConfiguration(prefillEnabled: true, forceLogitsHead: false)
        let model = fixture.model()
        let runtime = try ForwardRunnerFactory.make(model: model, context: fixture.context,
                                                    maxContext: 4, runtimeConfiguration: requested)
        let runner = try #require(runtime.producer as? MapleForwardRunner)
        #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)
        #expect(model.routedExpertCacheSlotCount(layer: 0) == 8)
        #expect(runtime.prefillConfig == requested.prefillConfig)
        #expect(runtime.executedPrefillMode == .chunked)
        #expect(runtime.kvStorageMode == .bf16)
        #expect(runtime.producer is any ChunkedPrefillRunner)
        #expect(!(runtime.producer is any FusedHeadLogitProducer))

        let logits = try makeLogits(fixture.context)
        fillSentinel(logits)
        #expect(model.openLayerFileCount() == 0)
        try await runner.produce(token: 0, position: 0, into: logits)
        let first = bits(logits)
        #expect(first.allSatisfy { $0 == 0 })
        #expect(runner.continuationPosition == 1)
        #expect(model.openLayerFileCount() == Self.layers)
        for layer in 0..<Self.layers {
            #expect(try model.planRoutedExperts(layer: layer, experts: Array(0..<8))?.misses.isEmpty == true)
        }

        let displacedExperts = Array(0..<4) + Array(8..<12)
        for layer in 0..<Self.layers {
            let planned = try model.planRoutedExperts(layer: layer, experts: displacedExperts)
            let plan = try #require(planned)
            #expect(plan.hits == 4)
            #expect(plan.misses == Array(4..<8))
            _ = try await model.fetchRoutedExperts(plan: plan)
        }

        try runner.prepareForContinuation(expectedPosition: 1)
        #expect(throws: PrefillError.self) {
            try runner.prepareForContinuation(expectedPosition: 2)
        }
        try await runner.produce(token: 0, position: 1, into: logits)
        #expect(bits(logits) == first)
        #expect(runner.continuationPosition == 2)
        for layer in 0..<Self.layers {
            #expect(try model.planRoutedExperts(layer: layer, experts: Array(0..<8))?.misses.isEmpty == true)
        }

        runner.reset()
        #expect(runner.continuationPosition == 0)
        fillSentinel(logits)
        try await runner.produce(token: 0, position: 0, into: logits)
        #expect(bits(logits) == first)
    }

    @Test func mapleHeadlessPrefillAdvancesExpertsWithoutWritingLogits() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let model = fixture.model()
        let runtime = try ForwardRunnerFactory.make(
            model: model, context: fixture.context, maxContext: 4,
            runtimeConfiguration: RuntimeConfiguration(prefillEnabled: false, forceLogitsHead: true))
        let runner = try #require(runtime.producer as? MapleForwardRunner)
        let headless: any HeadlessSequentialPrefillRunner = runner
        let logits = try makeLogits(fixture.context)
        fillSentinel(logits)

        try await headless.produceWithoutLogits(token: 0, position: 0)

        #expect(bits(logits).allSatisfy { $0 == 0x7BFF })
        #expect(runner.continuationPosition == 1)
        #expect(model.openLayerFileCount() == Self.layers)
        for layer in 0..<Self.layers {
            #expect(try model.planRoutedExperts(layer: layer,
                                                 experts: Array(0..<8))?.misses.isEmpty == true)
        }

        try await runner.produce(token: 0, position: 1, into: logits)

        #expect(bits(logits).allSatisfy { $0 == 0 })
        #expect(runner.continuationPosition == 2)
    }

    @Test func mapleFactory_keepsDisabledPrefillExactWithFlashHead() async throws {
        let fixture = try makeFixture(includeFlashHead: true)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let runtime = try ForwardRunnerFactory.make(
            model: fixture.model(), context: fixture.context, maxContext: 4,
            runtimeConfiguration: RuntimeConfiguration(prefillEnabled: false,
                                                       useMapleFlashHead: true))
        let runner = try #require(runtime.producer as? MapleForwardRunner)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.executedPrefillMode == .off)
        #expect(runtime.kvStorageMode == .bf16)

        let logits = try makeLogits(fixture.context)
        try await runner.produce(token: 0, position: 0, into: logits)
        let negativeInfinity = Float16(-Float.infinity).bitPattern
        #expect(bits(logits)[1] == negativeInfinity)

        runner.reset()
        let prefill: any ExactPrefillLogitProducer = runner
        try await prefill.produceExactPrefill(token: 0, position: 0, into: logits)
        #expect(bits(logits).allSatisfy { $0 == 0 })
    }

    @Test func mapleChunkedPrefill_matchesSequentialLogitsAndContinuation() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let sequential = try MapleForwardRunner(model: fixture.model(), context: fixture.context,
                                                 maxContext: 4)
        let chunked = try MapleForwardRunner(model: fixture.model(), context: fixture.context,
                                              maxContext: 4)
        let sequentialLogits = try makeLogits(fixture.context)
        let chunkedLogits = try makeLogits(fixture.context)

        try await sequential.produce(token: 0, position: 0, into: sequentialLogits)
        try await sequential.produce(token: 0, position: 1, into: sequentialLogits)
        let expected = bits(sequentialLogits)

        let prefill: any ChunkedPrefillRunner = chunked
        var progress: [Int] = []
        let result = try await prefill.prefillChunked(
            tokens: [0, 0][...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: chunkedLogits,
            onProgress: { progress.append($0) })

        #expect(result == PrefillResult(newPosition: 2, seed: .logitsWritten))
        #expect(progress == [2])
        #expect(bits(chunkedLogits) == expected)
        #expect(chunked.continuationPosition == 2)

        try await sequential.produce(token: 0, position: 2, into: sequentialLogits)
        try await chunked.produce(token: 0, position: 2, into: chunkedLogits)
        #expect(bits(chunkedLogits) == bits(sequentialLogits))
        #expect(chunked.continuationPosition == 3)
    }

    @Test func mapleFlashHead_isOptInAndPrefillRemainsExact() async throws {
        let fixture = try makeFixture(includeFlashHead: true)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let flashRuntime = try ForwardRunnerFactory.make(
            model: fixture.model(), context: fixture.context, maxContext: 4,
            runtimeConfiguration: RuntimeConfiguration(useMapleFlashHead: true))
        let runner = try #require(flashRuntime.producer as? MapleForwardRunner)
        let logits = try makeLogits(fixture.context)

        try await runner.produce(token: 0, position: 0, into: logits)
        let selected: Set<Int> = [0, 151_645, 151_668, 151_643]
        let negativeInfinity = Float16(-Float.infinity).bitPattern
        #expect(bits(logits).enumerated().allSatisfy { index, value in
            selected.contains(index) ? value == 0 : value == negativeInfinity
        })

        runner.reset()
        let chunked: any ChunkedPrefillRunner = runner
        _ = try await chunked.prefillChunked(
            tokens: [0][...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
        let chunkedPrefillBits = bits(logits)
        #expect(chunkedPrefillBits.allSatisfy { $0 == 0 })

        runner.reset()
        let sequentialPrefill: any ExactPrefillLogitProducer = runner
        try await sequentialPrefill.produceExactPrefill(token: 0, position: 0, into: logits)
        #expect(bits(logits) == chunkedPrefillBits)

        let exactRuntime = try ForwardRunnerFactory.make(
            model: fixture.model(), context: fixture.context, maxContext: 4,
            runtimeConfiguration: RuntimeConfiguration(useMapleFlashHead: false))
        let exactRunner = try #require(exactRuntime.producer as? MapleForwardRunner)
        try await exactRunner.produce(token: 0, position: 0, into: logits)
        #expect(bits(logits).allSatisfy { $0 == 0 })

        let unavailable = try makeFixture()
        defer { try? FileManager.default.removeItem(at: unavailable.directory) }
        let fallbackRuntime = try ForwardRunnerFactory.make(
            model: unavailable.model(), context: unavailable.context, maxContext: 4,
            runtimeConfiguration: RuntimeConfiguration(useMapleFlashHead: true))
        let fallback = try #require(fallbackRuntime.producer as? MapleForwardRunner)
        let fallbackLogits = try makeLogits(unavailable.context)
        try await fallback.produce(token: 0, position: 0, into: fallbackLogits)
        #expect(bits(fallbackLogits).allSatisfy { $0 == 0 })
    }

    @Test func mapleInit_rejectsMalformedResidentAndExpertLayoutMetadata() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        var entries = fixture.residentIndex.entries
        let qName = "model.layers.0.self_attn.q_proj.weight"
        let q = try #require(entries[qName])
        entries[qName] = ResidentIndexEntry(name: q.name, dtype: q.dtype,
                                             fileOffset: q.fileOffset, sizeBytes: q.sizeBytes,
                                             shape: (q.shape.0 - 1, q.shape.1, 0, 0),
                                             scaleOffset: q.scaleOffset, scaleSize: q.scaleSize,
                                             biasOffset: q.biasOffset, biasSize: q.biasSize)
        let badResident = fixture.model(residentIndex: ResidentIndex(header: fixture.residentIndex.header,
                                                                       entries: entries))
        #expect(throws: MapleForwardRunnerError.self) {
            _ = try MapleForwardRunner(model: badResident, context: fixture.context, maxContext: 4)
        }

        let underflow = fixture.model(residentIndex: ResidentIndex(
            header: ResidentIndexHeader(indexSize: 1,
                                        residentSize: fixture.residentIndex.header.residentSize,
                                        entryCount: fixture.residentIndex.header.entryCount),
            entries: fixture.residentIndex.entries))
        #expect(throws: ModelError.self) {
            _ = try underflow.resident(name: "model.word_embeddings.weight")
        }

        var slicedLayers = fixture.layout.layers
        var experts = slicedLayers[0].experts
        let nonzeroExpert = 17
        let original = try #require(experts[nonzeroExpert].subTensors["gate"])
        var slices = experts[nonzeroExpert].subTensors
        slices["gate"] = SubTensorEntry(offset: original.offset + 1, size: original.size)
        experts[nonzeroExpert] = ExpertEntry(expert: nonzeroExpert,
                                             offset: experts[nonzeroExpert].offset,
                                             size: experts[nonzeroExpert].size,
                                             subTensors: slices)
        slicedLayers[0] = LayerLayout(layer: 0, file: slicedLayers[0].file, experts: experts)
        let badSlice = fixture.model(layout: PackedExpertsLayout(expertStride: fixture.layout.expertStride,
                                                                   numLayers: fixture.layout.numLayers,
                                                                   expertsPerLayer: fixture.layout.expertsPerLayer,
                                                                   layers: slicedLayers))
        #expect(throws: MapleForwardRunnerError.self) {
            _ = try MapleForwardRunner(model: badSlice, context: fixture.context, maxContext: 4)
        }

        var layers = fixture.layout.layers
        layers[0] = LayerLayout(layer: 0, file: "layer_0.bin", experts: layers[0].experts)
        let badLayout = fixture.model(layout: PackedExpertsLayout(expertStride: fixture.layout.expertStride,
                                                                    numLayers: fixture.layout.numLayers,
                                                                    expertsPerLayer: fixture.layout.expertsPerLayer,
                                                                    layers: layers))
        #expect(throws: MapleForwardRunnerError.self) {
            _ = try MapleForwardRunner(model: badLayout, context: fixture.context, maxContext: 4)
        }
    }

    @Test func qwenFactory_preservesChunkedFusedRunnerSelection() throws {
        let directory = try QwenToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   expecting: .qwen36Toy())
        let requested = RuntimeConfiguration(prefillEnabled: true, forceLogitsHead: false)
        let runtime = try ForwardRunnerFactory.make(model: model, context: context,
                                                    maxContext: 64, runtimeConfiguration: requested)
        #expect(runtime.producer is RealForwardRunner)
        #expect(runtime.producer is any ChunkedPrefillRunner)
        #expect(runtime.prefillConfig == requested.prefillConfig)
        #expect(runtime.executedPrefillMode == .chunked)
        #expect(runtime.kvStorageMode == .fp16)
        #expect((runtime.producer as? any FusedHeadLogitProducer)?.usesFusedGreedyHead == true)
    }
}
