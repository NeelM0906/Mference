import Foundation
@testable import Mference
@testable import MferenceRepackCore

/// Locates the committed `minicpm5` reference goldens and the toy checkpoint
/// they were captured from, and turns that checkpoint into a real `.gturbo`
/// install the runtime loader opens.
///
/// # Why this writes the install instead of driving `RemoteStreamingRepacker`
///
/// The fake remote lives in the `MferenceRepackTests` target, which this
/// target cannot import. What matters for parity is the *bytes*: every
/// projection goes through `Int4AffineEncoder.encodeTensor`, the whole-tensor
/// encoder the streaming installer is locked to by W2.1a, and the goldens'
/// gate set was captured from the same reconstruction. Everything downstream
/// is the real thing: `GTurboBinary`, `ManifestReader` validation (with the
/// family's quant contract), `Model.load` and the runner.
enum MiniCPM5Parity {

    static let repoRoot: URL = {
        var root = URL(fileURLWithPath: #filePath)
        // Tests/Mference/Core/Runtime/MiniCPM5/<this file> -> repo root.
        for _ in 0..<6 { root.deleteLastPathComponent() }
        return root
    }()

    static let fixturesDirectory = repoRoot
        .appendingPathComponent("Tests/Mference/Fixtures/minicpm5")
    static let checkpointURL = fixturesDirectory
        .appendingPathComponent("toy-ckpt/model.safetensors")

    /// The toy `ArchConfig` the goldens' `LlamaConfig` maps to.
    static func archConfig() -> ArchConfig { .miniCPM5Toy() }

    /// Cross-checks the toy config in `goldens-manifest.json` against
    /// `archConfig()`, so the two cannot drift apart silently.
    static func manifestConfigMatchesArch() throws -> [String] {
        let url = fixturesDirectory.appendingPathComponent("goldens-manifest.json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any],
              let config = root["config"] as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let a = archConfig()
        var mismatches: [String] = []
        func check(_ key: String, _ expected: Int) {
            if (config[key] as? Int) != expected {
                mismatches.append("\(key): manifest \(String(describing: config[key])) vs arch \(expected)")
            }
        }
        check("hidden_size", a.hiddenSize)
        check("intermediate_size", a.intermediateSize)
        check("num_hidden_layers", a.numLayers)
        check("num_attention_heads", a.numHeads)
        check("num_key_value_heads", a.numFullKVHeads)
        check("head_dim", a.fullHeadDim)
        check("vocab_size", a.vocabSize)
        // transformers 5.6.2 nests the RoPE base under `rope_parameters`.
        let ropeParameters = config["rope_parameters"] as? [String: Any]
        if (ropeParameters?["rope_theta"] as? Double) != a.fullRopeTheta {
            mismatches.append("rope_parameters.rope_theta")
        }
        if (config["tie_word_embeddings"] as? Bool) != a.tieWordEmbeddings {
            mismatches.append("tie_word_embeddings")
        }
        return mismatches
    }

    /// Writes a `.gturbo` install of the toy checkpoint: every rank-2
    /// projection INT4 affine group-64 through `Int4AffineEncoder`, norms
    /// BF16 verbatim, zero experts, the minicpm5 manifest.
    static func installToyCheckpoint() throws -> URL {
        let arch = archConfig()
        let source = try Safetensors(url: checkpointURL)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-minicpm5-parity-\(UUID().uuidString)")
        let expertsDir = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: expertsDir,
                                                withIntermediateDirectories: true)

        // Resident order: the family's own (embedding, layers, norm, head).
        let names = source.names.sorted {
            FlashNextPlanner.residentOrdering($0, $1, family: .minicpm5)
        }

        struct Placed {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let offset: UInt64
            let weightBytes: [UInt8]
            let scaleBytes: [UInt8]
            let biasBytes: [UInt8]
        }
        var stringTable: [UInt8] = []
        var nameOffsets: [UInt32] = []
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes = GTurboBinary.indexEntryBytes
        let stringTableBase = headerBytes + names.count * entryBytes
        for name in names {
            nameOffsets.append(UInt32(stringTableBase + stringTable.count))
            stringTable.append(contentsOf: name.utf8)
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)

        var placed: [Placed] = []
        var cursor = indexBytes
        for name in names {
            let entry = try source.entry(name)
            let quantizes = entry.dtype == "BF16" && entry.shape.count == 2
                && name.hasSuffix(".weight") && !name.contains("norm")
                && entry.shape[1] % Int4AffineEncoder.groupSize == 0
            if quantizes {
                let floats = try source.floats(name)
                let encoded = floats.withUnsafeBufferPointer {
                    Int4AffineEncoder.encodeTensor($0, rowLength: entry.shape[1])
                }
                placed.append(Placed(name: name, dtype: 0,
                                     shape: padTo4(entry.shape),
                                     offset: cursor,
                                     weightBytes: encoded.packed,
                                     scaleBytes: widen(encoded.scales),
                                     biasBytes: widen(encoded.biases)))
                cursor += UInt64(encoded.packed.count + 2 * encoded.scales.count * 2)
            } else {
                precondition(entry.dtype == "BF16", "\(name): unexpected dtype \(entry.dtype)")
                let bytes = source.rawBytes(entry)
                placed.append(Placed(name: name, dtype: 1,
                                     shape: padTo4(entry.shape),
                                     offset: cursor,
                                     weightBytes: bytes, scaleBytes: [], biasBytes: []))
                cursor += UInt64(bytes.count)
            }
        }
        let residentSize = cursor - indexBytes
        var file = [UInt8](repeating: 0, count: Int(indexBytes + residentSize))
        file.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(placed.count))
            for (i, p) in placed.enumerated() {
                let weightSize = UInt64(p.weightBytes.count)
                let scaleSize = UInt64(p.scaleBytes.count)
                let scaleOffset = scaleSize > 0 ? p.offset + weightSize : 0
                let biasOffset = scaleSize > 0 ? scaleOffset + scaleSize : 0
                let entry = ResidentEntry(
                    name: p.name, dtype: p.dtype, logicalShape4: p.shape,
                    fileOffset: p.offset, sizeBytes: weightSize,
                    scaleOffset: scaleOffset, scaleSize: scaleSize,
                    biasOffset: biasOffset, biasSize: scaleSize,
                    quantSpec: nil,
                    sourceWeight: ModelLoaderTests.dummySource(p.name),
                    sourceScales: nil, sourceBiases: nil)
                GTurboBinary.writeIndexEntry(
                    into: base.advanced(by: headerBytes + i * entryBytes),
                    entry: entry, nameOffset: nameOffsets[i])
            }
            _ = stringTable.withUnsafeBytes {
                memcpy(base.advanced(by: stringTableBase), $0.baseAddress!, stringTable.count)
            }
            for p in placed {
                var at = Int(p.offset)
                for chunk in [p.weightBytes, p.scaleBytes, p.biasBytes] where !chunk.isEmpty {
                    _ = chunk.withUnsafeBytes {
                        memcpy(base.advanced(by: at), $0.baseAddress!, chunk.count)
                    }
                    at += chunk.count
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(file).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        var layersArr: [[String: Any]] = []
        for L in 0..<arch.numLayers {
            layersArr.append([
                "layer": L,
                "file": String(format: "layer_%02d.bin", L),
                "experts": [[String: Any]](),
            ])
        }
        let layoutData = try JSONSerialization.data(
            withJSONObject: ["expertStride": 16_384, "numLayers": arch.numLayers,
                             "expertsPerLayer": 0, "layers": layersArr],
            options: [.sortedKeys])
        let layoutURL = expertsDir.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        let affine: [String: Any] = ["weightBits": 4, "scheme": "affine",
                                     "scaleType": "BF16", "biasType": "BF16",
                                     "groupSize": Int4AffineEncoder.groupSize]
        let absent: [String: Any] = ["weightBits": 0, "scheme": "none",
                                     "scaleType": "none", "biasType": "none",
                                     "groupSize": 0]
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "minicpm5-parity-toy",
            "arch": archDictionary(arch),
            "quant": ["embedding": affine, "attention": affine,
                      "router": absent, "sharedExpert": absent, "routedExpert": absent],
            "files": [
                "model_weights.bin": ["size": file.count, "sha256": weightsSha],
                "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
            ],
            "expertsPerLayer": 0,
            "numLayers": arch.numLayers,
            "expertStride": 16_384,
        ]
        try JSONSerialization
            .data(withJSONObject: manifest, options: [.sortedKeys, .withoutEscapingSlashes])
            .write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    static func archDictionary(_ a: ArchConfig) -> [String: Any] {
        [
            "hiddenSize": a.hiddenSize, "ffnIntermediate": a.intermediateSize,
            "moeIntermediateSize": a.moeIntermediateSize,
            "numHeads": a.numHeads, "numKVHeads": a.numKVHeads,
            "numFullKVHeads": a.numFullKVHeads,
            "headDim": a.headDim, "fullHeadDim": a.fullHeadDim,
            "vocabSize": a.vocabSize, "slidingWindow": a.slidingWindow,
            "finalLogitSoftcap": a.finalLogitSoftcap,
            "ropeTheta": a.ropeTheta, "fullRopeTheta": a.fullRopeTheta,
            "partialRotaryFactor": a.partialRotaryFactor,
            "numLayers": a.numLayers, "numExperts": a.numExperts,
            "topKExperts": a.topKExperts,
            "tieWordEmbeddings": a.tieWordEmbeddings,
            "attentionKEqV": a.attentionKEqV,
            "hiddenActivation": a.hiddenActivation,
            "fullAttentionLayerMask": a.fullAttentionLayerMask.map { Int($0) },
            "family": a.family.rawValue,
            "attnOutputGate": a.attnOutputGate,
            "attentionScale": a.attentionScale,
            "embeddingScaledBySqrtHidden": a.embeddingScaledBySqrtHidden,
            "routerScaled": a.routerScaled,
            "ffnSandwichNorms": a.ffnSandwichNorms,
            "sharedExpertGated": a.sharedExpertGated,
            "ropeNeoxSubdim": a.ropeNeoxSubdim,
            "linearNumKHeads": 0, "linearNumVHeads": 0,
            "linearKeyHeadDim": 0, "linearValueHeadDim": 0, "linearConvKernelSize": 0,
            "numSharedExperts": a.numSharedExperts,
            "numDenseLayers": a.numDenseLayers,
            "denseIntermediateSize": a.denseIntermediateSize,
            "qkNorm": a.qkNorm,
        ]
    }

    private static func widen(_ companions: [UInt16]) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(companions.count * 2)
        for value in companions {
            out.append(UInt8(truncatingIfNeeded: value))
            out.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        return out
    }

    private static func padTo4(_ shape: [Int]) -> [UInt32] {
        var out = shape.map { UInt32($0) }
        while out.count < 4 { out.append(0) }
        return out
    }
}

/// Reader for the committed `minicpm5` goldens (see `Scripts/parity/README.md`).
struct MiniCPM5Goldens {
    enum Prompt: String, CaseIterable { case short, long }
    enum Phase: String { case prefill, decode }

    let tensors: [String: [Float]]
    let shapes: [String: [Int]]
    let integers: [String: Any]

    init(prompt: Prompt, phase: Phase) throws {
        let dir = MiniCPM5Parity.fixturesDirectory
        let safetensors = try Safetensors(
            url: dir.appendingPathComponent("\(phase.rawValue)_\(prompt.rawValue).safetensors"))
        var tensors: [String: [Float]] = [:]
        var shapes: [String: [Int]] = [:]
        for name in safetensors.names {
            tensors[name] = try safetensors.floats(name)
            shapes[name] = try safetensors.entry(name).shape
        }
        self.tensors = tensors
        self.shapes = shapes
        let integersURL = dir.appendingPathComponent(
            "integers_\(phase.rawValue)_\(prompt.rawValue).json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: integersURL))
                as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.integers = root
    }

    static func promptTokens(_ prompt: Prompt) throws -> [Int32] {
        let url = MiniCPM5Parity.fixturesDirectory.appendingPathComponent("goldens-manifest.json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any],
              let prompts = root["prompts"] as? [String: Any],
              let entry = prompts[prompt.rawValue] as? [String: Any],
              let ids = entry["ids"] as? [Int] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return ids.map { Int32($0) }
    }

    /// Row `r` of a rank-2 tensor.
    func row(_ name: String, _ r: Int) throws -> [Float] {
        guard let values = tensors[name], let shape = shapes[name], shape.count == 2 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let width = shape[1]
        return Array(values[(r * width)..<((r + 1) * width)])
    }

    func ints(_ key: String) throws -> [Int] {
        guard let values = integers[key] as? [Int] else { throw CocoaError(.fileReadCorruptFile) }
        return values
    }

    func floats(_ key: String) throws -> [Float] {
        guard let values = integers[key] as? [Double] else { throw CocoaError(.fileReadCorruptFile) }
        return values.map { Float($0) }
    }
}
