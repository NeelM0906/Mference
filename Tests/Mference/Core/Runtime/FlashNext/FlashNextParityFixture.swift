import Foundation
@testable import Mference
@testable import MferenceRepackCore

/// Locates the committed `qwen4_exp` parity goldens and the regenerable toy
/// checkpoint they were captured from, and turns that checkpoint into a real
/// `.gturbo` install the runtime loader can open.
///
/// # Why this writes the install instead of driving `RemoteStreamingRepacker`
///
/// The production installer path (`ArchInfo.load` -> `FlashNextPlanner.plan` ->
/// `RemoteStreamingRepacker`) **cannot ingest this checkpoint**. Three
/// independent refusals, all verified by
/// `FlashNextToyRepackBlockerTests`:
///
///   1. `scratch/qwen4exp-toy-ckpt-prodlayout/config.json` is the *text-only*
///      config the harness saved: `model_type` is `qwen4_exp_text` and there is
///      no `text_config` wrapper. `ArchInfo.load` dispatches on
///      `model_type == "qwen4_exp"` and then requires `text_config`.
///   2. Its `layer_types` carry `"qwen_sparse_attention"`, the value
///      `Qwen4ExpTextConfig.__post_init__` *rewrites* `"full_attention"` into
///      (ref `configuration_qwen4_exp.py:180-183`). The real vendor config on
///      disk says `"full_attention"`, which is exactly what `ArchInfo` accepts —
///      so this is an artifact of the harness saving the post-init config, not
///      a production gap.
///   3. `moe_intermediate_size` is **32**. `FlashNextPlanner.planLayerFile`
///      requires the expert intermediate dim to be a multiple of the INT4
///      group size (64) because every routed expert is quantized in flight.
///      A 32-wide toy expert cannot be group-64 quantized at all.
///
/// (1) and (2) are cosmetic and are repaired here (`productionShapedConfig`).
/// (3) is structural: it would need either a bigger toy (regenerating every
/// committed golden) or a no-quantize mode in the planner (a production
/// feature). Neither is in this change's scope.
///
/// # Why the install is BF16 passthrough
///
/// The goldens were captured from the **bf16 checkpoint upcast to fp32**. An
/// INT4 affine group-64 install reconstructs a weight to within roughly
/// `range/30` of its source — three to four orders of magnitude coarser than
/// the manifest's `atol = rtol = 1e-4` gate. Comparing an INT4 forward against
/// fp32 goldens would measure the quantizer, not the port.
///
/// The repacker has no no-quantize mode, so this fixture writes the same
/// `.gturbo` byte contract with every tensor left at its **source BF16**
/// dtype. Everything downstream is the real thing: the real resident index
/// encoder (`GTurboBinary`), the real `packed_experts/layout.json` shape, the
/// real `ple/` row-pool geometry (`PleRowPoolPlan`'s rules, at toy width), the
/// real `ManifestReader` validation, the real `Model.load` and the real
/// `PleRowPool` reader.
///
/// The quantization loss is not swept under the rug: `FlashNextWeights` can
/// round-trip every tensor the planner *would* quantize through
/// `Int4AffineEncoder` + `Quantization.dequantizeInt4Affine`, and
/// `FlashNextReferenceParityTests.int4RoundTripQuantifiesTheDequantCaveat`
/// reports the resulting logit spread.
enum FlashNextParity {

    // MARK: - Locations

    /// Package root, found from this file's own path.
    static let repoRoot: URL = {
        var root = URL(fileURLWithPath: #filePath)
        // .../Tests/Mference/Core/Runtime/FlashNext/<this file>
        for _ in 0..<6 { root.deleteLastPathComponent() }
        return root
    }()

    /// The production-name/production-layout copy of the toy checkpoint the
    /// golden harness emits. Not committed; regenerate with
    /// `./scratch/qwen4exp-parity-venv/bin/python
    ///  Scripts/parity/qwen4exp_make_goldens.py --emit-checkpoint`.
    static var checkpointDirectory: URL {
        repoRoot.appendingPathComponent("scratch/qwen4exp-toy-ckpt-prodlayout")
    }

    static var checkpointIsPresent: Bool {
        FileManager.default.fileExists(
            atPath: checkpointDirectory.appendingPathComponent("model.safetensors").path)
    }

    static let trunk = "model.language_model."

    // MARK: - Arch baseline

    /// The runtime `ArchConfig` for the goldens' toy config, transcribed from
    /// `Tests/Mference/Fixtures/qwen4exp/goldens-manifest.json -> config`.
    ///
    /// `layer_types` `[lin, lin, lin, qsa, lin, qsa]` maps to the runtime mask
    /// `[2, 2, 2, 1, 2, 1]` (2 = linear attention, 1 = full attention), the same
    /// mapping `ArchInfo.loadQwen38FlashNext` performs.
    static func archConfig() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 32,              // shared_expert_intermediate_size
            moeIntermediateSize: 32,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 16,
            fullHeadDim: 16,
            vocabSize: 128,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 10_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 6,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 2, 2, 1, 2, 1],
            hiddenActivation: "silu",
            family: .qwen38flashnext,
            attnOutputGate: true,
            attentionScale: 0.25,              // 16^-0.5
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 4,
                keyHeadDim: 8, valueHeadDim: 8,
                convKernelSize: 4),
            numSharedExperts: 1,
            flashNext: FlashNextConfig(
                hcCount: 4,
                hcLowRank: 8,
                indexerNumHeads: 2,
                indexerHeadDim: 8,
                indexerNumKVHeads: 1,
                indexerBudget: 8,
                indexerCompressRatio: 4,
                pleLayerIDs: [2],
                pleNgramShardCount: 2,
                pleNgramVocabSizeBase: 97,
                pleConvKernelSize: 4,
                pleEosTokenID: 0))
    }

    // MARK: - Row pool geometry (PleRowPoolPlan's rules at toy width)

    enum Pool {
        static let rowDim = 4                                  // hidden 64 / 16 heads
        static let rowStride = rowDim * 2                       // BF16, dense
        static let pageBytes = 16_384
        static let rowsPerBlock = pageBytes / rowStride         // 2048
        static let blockStride = pageBytes
        static let file = "ple/layer_01_ngram_rows.bin"
        static let layer = 1                                    // ple_layer_ids [2] -> layers[1]
    }

    // MARK: - Install

    /// Build a `.gturbo` from the toy checkpoint in a fresh temp directory.
    /// The caller owns (and should remove) the returned directory.
    @discardableResult
    static func installToyCheckpoint() throws -> URL {
        let arch = archConfig()
        let source = try Safetensors(
            url: checkpointDirectory.appendingPathComponent("model.safetensors"))
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-qwen4exp-parity-\(UUID().uuidString)")
        let expertsDir = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: expertsDir,
                                                withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("ple"), withIntermediateDirectories: true)

        var files: [String: [String: Any]] = [:]

        // --- 1. Resident file: everything that is neither a fused expert
        //        tensor nor an n-gram shard, at its source dtype.
        let residentNames = source.names.filter {
            FlashNextPlanner.fusedExpertRole(in: $0) == nil
                && FlashNextPlanner.ngramShardIndex(in: $0) == nil
        }.sorted(by: FlashNextPlanner.residentOrdering)

        let (residentBytes, residentSha) = try writeResident(
            names: residentNames, source: source,
            to: dir.appendingPathComponent("model_weights.bin"))
        files["model_weights.bin"] = ["size": residentBytes, "sha256": residentSha]

        // --- 2. Routed experts: the fused [E, 2I, H] / [E, H, I] tensors split
        //        into per-expert page-aligned blobs, gate | up | down.
        let layout = try writeExpertPool(source: source, arch: arch, to: expertsDir,
                                         files: &files)
        files["packed_experts/layout.json"] = ["size": layout.size,
                                               "sha256": layout.sha256]

        // --- 3. The PLE n-gram table as a row-lookup pool.
        let pool = try writeRowPool(source: source, to: dir)
        files[Pool.file] = ["size": pool.size, "sha256": pool.sha256]

        // --- 4. manifest.json.
        let manifest: [String: Any] = [
            "magic": "GTURBO", "versionMajor": 1, "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "qwen4exp-parity-toy",
            "arch": archDictionary(arch),
            "files": files,
            "expertsPerLayer": arch.numExperts,
            "numLayers": arch.numLayers,
            "expertStride": layout.expertStride,
            "plePool": plePoolBlock(rows: pool.rows, shardRows: pool.shardRows,
                                    fileSize: pool.size),
            "sidecars": ["mtp": ["carried": false, "tensorCount": 0],
                         "vision": ["carried": false, "tensorCount": 0]],
            // The install stores norms exactly as the checkpoint does, so the
            // loader's family-gated `(1 + w)` bake is the one that applies.
            "zeroCenteredNormsBakedAtInstall": false,
        ]
        try JSONSerialization
            .data(withJSONObject: manifest,
                  options: [.sortedKeys, .withoutEscapingSlashes])
            .write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    // MARK: - Resident

    private static func writeResident(names: [String],
                                      source: Safetensors,
                                      to url: URL) throws -> (Int, String) {
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

        struct Placed { let name: String; let dtype: UInt8; let shape: [UInt32]
                        let offset: UInt64; let bytes: [UInt8] }
        var placed: [Placed] = []
        var cursor = indexBytes
        for name in names {
            let entry = try source.entry(name)
            let bytes = source.rawBytes(entry)
            placed.append(Placed(name: name,
                                 dtype: dtypeCode(entry.dtype),
                                 shape: padTo4(entry.shape),
                                 offset: cursor,
                                 bytes: bytes))
            cursor += UInt64(bytes.count)
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
                let entry = ResidentEntry(
                    name: p.name, dtype: p.dtype, logicalShape4: p.shape,
                    fileOffset: p.offset, sizeBytes: UInt64(p.bytes.count),
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: dummySourceTensor(p.name),
                    sourceScales: nil, sourceBiases: nil)
                GTurboBinary.writeIndexEntry(
                    into: base.advanced(by: headerBytes + i * entryBytes),
                    entry: entry, nameOffset: nameOffsets[i])
            }
            _ = stringTable.withUnsafeBytes {
                memcpy(base.advanced(by: stringTableBase), $0.baseAddress!,
                       stringTable.count)
            }
            for p in placed {
                _ = p.bytes.withUnsafeBytes {
                    memcpy(base.advanced(by: Int(p.offset)), $0.baseAddress!,
                           p.bytes.count)
                }
            }
        }
        try Data(file).write(to: url)
        return (file.count, try Sha256Verifier.hashFile(at: url))
    }

    // MARK: - Expert pool

    private struct LayoutResult { let size: Int; let sha256: String
                                  let expertStride: UInt64 }

    private static func writeExpertPool(source: Safetensors,
                                        arch: ArchConfig,
                                        to dir: URL,
                                        files: inout [String: [String: Any]]) throws
        -> LayoutResult {
        let hidden = arch.hiddenSize
        let inter = arch.moeIntermediateSize
        let gateBytes = inter * hidden * 2
        let downBytes = hidden * inter * 2
        let blobBytes = 2 * gateBytes + downBytes
        let expertStride = roundUpToPage(UInt64(blobBytes))

        var layers: [[String: Any]] = []
        for L in 0..<arch.numLayers {
            let gateUp = try source.entry("\(trunk)layers.\(L).mlp.experts.gate_up_proj")
            let down = try source.entry("\(trunk)layers.\(L).mlp.experts.down_proj")
            precondition(gateUp.shape == [arch.numExperts, 2 * inter, hidden])
            precondition(down.shape == [arch.numExperts, hidden, inter])
            let gateUpRaw = source.rawBytes(gateUp)
            let downRaw = source.rawBytes(down)

            var payload = [UInt8](repeating: 0,
                                  count: Int(expertStride) * arch.numExperts)
            var experts: [[String: Any]] = []
            for e in 0..<arch.numExperts {
                let base = e * Int(expertStride)
                var tensors: [String: [String: Any]] = [:]
                var cursor = 0
                func place(_ role: String, _ src: ArraySlice<UInt8>,
                           rows: Int, cols: Int) {
                    payload.replaceSubrange(
                        (base + cursor)..<(base + cursor + src.count), with: src)
                    tensors[role] = ["offset": cursor, "size": src.count,
                                     "dtype": "BF16", "shape": [rows, cols]]
                    cursor += src.count
                }
                let gu = e * 2 * gateBytes
                place("gate", gateUpRaw[gu..<(gu + gateBytes)],
                      rows: inter, cols: hidden)
                place("up", gateUpRaw[(gu + gateBytes)..<(gu + 2 * gateBytes)],
                      rows: inter, cols: hidden)
                let dn = e * downBytes
                place("down", downRaw[dn..<(dn + downBytes)],
                      rows: hidden, cols: inter)
                experts.append(["expert": e,
                                "offset": UInt64(base),
                                "size": expertStride,
                                "tensors": tensors])
            }
            let basename = String(format: "layer_%02d.bin", L)
            let url = dir.appendingPathComponent(basename)
            try Data(payload).write(to: url)
            files["packed_experts/\(basename)"] =
                ["size": payload.count,
                 "sha256": try Sha256Verifier.hashFile(at: url)]
            layers.append(["layer": L, "file": basename, "experts": experts])
        }
        let data = try JSONSerialization.data(
            withJSONObject: ["expertStride": expertStride,
                             "numLayers": arch.numLayers,
                             "expertsPerLayer": arch.numExperts,
                             "layers": layers] as [String: Any],
            options: [.sortedKeys])
        let url = dir.appendingPathComponent("layout.json")
        try data.write(to: url)
        return LayoutResult(size: data.count,
                            sha256: try Sha256Verifier.hashFile(at: url),
                            expertStride: expertStride)
    }

    // MARK: - PLE row pool

    private struct PoolResult { let size: Int; let sha256: String
                                let rows: Int; let shardRows: [Int] }

    /// One page-aligned region per source shard, dense BF16 rows inside it —
    /// the layout `PleRowPoolPlan` produces, at toy width (rowDim 4, so 2048
    /// rows fit a 16 KiB block and every shard is a single partial block).
    private static func writeRowPool(source: Safetensors, to dir: URL) throws
        -> PoolResult {
        let prefix = "\(trunk)layers.\(Pool.layer).ple.ple_embedding.ngram_embedding.shard_"
        var shardRows: [Int] = []
        var bytes: [UInt8] = []
        var shard = 0
        while let entry = try? source.entry("\(prefix)\(shard).weight") {
            precondition(entry.shape.count == 2 && entry.shape[1] == Pool.rowDim)
            let rows = entry.shape[0]
            let region = Int(roundUpTo(UInt64(rows * Pool.rowStride),
                                       UInt64(Pool.blockStride)))
            var block = [UInt8](repeating: 0, count: region)
            let raw = source.rawBytes(entry)
            precondition(raw.count == rows * Pool.rowStride)
            block.replaceSubrange(0..<raw.count, with: raw)
            bytes.append(contentsOf: block)
            shardRows.append(rows)
            shard += 1
        }
        precondition(!shardRows.isEmpty, "no n-gram shards in the checkpoint")
        let url = dir.appendingPathComponent(Pool.file)
        try Data(bytes).write(to: url)
        return PoolResult(size: bytes.count,
                          sha256: try Sha256Verifier.hashFile(at: url),
                          rows: shardRows.reduce(0, +),
                          shardRows: shardRows)
    }

    private static func plePoolBlock(rows: Int, shardRows: [Int],
                                     fileSize: Int) -> [String: Any] {
        var shards: [[String: Any]] = []
        var offset = 0
        for (index, count) in shardRows.enumerated() {
            let region = Int(roundUpTo(UInt64(count * Pool.rowStride),
                                       UInt64(Pool.blockStride)))
            shards.append(["shard": index, "rows": count,
                           "offset": offset, "size": region])
            offset += region
        }
        return [
            "kind": "rowLookupPoolV1",
            "layers": [[
                "layer": Pool.layer,
                "file": Pool.file,
                "sourceTensor":
                    "\(trunk)layers.\(Pool.layer).ple.ple_embedding.ngram_embedding",
                "rows": rows,
                "rowDim": Pool.rowDim,
                "storage": "bf16",
                "weightBits": 16,
                "scheme": "none", "scaleType": "none", "biasType": "none",
                "groupSize": 0,
                "rowWeightBytes": Pool.rowStride,
                "rowScaleBytes": 0, "rowBiasBytes": 0,
                "rowStride": Pool.rowStride,
                "rowsPerBlock": Pool.rowsPerBlock,
                "blockStride": Pool.blockStride,
                "fileSize": fileSize,
                "shards": shards,
            ] as [String: Any]],
        ]
    }

    // MARK: - Manifest arch block

    private static func archDictionary(_ a: ArchConfig) -> [String: Any] {
        let la = a.linearAttention
        let fn = a.flashNext
        return [
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
            "linearNumKHeads": la.numKHeads, "linearNumVHeads": la.numVHeads,
            "linearKeyHeadDim": la.keyHeadDim,
            "linearValueHeadDim": la.valueHeadDim,
            "linearConvKernelSize": la.convKernelSize,
            "numSharedExperts": a.numSharedExperts,
            "hcCount": fn.hcCount, "hcLowRank": fn.hcLowRank,
            "indexerNumHeads": fn.indexerNumHeads,
            "indexerHeadDim": fn.indexerHeadDim,
            "indexerNumKVHeads": fn.indexerNumKVHeads,
            "indexerBudget": fn.indexerBudget,
            "indexerCompressRatio": fn.indexerCompressRatio,
            "pleLayerIDs": fn.pleLayerIDs,
            "pleNgramShardCount": fn.pleNgramShardCount,
            "pleNgramVocabSizeBase": fn.pleNgramVocabSizeBase,
            "pleConvKernelSize": fn.pleConvKernelSize,
            "pleEosTokenID": fn.pleEosTokenID,
            "requiredAxes": ["hyperConnectionsLowRank", "attentionIndexer",
                             "pleNgramEmbedding"],
        ]
    }

    // MARK: - Production-shaped config (for the repack-blocker test)

    /// The toy `config.json` rewritten into the shape the production checkpoint
    /// ships: a `qwen4_exp` multimodal wrapper with a `text_config`, and
    /// `layer_types` back in the pre-`__post_init__` `"full_attention"` form.
    /// Repairs blockers (1) and (2); blocker (3) survives it.
    static func productionShapedConfig() throws -> [String: Any] {
        let data = try Data(contentsOf:
            checkpointDirectory.appendingPathComponent("config.json"))
        guard var text = try JSONSerialization.jsonObject(with: data)
            as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        text["layer_types"] = (text["layer_types"] as? [String] ?? []).map {
            $0 == "qwen_sparse_attention" ? "full_attention" : $0
        }
        return ["model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config": text]
    }

    // MARK: - Small helpers

    private static func dtypeCode(_ dtype: String) -> UInt8 {
        switch dtype {
        case "BF16": 1
        case "F16":  2
        case "F32":  3
        case "I64":  4
        case "I32":  5
        default: preconditionFailure("unsupported checkpoint dtype \(dtype)")
        }
    }

    private static func padTo4(_ shape: [Int]) -> [UInt32] {
        var out = shape.map { UInt32($0) }
        while out.count < 4 { out.append(0) }
        return Array(out.prefix(4))
    }

    private static func roundUpTo(_ value: UInt64, _ multiple: UInt64) -> UInt64 {
        multiple == 0 ? value : ((value + multiple - 1) / multiple) * multiple
    }

    private static func roundUpToPage(_ value: UInt64) -> UInt64 {
        roundUpTo(value, UInt64(getpagesize()))
    }

    private static func dummySourceTensor(_ name: String) -> SourceTensor {
        SourceTensor(name: name, shardPath: "/dev/null", dtype: .bf16,
                     shape: [1, 1], absoluteOffset: 0, sizeBytes: 0)
    }
}

// MARK: - Minimal safetensors reader

/// Read-only safetensors reader: the header JSON plus byte ranges into an
/// mmapped `Data`. Enough for the toy checkpoint; no dtype conversion beyond
/// BF16/I64, which is all this family's source carries.
struct Safetensors {
    struct Entry {
        let name: String
        let dtype: String
        let shape: [Int]
        let begin: Int
        let end: Int
    }

    private let data: Data
    private let payloadBase: Int
    let entries: [String: Entry]
    let names: [String]

    init(url: URL) throws {
        let bytes = try Data(contentsOf: url, options: .mappedIfSafe)
        guard bytes.count >= 8 else { throw CocoaError(.fileReadCorruptFile) }
        var headerLength: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerLength) { dst in
            bytes.copyBytes(to: dst, from: 0..<8)
        }
        let headerEnd = 8 + Int(headerLength)
        guard headerEnd <= bytes.count else { throw CocoaError(.fileReadCorruptFile) }
        data = bytes
        payloadBase = headerEnd
        guard let root = try JSONSerialization
            .jsonObject(with: bytes.subdata(in: 8..<headerEnd)) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var parsed: [String: Entry] = [:]
        for (name, value) in root where name != "__metadata__" {
            guard let object = value as? [String: Any],
                  let dtype = object["dtype"] as? String,
                  let shape = object["shape"] as? [Int],
                  let offsets = object["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }
            parsed[name] = Entry(name: name, dtype: dtype, shape: shape,
                                 begin: offsets[0], end: offsets[1])
        }
        entries = parsed
        names = parsed.keys.sorted()
    }

    func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return entry
    }

    func rawBytes(_ entry: Entry) -> [UInt8] {
        [UInt8](data[(payloadBase + entry.begin)..<(payloadBase + entry.end)])
    }

    /// BF16 (or F32) tensor widened to `[Float]`, row-major.
    func floats(_ name: String) throws -> [Float] {
        let entry = try self.entry(name)
        let bytes = rawBytes(entry)
        switch entry.dtype {
        case "BF16":
            return stride(from: 0, to: bytes.count, by: 2).map {
                Quantization.bf16ToFloat(UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8)
            }
        case "F32":
            return stride(from: 0, to: bytes.count, by: 4).map { i in
                Float(bitPattern: UInt32(bytes[i]) | UInt32(bytes[i + 1]) << 8
                        | UInt32(bytes[i + 2]) << 16 | UInt32(bytes[i + 3]) << 24)
            }
        default:
            preconditionFailure("\(name) is \(entry.dtype), not a float tensor")
        }
    }
}
