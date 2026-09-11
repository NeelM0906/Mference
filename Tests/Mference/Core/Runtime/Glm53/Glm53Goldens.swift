import Foundation
@testable import Mference

/// Where the `glm53flash` reference goldens and the toy checkpoint they were
/// captured from live (see `Scripts/parity/README.md`, "glm53flash").
enum Glm53Fixtures {
    static let repoRoot: URL = {
        var root = URL(fileURLWithPath: #filePath)
        // Tests/Mference/Core/Runtime/Glm53/<this file> -> repo root.
        for _ in 0..<6 { root.deleteLastPathComponent() }
        return root
    }()

    static let directory = repoRoot.appendingPathComponent("Tests/Mference/Fixtures/glm53")
    static let checkpointDirectory = directory.appendingPathComponent("toy-ckpt")

    static func manifest() throws -> [String: Any] {
        let url = directory.appendingPathComponent("goldens-manifest.json")
        guard let root = try JSONSerialization
            .jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return root
    }

    /// The generator's `TOY` dictionary, from the manifest.
    static func toyConfig() throws -> [String: Any] {
        guard let config = try manifest()["config"] as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return config
    }

    /// Cross-checks `ArchConfig.glm53Toy()` against the manifest's toy
    /// configuration; returns the mismatching field names.
    static func toyConfigMismatches() throws -> [String] {
        let c = try toyConfig()
        let a = ArchConfig.glm53Toy()
        let ca = a.compressedAttention
        let la = a.linearAttention
        let g = a.glm53
        var out: [String] = []
        func int(_ key: String, _ expected: Int) {
            if (c[key] as? NSNumber)?.intValue != expected { out.append(key) }
        }
        func double(_ key: String, _ expected: Double) {
            if (c[key] as? NSNumber)?.doubleValue != expected { out.append(key) }
        }
        int("vocab_size", a.vocabSize); int("hidden_size", a.hiddenSize)
        int("num_hidden_layers", a.numLayers)
        int("moe_intermediate_size", a.moeIntermediateSize)
        int("intermediate_size", a.denseIntermediateSize)
        int("num_attention_heads", a.numHeads)
        int("qk_nope_head_dim", g.qkNopeHeadDim); int("v_head_dim", g.vHeadDim)
        int("kv_lora_rank", g.kvLoraRank); int("q_lora_rank", ca.qLoraRank)
        int("n_routed_experts", a.numExperts); int("num_experts_per_tok", a.topKExperts)
        int("n_shared_experts", a.numSharedExperts)
        int("first_k_dense_replace", a.numDenseLayers)
        double("routed_scaling_factor", a.routedScalingFactor)
        double("swiglu_limit", a.swigluLimit)
        double("rms_norm_eps", g.rmsNormEps)
        int("index_topk", ca.indexTopK); int("index_head_dim", ca.indexHeadDim)
        int("index_n_heads", ca.indexNHeads); int("index_kpool", g.indexKPool)
        if (c["index_kpool_always_select_tail"] as? Bool) != g.indexKPoolAlwaysSelectTail {
            out.append("index_kpool_always_select_tail")
        }
        int("hc_mult", a.hyperConnections.mult)
        int("hc_sinkhorn_iters", a.hyperConnections.sinkhornIters)
        double("hc_eps", a.hyperConnections.eps)
        if let lac = c["linear_attn_config"] as? [String: Any] {
            if (lac["num_heads"] as? NSNumber)?.intValue != la.numKHeads { out.append("linear_attn_config.num_heads") }
            if (lac["head_dim"] as? NSNumber)?.intValue != la.keyHeadDim { out.append("linear_attn_config.head_dim") }
            if (lac["short_conv_kernel_size"] as? NSNumber)?.intValue != la.convKernelSize {
                out.append("linear_attn_config.short_conv_kernel_size")
            }
            if (lac["gate_lower_bound"] as? NSNumber)?.doubleValue != g.kdaGateLowerBound {
                out.append("linear_attn_config.gate_lower_bound")
            }
        } else {
            out.append("linear_attn_config")
        }
        if let types = c["layer_types"] as? [String] {
            let mask = types.map { $0 == "linear_attention" ? 7 : 8 }
            if mask != a.fullAttentionLayerMask.map({ Int($0) }) { out.append("layer_types") }
        }
        return out
    }
}

/// Reader for one `prompt_*` / `decode_*` golden pair.
struct Glm53Goldens {
    enum Prompt: String, CaseIterable { case short, long }
    enum Phase: String { case prompt, decode }

    struct Tensor {
        let shape: [Int]
        let values: [Float]

        func row(_ r: Int) -> [Float] {
            let width = shape[shape.count - 1]
            return Array(values[(r * width)..<((r + 1) * width)])
        }

        var rows: Int { shape.count >= 2 ? shape[0] : 1 }
    }

    let prompt: Prompt
    let phase: Phase
    let tensors: [String: Tensor]
    let integers: [String: Any]

    init(prompt: Prompt, phase: Phase) throws {
        self.prompt = prompt
        self.phase = phase
        let base = Glm53Fixtures.directory
        tensors = try Self.readSafetensors(
            base.appendingPathComponent("\(phase.rawValue)_\(prompt.rawValue).safetensors"))
        let json = try Data(contentsOf: base.appendingPathComponent(
            "integers_\(phase.rawValue)_\(prompt.rawValue).json"))
        guard let root = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        integers = root
    }

    static func promptTokens(_ prompt: Prompt) throws -> [Int] {
        guard let prompts = try Glm53Fixtures.manifest()["prompts"] as? [String: Any],
              let entry = prompts[prompt.rawValue] as? [String: Any],
              let ids = entry["ids"] as? [Int] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return ids
    }

    static func decodeSteps() throws -> Int {
        guard let steps = (try Glm53Fixtures.manifest()["decode_steps"] as? NSNumber)?.intValue
        else { throw CocoaError(.fileReadCorruptFile) }
        return steps
    }

    /// `seq.posNNN.` or `decode.stepNN.`
    static func phasePrefix(position: Int? = nil, step: Int? = nil) -> String {
        if let step { return String(format: "decode.step%02d.", step) }
        return String(format: "seq.pos%03d.", position ?? 0)
    }

    static func layerKey(_ L: Int) -> String { String(format: "layer%02d", L) }

    func tensor(_ key: String) throws -> Tensor {
        guard let value = tensors[key] else { throw ModelError.tensorNotFound(name: key) }
        return value
    }

    func has(_ key: String) -> Bool { tensors[key] != nil || integers[key] != nil }

    func ints(_ key: String) throws -> [Int] {
        guard let value = integers[key] as? [Int] else {
            throw ModelError.tensorNotFound(name: key)
        }
        return value
    }

    func bools(_ key: String) throws -> [Bool] {
        guard let value = integers[key] as? [Bool] else {
            throw ModelError.tensorNotFound(name: key)
        }
        return value
    }

    func int(_ key: String) throws -> Int {
        guard let value = (integers[key] as? NSNumber)?.intValue else {
            throw ModelError.tensorNotFound(name: key)
        }
        return value
    }

    func doubles(_ key: String) throws -> [Double] {
        guard let value = integers[key] as? [Double] else {
            throw ModelError.tensorNotFound(name: key)
        }
        return value
    }

    /// The indexer selection at a key: nil when the goldens record `"dense"`.
    func selection(_ key: String) -> [Int]? {
        integers[key] as? [Int]
    }

    // MARK: - safetensors (float32 only)

    static func readSafetensors(_ url: URL) throws -> [String: Tensor] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8 else { throw CocoaError(.fileReadCorruptFile) }
        var headerLength: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerLength) { dst in
            data.copyBytes(to: dst, from: 0..<8)
        }
        let headerEnd = 8 + Int(headerLength)
        guard let root = try JSONSerialization
            .jsonObject(with: data.subdata(in: 8..<headerEnd)) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var out: [String: Tensor] = [:]
        for (name, value) in root where name != "__metadata__" {
            guard let object = value as? [String: Any],
                  let dtype = object["dtype"] as? String,
                  let shape = object["shape"] as? [Int],
                  let offsets = object["data_offsets"] as? [Int],
                  offsets.count == 2 else { continue }
            precondition(dtype == "F32", "\(name) is \(dtype); goldens are float32")
            let begin = headerEnd + offsets[0]
            let count = (offsets[1] - offsets[0]) / 4
            var values = [Float](repeating: 0, count: count)
            _ = values.withUnsafeMutableBytes { dst in
                data.copyBytes(to: dst, from: begin..<(begin + count * 4))
            }
            out[name] = Tensor(shape: shape, values: values)
        }
        return out
    }
}

/// The toy checkpoint's tensors, dequantized to float32 the way the reference
/// consumed them: INT8 / INT4 MLX affine group-64 as `q * scale + bias` with
/// the BF16 companions widened exactly, BF16 widened, F32 as stored.
final class Glm53ToyCheckpoint {
    struct Matrix {
        let rows: Int
        let cols: Int
        let values: [Float]

        func row(_ r: Int) -> ArraySlice<Float> { values[(r * cols)..<((r + 1) * cols)] }
    }

    private let shards: [String: Safetensors]
    private let weightMap: [String: String]
    private var cache: [String: Matrix] = [:]

    init(directory: URL = Glm53Fixtures.checkpointDirectory) throws {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
                as? [String: Any],
              let map = root["weight_map"] as? [String: String] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        weightMap = map
        var opened: [String: Safetensors] = [:]
        for shard in Set(map.values) {
            opened[shard] = try Safetensors(url: directory.appendingPathComponent(shard))
        }
        shards = opened
    }

    var names: [String] { weightMap.keys.sorted() }

    private func shard(for name: String) throws -> Safetensors {
        guard let file = weightMap[name], let shard = shards[file] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return shard
    }

    func entry(_ name: String) throws -> Safetensors.Entry {
        try shard(for: name).entry(name)
    }

    /// A stored BF16 or F32 tensor, widened.
    func floats(_ name: String) throws -> [Float] {
        try shard(for: name).floats(name)
    }

    /// Raw bytes of any stored tensor.
    func rawBytes(_ name: String) throws -> [UInt8] {
        let shard = try shard(for: name)
        return shard.rawBytes(try shard.entry(name))
    }

    /// A weight matrix under `base` — either `base.weight` stored BF16, or the
    /// `base.weight` / `.scales` / `.biases` affine-packed triplet at 8 or 4
    /// bits — dequantized to `[rows, cols]` float32. For a stacked tensor
    /// (`[E, rows, cols]` experts or `[heads, rows, cols]` per-head folds)
    /// pass `slab` to select one `[rows, cols]` slab.
    func matrix(_ base: String, slab: Int? = nil) throws -> Matrix {
        let key = slab.map { "\(base)#\($0)" } ?? base
        if let cached = cache[key] { return cached }
        let stored = weightMap[base + ".weight"] != nil ? base + ".weight" : base
        let shard = try shard(for: stored)
        let entry = try shard.entry(stored)
        let result: Matrix
        if entry.dtype == "U32" {
            let scalesEntry = try shard.entry(base + ".scales")
            let packed = shard.rawBytes(entry)
            let scales = try shard.floats(base + ".scales")
            let biases = try shard.floats(base + ".biases")
            let (rows, cols, bits, index) = Self.geometry(entry: entry, scales: scalesEntry, slab: slab)
            let groups = cols / Quantization.groupSize
            var values = [Float](repeating: 0, count: rows * cols)
            let packedRowBytes = cols * bits / 8
            let packedBase = index * rows * packedRowBytes
            let companionBase = index * rows * groups
            for r in 0..<rows {
                let rowBytes = packedBase + r * packedRowBytes
                for c in 0..<cols {
                    let q: Int
                    if bits == 8 {
                        q = Int(packed[rowBytes + c])
                    } else {
                        let byte = packed[rowBytes + c / 2]
                        q = Int(c % 2 == 0 ? (byte & 0x0F) : (byte >> 4))
                    }
                    let g = companionBase + r * groups + c / Quantization.groupSize
                    values[r * cols + c] = Float(q) * scales[g] + biases[g]
                }
            }
            result = Matrix(rows: rows, cols: cols, values: values)
        } else {
            let shape = entry.shape
            let all = try shard.floats(stored)
            if let slab, shape.count == 3 {
                let rows = shape[1], cols = shape[2]
                result = Matrix(rows: rows, cols: cols,
                                values: Array(all[(slab * rows * cols)..<((slab + 1) * rows * cols)]))
            } else {
                precondition(slab == nil, "\(base) is rank \(shape.count); no slab to select")
                let rows = shape.count >= 2 ? shape[shape.count - 2] : 1
                let cols = shape[shape.count - 1]
                result = Matrix(rows: rows, cols: cols, values: all)
            }
        }
        cache[key] = result
        return result
    }

    private static func geometry(entry: Safetensors.Entry, scales: Safetensors.Entry,
                                 slab: Int?) -> (Int, Int, Int, Int) {
        let groups = scales.shape[scales.shape.count - 1]
        let cols = groups * Quantization.groupSize
        let rows = entry.shape[entry.shape.count - 2]
        let packedWords = entry.shape[entry.shape.count - 1]
        let bits = packedWords * 32 / cols
        precondition(bits == 4 || bits == 8, "unexpected packed width \(bits)")
        if entry.shape.count == 3 {
            precondition(slab != nil && slab! < entry.shape[0], "slab out of range")
            return (rows, cols, bits, slab!)
        }
        precondition(slab == nil)
        return (rows, cols, bits, 0)
    }
}
