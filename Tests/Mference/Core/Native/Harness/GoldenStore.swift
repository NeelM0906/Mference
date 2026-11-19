import Foundation
@testable import Mference

/// Loader for the llama.cpp goldens of the native engine
/// (`docs/NATIVE_PARITY.md`, format `mfe-golden-v1`).
///
/// A golden run is a directory with one subdirectory per prompt of the fixed
/// prompt set; each holds `golden.json` (the manifest), `prompt.txt` (the
/// chat template as llama.cpp rendered it), `logits.f32` (pass A: the logits
/// row behind every greedy choice) and `tensors/*.f32` (pass B: graph tensors
/// at chosen positions). Every `.f32` file is row-major little-endian F32
/// `[rows x cols]`, row `i` belonging to the manifest's `i`-th position.
///
/// Files are memory-mapped, never read into the heap, so a test can open the
/// whole set (bounded memory). `MFERENCE_NATIVE_GOLDENS` names the run
/// directory; tests that need it skip when it is unset.
enum GoldenStore {
    /// The environment variable that gates every real-golden test.
    static let environmentKey = "MFERENCE_NATIVE_GOLDENS"

    /// The golden run directory, or nil when the gate is unset.
    static var root: URL? {
        guard let path = ProcessInfo.processInfo.environment[environmentKey], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// The repository root, from this file's path
    /// (`Tests/Mference/Core/Native/Harness/<file>`).
    static let repoRoot: URL = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 { url.deleteLastPathComponent() }
        return url
    }()

    /// Committed fixtures of the native harness.
    static let fixturesDirectory = repoRoot.appendingPathComponent("Tests/Mference/Fixtures/native")

    /// The committed fixed prompt set.
    static func promptSet(named name: String = "quick-v1") throws -> PromptSet {
        let url = fixturesDirectory.appendingPathComponent("prompts/\(name).json")
        return try JSONDecoder().decode(PromptSet.self, from: Data(contentsOf: url))
    }

    /// Opens prompt `id` of the run at `root` (default: the gated root).
    static func open(_ id: String, root: URL? = GoldenStore.root) throws -> GoldenPrompt {
        guard let root else { throw GoldenError.notConfigured }
        return try GoldenPrompt(directory: root.appendingPathComponent(id, isDirectory: true))
    }
}

enum GoldenError: Error, Equatable, CustomStringConvertible {
    case notConfigured
    case badFormat(String)
    case missing(String)
    case sizeMismatch(String)

    var description: String {
        switch self {
        case .notConfigured: return "golden: \(GoldenStore.environmentKey) is not set"
        case .badFormat(let s): return "golden: bad format: \(s)"
        case .missing(let s): return "golden: missing \(s)"
        case .sizeMismatch(let s): return "golden: size mismatch: \(s)"
        }
    }
}

// MARK: - The fixed prompt set (Tests/Mference/Fixtures/native/prompts)

struct PromptSet: Decodable {
    struct Prompt: Decodable {
        var id: String
        var origin: String
        var maxNew: Int
        var enableThinking: Bool?
        enum CodingKeys: String, CodingKey {
            case id, origin
            case maxNew = "max_new"
            case enableThinking = "enable_thinking"
        }
    }
    var name: String
    var version: Int
    var prompts: [Prompt]
}

// MARK: - Manifest (golden.json)

struct GoldenManifest: Decodable {
    struct Source: Decodable {
        var engine: String
        var commit: String
        var backend: String
        var modelDesc: String
        var modelPath: String
        var nBatch: Int
        var nUbatch: Int
        var kv: String
        var flashAttn: String
        var nCtx: Int
        enum CodingKeys: String, CodingKey {
            case engine, commit, backend, kv
            case modelDesc = "model_desc"
            case modelPath = "model_path"
            case nBatch = "n_batch"
            case nUbatch = "n_ubatch"
            case flashAttn = "flash_attn"
            case nCtx = "n_ctx"
        }
    }

    struct Logits: Decodable {
        var file: String
        var dtype: String
        var rows: Int
        var cols: Int
        var positionRanges: [[Int]]
        enum CodingKeys: String, CodingKey {
            case file, dtype, rows, cols
            case positionRanges = "position_ranges"
        }
    }

    struct Capture: Decodable {
        var chunk: Int
        var selectedRanges: [[Int]]
        var decodeSteps: Int
        var stateSteps: Int
        enum CodingKeys: String, CodingKey {
            case chunk
            case selectedRanges = "selected_ranges"
            case decodeSteps = "decode_steps"
            case stateSteps = "state_steps"
        }
    }

    struct Tensor: Decodable {
        /// File key: `name`, plus `.k` for the k-th (k > 0) node of that name.
        var key: String
        /// llama.cpp graph name, `<base>-<layer>` or `<base>` for il = -1.
        var name: String
        var base: String
        var layer: Int
        var occurrence: Int
        var op: String
        var ne: [Int]
        var dtype: String
        var cols: Int
        var rows: Int
        var positionRanges: [[Int]]
        var file: String
        enum CodingKeys: String, CodingKey {
            case key, name, base, layer, occurrence, op, ne, dtype, cols, rows, file
            case positionRanges = "position_ranges"
        }
    }

    var format: String
    var promptID: String
    var promptSet: String
    var mode: String
    var source: Source
    var enableThinking: Bool
    var renderedPromptFile: String
    var nVocab: Int
    var promptTokens: [Int32]
    var generatedTokens: [Int32]
    /// Each step's own argmax. Equal to `generatedTokens` in a free-running
    /// golden; differs where a teacher-forced run disagrees with its teacher.
    var argmaxTokens: [Int32]?
    /// The run directory whose tokens were forced (empty: free running).
    var teacher: String?
    var stoppedOnEog: Bool
    var logits: Logits
    var capture: Capture
    var tensors: [Tensor]

    enum CodingKeys: String, CodingKey {
        case format, mode, source, logits, capture, tensors
        case promptID = "prompt_id"
        case promptSet = "prompt_set"
        case enableThinking = "enable_thinking"
        case renderedPromptFile = "rendered_prompt_file"
        case nVocab = "n_vocab"
        case promptTokens = "prompt_tokens"
        case generatedTokens = "generated_tokens"
        case argmaxTokens = "argmax_tokens"
        case teacher
        case stoppedOnEog = "stopped_on_eog"
    }

    static let currentFormat = "mfe-golden-v1"

    /// Expands `[[start, end), ...]` into positions.
    static func positions(_ ranges: [[Int]]) throws -> [Int] {
        var out: [Int] = []
        for r in ranges {
            guard r.count == 2, r[0] <= r[1] else { throw GoldenError.badFormat("range \(r)") }
            out.append(contentsOf: r[0]..<r[1])
        }
        return out
    }
}

// MARK: - Tensors

/// A memory-mapped `[rows x cols]` F32 matrix whose row `i` is the value at
/// `positions[i]`.
final class GoldenMatrix {
    let positions: [Int]
    let cols: Int
    private let data: Data
    private let rowOfPosition: [Int: Int]

    init(url: URL, positions: [Int], cols: Int) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw GoldenError.missing(url.path) }
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let expected = positions.count * cols * MemoryLayout<Float>.size
        guard data.count == expected else {
            throw GoldenError.sizeMismatch("\(url.lastPathComponent): \(data.count) bytes, expected \(expected)")
        }
        self.data = data
        self.positions = positions
        self.cols = cols
        var map: [Int: Int] = [:]
        for (i, p) in positions.enumerated() where map[p] == nil { map[p] = i }
        self.rowOfPosition = map
    }

    var rows: Int { positions.count }

    /// Row index of `position`, nil when it was not captured.
    func rowIndex(ofPosition position: Int) -> Int? { rowOfPosition[position] }

    /// Calls `body` with row `index` (no copy).
    func withRow<R>(_ index: Int, _ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        precondition(index >= 0 && index < rows, "golden row \(index) out of 0..<\(rows)")
        return try data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: index * cols * MemoryLayout<Float>.size)
            return try body(UnsafeBufferPointer(start: base.assumingMemoryBound(to: Float.self), count: cols))
        }
    }

    /// A copy of row `index`.
    func row(_ index: Int) -> [Float] { withRow(index) { Array($0) } }

    /// A copy of the row at `position`, nil when it was not captured.
    func row(atPosition position: Int) -> [Float]? {
        rowIndex(ofPosition: position).map { row($0) }
    }
}

// MARK: - One prompt of a golden run

final class GoldenPrompt {
    let directory: URL
    let manifest: GoldenManifest
    private var tensorsByKey: [String: GoldenManifest.Tensor] = [:]

    init(directory: URL) throws {
        self.directory = directory
        let url = directory.appendingPathComponent("golden.json")
        guard FileManager.default.fileExists(atPath: url.path) else { throw GoldenError.missing(url.path) }
        manifest = try JSONDecoder().decode(GoldenManifest.self, from: Data(contentsOf: url))
        guard manifest.format == GoldenManifest.currentFormat else {
            throw GoldenError.badFormat("format \(manifest.format)")
        }
        for t in manifest.tensors { tensorsByKey[t.key] = t }
    }

    var id: String { manifest.promptID }
    var promptTokens: [Int32] { manifest.promptTokens }
    var generatedTokens: [Int32] { manifest.generatedTokens }

    /// Prompt tokens followed by every generated token but the last: the
    /// teacher-forced input whose step `i >= promptTokens.count - 1` produced
    /// `logits` row `i - (promptTokens.count - 1)`.
    var teacherForcedTokens: [Int32] { promptTokens + generatedTokens.dropLast() }

    /// The chat template as llama.cpp rendered it.
    func renderedPrompt() throws -> String {
        try String(contentsOf: directory.appendingPathComponent(manifest.renderedPromptFile), encoding: .utf8)
    }

    /// Pass A logits: row `g` chose `generatedTokens[g]` at position
    /// `promptTokens.count - 1 + g`.
    func logits() throws -> GoldenMatrix {
        let l = manifest.logits
        let positions = try GoldenManifest.positions(l.positionRanges)
        guard positions.count == l.rows, l.rows == manifest.generatedTokens.count else {
            throw GoldenError.badFormat("logits rows \(l.rows), positions \(positions.count), tokens \(manifest.generatedTokens.count)")
        }
        return try GoldenMatrix(url: directory.appendingPathComponent(l.file), positions: positions, cols: l.cols)
    }

    /// Every captured tensor's manifest entry.
    var tensors: [GoldenManifest.Tensor] { manifest.tensors }

    /// Manifest entry of graph tensor `name` (`<base>-<layer>`), `occurrence`
    /// counting nodes of that name in one graph from 0.
    func entry(named name: String, occurrence: Int = 0) -> GoldenManifest.Tensor? {
        tensorsByKey[occurrence == 0 ? name : "\(name).\(occurrence)"]
    }

    /// Graph tensor `name` at its captured positions.
    func tensor(named name: String, occurrence: Int = 0) throws -> GoldenMatrix {
        guard let t = entry(named: name, occurrence: occurrence) else {
            throw GoldenError.missing("tensor \(name) occurrence \(occurrence) in \(id)")
        }
        let positions = try GoldenManifest.positions(t.positionRanges)
        guard positions.count == t.rows else { throw GoldenError.badFormat("\(t.key): rows \(t.rows) vs \(positions.count)") }
        return try GoldenMatrix(url: directory.appendingPathComponent(t.file), positions: positions, cols: t.cols)
    }

    /// The golden for a `Native.TapPoint` (mapping in `GoldenTapMap`).
    func tensor(for point: Native.TapPoint, config: Native.Qwen35Config) throws -> GoldenMatrix {
        let ref = GoldenTapMap.graphTensor(for: point, config: config)
        return try tensor(named: ref.name, occurrence: ref.occurrence)
    }
}

// MARK: - TapPoint and op-boundary names

/// A llama.cpp graph tensor: its callback name and which node of that name.
struct GoldenGraphTensor: Equatable {
    var name: String
    var occurrence: Int = 0
}

/// Maps the native contract's tap points and op boundaries to the llama.cpp
/// `qwen35` graph tensors (`src/models/qwen35.cpp` at the pin). The table is
/// documented in `docs/NATIVE_PARITY.md`.
enum GoldenTapMap {
    static func layerName(_ base: String, _ layer: Int) -> String { "\(base)-\(layer)" }

    static func graphTensor(for point: Native.TapPoint, config: Native.Qwen35Config) -> GoldenGraphTensor {
        switch point {
        case .embedding: return GoldenGraphTensor(name: "model.input_embed")
        case .mixerInput(let l): return GoldenGraphTensor(name: layerName("attn_norm", l))
        case .mixerOutput(let l):
            switch config.layerKind(l) {
            case .fullAttention: return GoldenGraphTensor(name: layerName("attn_output", l))
            case .gatedDeltaNet: return GoldenGraphTensor(name: layerName("linear_attn_out", l))
            }
        case .ffnOutput(let l): return GoldenGraphTensor(name: layerName("ffn_out", l))
        case .layerOutput(let l): return GoldenGraphTensor(name: layerName("l_out", l))
        case .finalNorm: return GoldenGraphTensor(name: "result_norm")
        case .logits: return GoldenGraphTensor(name: "result_output")
        }
    }

    /// Op boundaries a component test feeds (inputs) and compares (output).
    enum Boundary: String, CaseIterable {
        // full attention (AttentionOp): inputs qGate, k, v; output gated
        case attentionQGate, attentionK, attentionV, attentionOutput
        // Gated DeltaNet (GDNOp): inputs qkv, z, alpha, beta; output
        case gdnQKV, gdnZ, gdnAlpha, gdnBeta, gdnOutput
        // GDN recurrent state before a decode step (first decode step only):
        // conv tail: F32 in ggml order ne [3, 8192], i.e. [8192][3] with
        // element `c*3 + k` (NOT `RecurrentLayout`'s [3][8192]; use
        // `GoldenPrompt.convState(layer:position:)` for that order).
        // S: F32 ne [128, 128, 32], element `h*16384 + j*128 + i` = S[i][j]
        // (key i, value j), the same order as `RecurrentLayout`.
        case gdnConvState, gdnState
        // FFN (QuantGEMM + GlueOps): input, gate, up, SwiGLU, down output
        case ffnInput, ffnGate, ffnUp, ffnSwiGLU, ffnDown
    }

    static func graphTensor(_ boundary: Boundary, layer l: Int) -> GoldenGraphTensor {
        switch boundary {
        case .attentionQGate: return GoldenGraphTensor(name: layerName("Qcur_full", l))
        case .attentionK: return GoldenGraphTensor(name: layerName("Kcur", l))
        case .attentionV: return GoldenGraphTensor(name: layerName("Vcur", l))
        case .attentionOutput: return GoldenGraphTensor(name: layerName("attn_gated", l))
        case .gdnQKV: return GoldenGraphTensor(name: layerName("linear_attn_qkv_mixed", l))
        case .gdnZ: return GoldenGraphTensor(name: layerName("z", l))
        case .gdnAlpha: return GoldenGraphTensor(name: layerName("alpha", l))
        case .gdnBeta: return GoldenGraphTensor(name: layerName("beta", l))
        case .gdnOutput: return GoldenGraphTensor(name: layerName("final_output", l))
        case .gdnConvState: return GoldenGraphTensor(name: layerName("conv_states", l))
        case .gdnState: return GoldenGraphTensor(name: layerName("state_predelta", l))
        case .ffnInput: return GoldenGraphTensor(name: layerName("attn_post_norm", l))
        case .ffnGate: return GoldenGraphTensor(name: layerName("ffn_gate", l))
        case .ffnUp: return GoldenGraphTensor(name: layerName("ffn_up", l))
        case .ffnSwiGLU: return GoldenGraphTensor(name: layerName("ffn_swiglu", l))
        case .ffnDown: return GoldenGraphTensor(name: layerName("ffn_out", l))
        }
    }
}

extension GoldenPrompt {
    /// The golden at an op boundary of block `layer`.
    func tensor(_ boundary: GoldenTapMap.Boundary, layer: Int) throws -> GoldenMatrix {
        let ref = GoldenTapMap.graphTensor(boundary, layer: layer)
        return try tensor(named: ref.name, occurrence: ref.occurrence)
    }

    /// GDN layer `layer`'s conv tail before the decode step at `position`, in
    /// `RecurrentLayout` order: `[convKernel - 1][convChannels]`, element
    /// `k * channels + c`. The golden stores ggml's `[channels][convKernel - 1]`
    /// (`c * taps + k`); this transposes it. Nil when not captured there.
    func convState(layer: Int, position: Int, taps: Int = 3) throws -> [Float]? {
        let m = try tensor(.gdnConvState, layer: layer)
        guard let row = m.row(atPosition: position) else { return nil }
        return Self.transposeConvState(row, taps: taps)
    }

    /// ggml `[channels][taps]` (`c * taps + k`) to `[taps][channels]`
    /// (`k * channels + c`).
    static func transposeConvState(_ ggml: [Float], taps: Int) -> [Float] {
        precondition(taps > 0 && ggml.count % taps == 0, "conv state of \(ggml.count) floats, \(taps) taps")
        let channels = ggml.count / taps
        var out = [Float](repeating: 0, count: ggml.count)
        for c in 0..<channels {
            for k in 0..<taps { out[k * channels + c] = ggml[c * taps + k] }
        }
        return out
    }

    /// GDN layer `layer`'s S before the decode step at `position`, already in
    /// `RecurrentLayout` order (`[valueHeads][valueDim][keyDim]`, element
    /// `h*dv*dk + j*dk + i` = S[i][j]); no transpose. Nil when not captured.
    func ssmState(layer: Int, position: Int) throws -> [Float]? {
        try tensor(.gdnState, layer: layer).row(atPosition: position)
    }
}
