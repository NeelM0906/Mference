import Foundation
import Metal
import Testing
@testable import Mference

/// The all-reference Quick path and each reference component against the
/// llama.cpp goldens, read through the harness's `GoldenStore`. Model runs,
/// skipped unless both are set:
///
/// - `MFERENCE_NATIVE_QUICK_PACK`: the Quick `.mfpack`
/// - `MFERENCE_NATIVE_GOLDENS`: the golden run directory (`GoldenStore.root`)
///
/// Optional: `MFERENCE_NATIVE_GOLDEN_PROMPTS` (comma-separated prompt ids),
/// `MFERENCE_NATIVE_PREFILL_CHUNK` (default 256) and
/// `MFERENCE_NATIVE_REFERENCE_OUT` (directory for the JSON reports).
///
/// D1 is measured teacher-forced: `GoldenPrompt.teacherForcedTokens` are fed,
/// and at every golden logits position the reference's logits are compared
/// with llama.cpp's (greedy agreement with its argmax, KL(golden ‖ reference)).
@Suite("ReferenceGoldenTests", .serialized)
struct ReferenceGoldenTests {
    static let env = ProcessInfo.processInfo.environment
    static let enabled = env["MFERENCE_NATIVE_QUICK_PACK"] != nil && GoldenStore.root != nil

    static func prompts() throws -> [GoldenPrompt] {
        let only = env["MFERENCE_NATIVE_GOLDEN_PROMPTS"].map { Set($0.split(separator: ",").map(String.init)) }
        return try GoldenStore.promptSet().prompts.filter { only?.contains($0.id) ?? true }
            .map { try GoldenStore.open($0.id) }
    }

    static func writeReport<T: Encodable>(_ value: T, name: String) throws {
        guard let out = env["MFERENCE_NATIVE_REFERENCE_OUT"] else { return }
        let url = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url.appendingPathComponent(name))
    }

    struct PromptReport: Codable {
        var prompt: String
        var promptTokens: Int
        var positions: Int
        var expectedPositions: Int
        var agreements: Int
        var greedyAgreement: Double
        var meanKL: Double
        var p99KL: Double
        var maxKL: Double
        var disagreements: [Int]
        var disagreementMargins: [Double]
        var seconds: Double
    }

    struct D1Summary: Codable {
        var head: String
        var positions: Int
        var expectedPositions: Int
        var agreements: Int
        var greedyAgreement: Double
        var meanKL: Double
        var p99KL: Double
        var maxKL: Double
        var bar: String
        var failures: [String]
        var passed: Bool
    }

    /// D1 scored by the harness (`ParityHarness`, `ParityTolerances.d1`):
    /// teacher-forced, greedy agreement over every golden position and
    /// per-token KL(golden ‖ reference) max and mean.
    @Test(.enabled(if: enabled))
    func teacherForcedD1() throws {
        let ctx = try MetalContext()
        let pack = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let chunk = Int(Self.env["MFERENCE_NATIVE_PREFILL_CHUNK"] ?? "") ?? 256
        var reports: [PromptReport] = []
        var results: [ParityHarness.LogitResult] = []
        for golden in try Self.prompts() {
            // Every model and KV arena is freed before the next prompt: buffers
            // from makeBuffer are autoreleased and would otherwise accumulate.
            try autoreleasepool {
                let start = Date()
                let prompt = golden.promptTokens
                let generated = golden.generatedTokens
                let positions = prompt.count + generated.count
                let model = try ReferenceQuickTests.makeModel(ctx: ctx, packURL: pack,
                                                              context: (positions + 63) / 64 * 64 + 64, chunk: chunk)
                func step(_ kind: Native.StepKind, _ tokens: [Int32], at position: Int,
                          logits: Native.LogitRows) throws -> Native.TensorRef? {
                    try model.kv.beginStep(rows: tokens.count, kind: kind)
                    var out: Native.TensorRef?
                    try RefTest.run(ctx) { enc in
                        out = try model.encodeStep(Native.StepPlan(kind: kind, startPosition: position,
                                                                   rows: tokens.count, input: .host(tokens),
                                                                   logits: logits), encoder: enc)
                    }
                    return out
                }
                var position = 0
                // Row g: logits after the prompt and the first g golden tokens.
                let result = try ParityHarness.compareLogits(golden: golden) { g in
                    var out: Native.TensorRef?
                    if g == 0 {
                        while position < prompt.count {
                            let n = min(chunk, prompt.count - position)
                            let last = position + n == prompt.count
                            out = try step(.prefill, Array(prompt[position ..< position + n]), at: position,
                                           logits: last ? .last : .none)
                            position += n
                        }
                    } else {
                        out = try step(.decode, [generated[g - 1]], at: position, logits: .last)
                        position += 1
                    }
                    return out.map(RefTest.floats)
                }
                let d = ParityMetrics.Distribution(result.kld)
                let report = PromptReport(
                    prompt: golden.id, promptTokens: prompt.count, positions: result.positions,
                    expectedPositions: result.expectedPositions, agreements: result.agreements,
                    greedyAgreement: result.agreement, meanKL: d.mean, p99KL: d.p99, maxKL: d.max,
                    disagreements: result.disagreements, disagreementMargins: result.disagreementMargins,
                    seconds: Date().timeIntervalSince(start))
                print("reference D1 \(golden.id): agreement \(result.agreements)/\(result.positions) mean KL "
                      + "\(d.mean) p99 KL \(d.p99) max KL \(d.max) disagreements \(result.disagreements) "
                      + "margins \(result.disagreementMargins) (\(Int(report.seconds)) s)")
                reports.append(report)
                results.append(result)
                try Self.writeReport(reports, name: "d1-teacher-forced.json")
            }
        }
        let report = ParityHarness.LogitReport(results: results)
        let verdict = report.verdict(.d1)
        let d = report.kld
        print(report.table(title: "reference vs quick-v1/metal"))
        try Self.writeReport(D1Summary(
            head: Self.env["MFERENCE_NATIVE_HEAD"] ?? "", positions: report.positions,
            expectedPositions: report.expectedPositions, agreements: report.agreements,
            greedyAgreement: report.agreement, meanKL: d.mean, p99KL: d.p99, maxKL: d.max,
            bar: "ParityTolerances.d1: agreement >= 0.99, per-token KL max <= 0.01, mean <= 0.001",
            failures: verdict.failures, passed: verdict.passed), name: "d1-summary.json")
        #expect(verdict.passed, "\(verdict.failures)")
    }

    // MARK: Per-component goldens

    struct ComponentReport: Codable {
        var prompt: String
        var component: String
        var layer: Int
        var toleranceClass: String
        var relativeL2: Double
        var maxAbsError: Double
        var meanAbsError: Double
    }

    /// The golden graph tensor each component's output is compared with,
    /// which picks its `OpToleranceClass`.
    static let goldenBase: [String: String] = [
        "get_rows": "model.input_embed", "rms_norm": "attn_norm", "swiglu": "ffn_swiglu",
        "gemm.attn_q": "Qcur_full", "gemm.attn_k": "Kcur", "gemm.attn_v": "Vcur", "attention": "attn_gated",
        "gemm.attn_output": "attn_output", "gemm.attn_qkv": "linear_attn_qkv_mixed", "gemm.attn_gate": "z",
        "gemm.ssm_alpha": "alpha", "gemm.ssm_beta": "beta", "gdn": "final_output", "gdn.conv_state": "conv_states",
        "gdn.ssm_state": "state_predelta", "gemm.ssm_out": "linear_attn_out", "gemm.ffn_gate": "ffn_gate",
        "gemm.ffn_up": "ffn_up", "gemm.ffn_down": "ffn_out",
    ]

    /// Each reference component fed the golden's own inputs at the op
    /// boundaries (`GoldenTapMap`), compared with the golden's outputs, on
    /// prompts captured from position 0 with no gaps; plus the GDN conv and
    /// S state where the golden captured it.
    @Test(.enabled(if: enabled))
    func componentsAgainstGoldens() throws {
        let ctx = try MetalContext()
        let pack = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let store = try Native.ReferenceWeightStore(directory: pack, device: ctx.device)
        let c = store.config
        let gemm = Native.ReferenceQuantGEMM(context: ctx)
        try gemm.prepare(types: Set(store.manifest.tensors.map(\.type)).intersection(Native.ReferenceDequant.supportedTypes))
        let glue = Native.ReferenceGlueOps(context: ctx)
        try glue.prepare(config: c)
        var reports: [ComponentReport] = []
        var worst: [String: Double] = [:]
        var notFullyCaptured: [String] = []
        var failures: [String] = []
        for golden in try Self.prompts() {
            let embed = try golden.tensor(for: .embedding, config: c)
            let rows = embed.rows
            // Ops with history (attention, GDN) need every position from 0.
            guard embed.positions == Array(0..<rows), rows <= 512 else {
                notFullyCaptured.append(golden.id)
                continue
            }
            try autoreleasepool {
                func all(_ m: GoldenMatrix) -> [Float] { (0..<m.rows).flatMap { m.row($0) } }
                func load(_ b: GoldenTapMap.Boundary, _ layer: Int) throws -> [Float] { all(try golden.tensor(b, layer: layer)) }
                func tap(_ p: Native.TapPoint) throws -> [Float] { all(try golden.tensor(for: p, config: c)) }
                func record(_ component: String, _ layer: Int, _ ours: [Float], _ want: [Float]) {
                    var stats = ParityMetrics.ErrorStats()
                    stats.add(golden: want, candidate: ours)
                    // `attn_output` names two graph nodes: the full-attention
                    // projection (MUL_MAT, gemm) and the GDN core view (mixer).
                    let base = Self.goldenBase[component] ?? component
                    guard let cls = OpToleranceClass.of(base: base, op: base == "attn_output" ? "MUL_MAT" : nil) else {
                        failures.append("\(component): no tolerance class for golden tensor \(base)")
                        return
                    }
                    reports.append(ComponentReport(prompt: golden.id, component: component, layer: layer,
                                                   toleranceClass: cls.rawValue, relativeL2: stats.relativeL2,
                                                   maxAbsError: stats.maxAbs, meanAbsError: stats.meanAbs))
                    worst[component] = max(worst[component] ?? 0, stats.relativeL2)
                    if let failure = ParityHarness.localVerdict(stats, class: cls,
                                                                name: "\(golden.id) \(component) layer \(layer)") {
                        failures.append(failure)
                    }
                    // Diagnostic split: prompt rows (llama.cpp prefill path)
                    // and decode rows (its single-row path), when the tensor
                    // has one row per position.
                    let prompt = golden.promptTokens.count
                    if want.count % rows == 0, ours.count == want.count, prompt < rows {
                        let cols = want.count / rows
                        for (suffix, range) in [("@prompt", 0 ..< prompt), ("@decode", prompt ..< rows)] {
                            var part = ParityMetrics.ErrorStats()
                            part.add(golden: Array(want[range.lowerBound * cols ..< range.upperBound * cols]),
                                     candidate: Array(ours[range.lowerBound * cols ..< range.upperBound * cols]))
                            reports.append(ComponentReport(prompt: golden.id, component: component + suffix, layer: layer,
                                                           toleranceClass: cls.rawValue, relativeL2: part.relativeL2,
                                                           maxAbsError: part.maxAbs, meanAbsError: part.meanAbs))
                        }
                    }
                }
                func zeros(_ n: Int, _ cols: Int) -> Native.TensorRef {
                    RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: n * cols), rows: n, cols: cols)
                }
                func matmul(_ weight: String, _ input: [Float]) throws -> [Float] {
                    let w = try store.tensor(weight)
                    let x = RefTest.tensor(ctx.device, floats: input, rows: rows, cols: w.cols)
                    let y = zeros(rows, w.rows)
                    try RefTest.run(ctx) { try gemm.encodePrefill($0, weight: w, input: x, output: y, mode: .overwrite) }
                    return RefTest.floats(y)
                }
                let tokens = Array((golden.promptTokens + golden.generatedTokens).prefix(rows)).map { UInt32(bitPattern: $0) }
                let tokenBuffer = RefTest.buffer(ctx.device, bytes: tokens.withUnsafeBytes { Array($0) })
                let emb = zeros(rows, c.hiddenSize)
                try RefTest.run(ctx) {
                    try gemm.encodeGetRows($0, weight: try store.tensor("token_embd.weight"),
                                           tokens: Native.TokenSlot(buffer: tokenBuffer, offset: 0, count: rows), output: emb)
                }
                record("get_rows", -1, RefTest.floats(emb), all(embed))

                let attention = Native.ReferenceAttentionOp(context: ctx)
                let gdn = Native.ReferenceGDNOp(context: ctx)
                let plan = try Native.ReferenceMemoryPlanner().resolve(
                    config: c, weightsMapped: 0, contextLength: (rows + 63) / 64 * 64, precision: .f16, maxVerifyRows: 1,
                    maxPrefillChunk: rows, activationBytes: 0, prefixCacheRAMBudget: 0, budget: 1 << 34)
                let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
                try attention.prepare(config: c, geometry: kv.geometry)
                try gdn.prepare(config: c, layout: kv.recurrentLayout)
                try kv.beginStep(rows: rows, kind: .prefill)
                var previous = all(embed)
                for layer in 0..<c.blockCount {
                    try autoreleasepool {
                        let p = "blk.\(layer)."
                        let normIn = try tap(.mixerInput(layer: layer))
                        let h = zeros(rows, c.hiddenSize)
                        try RefTest.run(ctx) {
                            try glue.encodeRMSNorm($0, input: RefTest.tensor(ctx.device, floats: previous, rows: rows, cols: c.hiddenSize),
                                                   weight: try store.tensor(p + "attn_norm.weight"), output: h)
                        }
                        record("rms_norm", layer, RefTest.floats(h), normIn)
                        if c.layerKind(layer) == .fullAttention {
                            let qg = try load(.attentionQGate, layer)
                            let k = try load(.attentionK, layer)
                            let v = try load(.attentionV, layer)
                            record("gemm.attn_q", layer, try matmul(p + "attn_q.weight", normIn), qg)
                            record("gemm.attn_k", layer, try matmul(p + "attn_k.weight", normIn), k)
                            record("gemm.attn_v", layer, try matmul(p + "attn_v.weight", normIn), v)
                            let out = zeros(rows, c.headCount * c.valueLength)
                            try RefTest.run(ctx) {
                                try attention.encodePrefill(
                                    $0, attnLayer: c.fullAttentionLayers.firstIndex(of: layer)!,
                                    weights: .init(qNorm: try store.tensor(p + "attn_q_norm.weight"),
                                                   kNorm: try store.tensor(p + "attn_k_norm.weight")),
                                    qGate: RefTest.tensor(ctx.device, floats: qg, rows: rows, cols: qg.count / rows),
                                    k: RefTest.tensor(ctx.device, floats: k, rows: rows, cols: k.count / rows),
                                    v: RefTest.tensor(ctx.device, floats: v, rows: rows, cols: v.count / rows),
                                    startPosition: 0, kv: kv.pageTable(), output: out)
                            }
                            let gated = try load(.attentionOutput, layer)
                            record("attention", layer, RefTest.floats(out), gated)
                            record("gemm.attn_output", layer, try matmul(p + "attn_output.weight", gated),
                                   try tap(.mixerOutput(layer: layer)))
                        } else {
                            let qkv = try load(.gdnQKV, layer)
                            let z = try load(.gdnZ, layer)
                            let alpha = try load(.gdnAlpha, layer)
                            let beta = try load(.gdnBeta, layer)
                            record("gemm.attn_qkv", layer, try matmul(p + "attn_qkv.weight", normIn), qkv)
                            record("gemm.attn_gate", layer, try matmul(p + "attn_gate.weight", normIn), z)
                            record("gemm.ssm_alpha", layer, try matmul(p + "ssm_alpha.weight", normIn), alpha)
                            record("gemm.ssm_beta", layer, try matmul(p + "ssm_beta.weight", normIn), beta)
                            let weights = Native.GDNLayerWeights(
                                conv1d: try store.tensor(p + "ssm_conv1d.weight"), dtBias: try store.tensor(p + "ssm_dt.bias"),
                                a: try store.tensor(p + "ssm_a"), norm: try store.tensor(p + "ssm_norm.weight"))
                            let banks = kv.recurrentState(gdnLayer: c.gatedDeltaNetLayers.firstIndex(of: layer)!)
                            // Runs the op over rows 0..<n from a zero state; returns the output.
                            func runGDN(_ n: Int) throws -> [Float] {
                                let layout = kv.recurrentLayout
                                memset(banks.input.buffer.contents().advanced(by: banks.input.offset), 0, layout.layerBytes)
                                func slice(_ x: [Float], _ cols: Int) -> Native.TensorRef {
                                    RefTest.tensor(ctx.device, floats: Array(x[0 ..< n * cols]), rows: n, cols: cols)
                                }
                                let out = zeros(n, c.ssmInnerSize)
                                try RefTest.run(ctx) {
                                    try gdn.encodePrefill($0, weights: weights, qkv: slice(qkv, c.ssmConvChannels),
                                                          z: slice(z, c.ssmInnerSize), alpha: slice(alpha, c.ssmTimeStepRank),
                                                          beta: slice(beta, c.ssmTimeStepRank), stateIn: banks.input,
                                                          stateOut: banks.output, output: out)
                                }
                                return RefTest.floats(out)
                            }
                            // State before a captured decode step = state after its position's rows.
                            // A capture with no state steps has no state tensors:
                            // skip only those comparisons (coverage is asserted
                            // across the prompt set below).
                            let stateRef = GoldenTapMap.graphTensor(.gdnState, layer: layer)
                            let statePositions = golden.entry(named: stateRef.name, occurrence: stateRef.occurrence) == nil
                                ? [] : try golden.tensor(.gdnState, layer: layer).positions
                            for position in statePositions where position <= rows {
                                _ = try runGDN(position)
                                let layout = kv.recurrentLayout
                                let base = banks.output.buffer.contents().advanced(by: banks.output.offset)
                                let conv = Array(UnsafeBufferPointer(start: base.assumingMemoryBound(to: Float.self),
                                                                     count: (layout.convKernel - 1) * layout.convChannels))
                                let ssm = Array(UnsafeBufferPointer(
                                    start: base.advanced(by: layout.ssmOffset).assumingMemoryBound(to: Float.self),
                                    count: layout.ssmBytes / 4))
                                if let want = try golden.convState(layer: layer, position: position, taps: layout.convKernel - 1) {
                                    record("gdn.conv_state", layer, conv, want)
                                }
                                if let want = try golden.ssmState(layer: layer, position: position) {
                                    record("gdn.ssm_state", layer, ssm, want)
                                }
                            }
                            let final = try load(.gdnOutput, layer)
                            record("gdn", layer, try runGDN(rows), final)
                            record("gemm.ssm_out", layer, try matmul(p + "ssm_out.weight", final),
                                   try tap(.mixerOutput(layer: layer)))
                        }
                        let post = try load(.ffnInput, layer)
                        let gate = try load(.ffnGate, layer)
                        let up = try load(.ffnUp, layer)
                        record("gemm.ffn_gate", layer, try matmul(p + "ffn_gate.weight", post), gate)
                        record("gemm.ffn_up", layer, try matmul(p + "ffn_up.weight", post), up)
                        let sw = zeros(rows, c.ffnSize)
                        try RefTest.run(ctx) {
                            try glue.encodeSwiGLU($0, gate: RefTest.tensor(ctx.device, floats: gate, rows: rows, cols: c.ffnSize),
                                                  up: RefTest.tensor(ctx.device, floats: up, rows: rows, cols: c.ffnSize), output: sw)
                        }
                        let swiglu = try load(.ffnSwiGLU, layer)
                        record("swiglu", layer, RefTest.floats(sw), swiglu)
                        record("gemm.ffn_down", layer, try matmul(p + "ffn_down.weight", swiglu), try load(.ffnDown, layer))
                        previous = try tap(.layerOutput(layer: layer))
                    }
                }
                print("reference components \(golden.id): \(rows) rows checked")
            }
        }
        for (component, value) in worst.sorted(by: { $0.key < $1.key }) {
            print("reference component worst relative L2 \(component): \(value)")
        }
        try Self.writeReport(reports, name: "components.json")
        print("reference components: prompts not captured from position 0 (skipped): \(notFullyCaptured)")
        // Every component of every layer of its kind was checked on at least one prompt.
        var covered: [String: Set<Int>] = [:]
        for r in reports { covered[r.component, default: []].insert(r.layer) }
        let attention = Set(c.fullAttentionLayers), gdn = Set(c.gatedDeltaNetLayers), all = Set(0..<c.blockCount)
        let expected: [String: Set<Int>] = [
            "get_rows": [-1], "rms_norm": all, "swiglu": all,
            "gemm.ffn_gate": all, "gemm.ffn_up": all, "gemm.ffn_down": all,
            "gemm.attn_q": attention, "gemm.attn_k": attention, "gemm.attn_v": attention, "attention": attention,
            "gemm.attn_output": attention,
            "gemm.attn_qkv": gdn, "gemm.attn_gate": gdn, "gemm.ssm_alpha": gdn, "gemm.ssm_beta": gdn, "gdn": gdn,
            "gdn.conv_state": gdn, "gdn.ssm_state": gdn, "gemm.ssm_out": gdn,
        ]
        for (component, layers) in expected {
            #expect(covered[component] == layers, "\(component) covered \(covered[component].map { $0.sorted() } ?? [])")
        }
        #expect(!reports.isEmpty)
        // Each component within its class's local relative-L2 bound
        // (ParityHarness.localVerdict, docs/NATIVE_PARITY.md).
        #expect(failures.isEmpty, "\(failures)")
    }
}
