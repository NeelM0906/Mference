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
        var agreements: Int
        var greedyAgreement: Double
        var meanKL: Double
        var p99KL: Double
        var maxKL: Double
        var firstDisagreement: Int?
        var seconds: Double
    }

    /// KL(golden ‖ ours) in F64 and our argmax (ties to the lowest id).
    static func klAndArgmax(golden: UnsafeBufferPointer<Float>, ours: UnsafeBufferPointer<Float>) -> (Double, Int) {
        var gMax = -Double.infinity, oMax = -Double.infinity
        var arg = 0
        for i in 0..<ours.count {
            gMax = max(gMax, Double(golden[i]))
            if Double(ours[i]) > oMax { oMax = Double(ours[i]); arg = i }
        }
        var gSum = 0.0, oSum = 0.0
        for i in 0..<ours.count {
            gSum += exp(Double(golden[i]) - gMax)
            oSum += exp(Double(ours[i]) - oMax)
        }
        let gLog = log(gSum), oLog = log(oSum)
        var kl = 0.0
        for i in 0..<ours.count {
            let lg = Double(golden[i]) - gMax - gLog
            let lo = Double(ours[i]) - oMax - oLog
            kl += exp(lg) * (lg - lo)
        }
        return (max(kl, 0), arg)
    }

    @Test(.enabled(if: enabled))
    func teacherForcedD1() throws {
        let ctx = try MetalContext()
        let pack = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let chunk = Int(Self.env["MFERENCE_NATIVE_PREFILL_CHUNK"] ?? "") ?? 256
        var reports: [PromptReport] = []
        var agree = 0, total = 0
        for golden in try Self.prompts() {
            // Every model and KV arena is freed before the next prompt: buffers
            // from makeBuffer are autoreleased and would otherwise accumulate.
            try autoreleasepool {
                let start = Date()
                let prompt = golden.promptTokens
                let generated = golden.generatedTokens
                let argmax = golden.manifest.argmaxTokens ?? generated
                let logits = try golden.logits()
                #expect(logits.rows == generated.count && logits.positions.first == prompt.count - 1)
                let positions = prompt.count + generated.count
                let model = try ReferenceQuickTests.makeModel(ctx: ctx, packURL: pack,
                                                              context: (positions + 63) / 64 * 64 + 64, chunk: chunk)
                var kls: [Double] = []
                var agreements = 0
                var firstMiss: Int?
                func compare(_ ours: Native.TensorRef, row: Int) {
                    let p = ours.buffer.contents().advanced(by: ours.offset).assumingMemoryBound(to: Float.self)
                    let (kl, arg) = logits.withRow(row) {
                        Self.klAndArgmax(golden: $0, ours: UnsafeBufferPointer(start: p, count: ours.cols))
                    }
                    kls.append(kl)
                    if Int32(arg) == argmax[row] { agreements += 1 } else if firstMiss == nil { firstMiss = row }
                }
                func step(_ kind: Native.StepKind, _ tokens: [Int32], at position: Int,
                          logits: Native.LogitRows) throws -> Native.TensorRef? {
                    try model.kv.beginStep(rows: tokens.count, kind: kind)
                    var out: Native.TensorRef?
                    try RefTest.run(ctx) { enc in
                        out = try model.encodeStep(Native.StepPlan(kind: kind, startPosition: position, rows: tokens.count,
                                                                   input: .host(tokens), logits: logits), encoder: enc)
                    }
                    return out
                }
                var position = 0
                while position < prompt.count {
                    let n = min(chunk, prompt.count - position)
                    let last = position + n == prompt.count
                    let out = try step(.prefill, Array(prompt[position ..< position + n]), at: position,
                                       logits: last ? .last : .none)
                    if last { compare(try #require(out), row: 0) }
                    position += n
                }
                for g in 1 ..< generated.count {
                    compare(try #require(try step(.decode, [generated[g - 1]], at: position, logits: .last)), row: g)
                    position += 1
                }
                let sorted = kls.sorted()
                let report = PromptReport(
                    prompt: golden.id, promptTokens: prompt.count, positions: kls.count, agreements: agreements,
                    greedyAgreement: Double(agreements) / Double(kls.count),
                    meanKL: kls.reduce(0, +) / Double(kls.count),
                    p99KL: sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.99))], maxKL: sorted.last ?? 0,
                    firstDisagreement: firstMiss, seconds: Date().timeIntervalSince(start))
                print("reference D1 \(report.prompt): agreement \(agreements)/\(kls.count) mean KL \(report.meanKL) "
                      + "p99 KL \(report.p99KL) max KL \(report.maxKL) first miss \(String(describing: firstMiss)) "
                      + "(\(Int(report.seconds)) s)")
                reports.append(report)
                try Self.writeReport(reports, name: "d1-teacher-forced.json")
                agree += agreements
                total += kls.count
            }
        }
        let overall = Double(agree) / Double(max(total, 1))
        print("reference D1 overall: greedy agreement \(agree)/\(total) = \(overall)")
        #expect(overall >= 0.99)
    }

    // MARK: Per-component goldens

    struct ComponentReport: Codable {
        var prompt: String
        var component: String
        var layer: Int
        var maxAbsError: Double
        var maxRelativeError: Double
        var meanAbsError: Double
    }

    static func errors(_ ours: [Float], _ golden: [Float]) -> (Double, Double, Double) {
        var maxAbs = 0.0, sumAbs = 0.0, scale = 0.0
        for (a, b) in zip(ours, golden) {
            let e = abs(Double(a) - Double(b))
            maxAbs = max(maxAbs, e)
            sumAbs += e
            scale = max(scale, abs(Double(b)))
        }
        return (maxAbs, maxAbs / max(scale, 1e-12), sumAbs / Double(max(ours.count, 1)))
    }

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
        for golden in try Self.prompts() {
            let embed = try golden.tensor(for: .embedding, config: c)
            let rows = embed.rows
            guard embed.positions == Array(0..<rows), rows <= 512 else { continue }
            try autoreleasepool {
                func all(_ m: GoldenMatrix) -> [Float] { (0..<m.rows).flatMap { m.row($0) } }
                func load(_ b: GoldenTapMap.Boundary, _ layer: Int) throws -> [Float] { all(try golden.tensor(b, layer: layer)) }
                func tap(_ p: Native.TapPoint) throws -> [Float] { all(try golden.tensor(for: p, config: c)) }
                func record(_ component: String, _ layer: Int, _ ours: [Float], _ want: [Float]) {
                    let (a, r, m) = Self.errors(ours, want)
                    reports.append(ComponentReport(prompt: golden.id, component: component, layer: layer,
                                                   maxAbsError: a, maxRelativeError: r, meanAbsError: m))
                    worst[component] = max(worst[component] ?? 0, r)
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
                            for position in try golden.tensor(.gdnState, layer: layer).positions where position <= rows {
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
            print("reference component worst relative error \(component): \(value)")
        }
        try Self.writeReport(reports, name: "components.json")
        #expect(!reports.isEmpty)
        for (component, value) in worst { #expect(value < 2e-2, "\(component) worst relative error \(value)") }
    }
}
