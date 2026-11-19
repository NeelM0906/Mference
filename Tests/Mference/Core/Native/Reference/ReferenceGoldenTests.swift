import Foundation
import Metal
import Testing
@testable import Mference

/// The all-reference Quick path and each reference component against the
/// llama.cpp goldens (`mfe-golden-v1`, written by the harness under
/// `$PROGRAM/goldens/<set>/<backend>/<prompt>/`). Model runs, skipped unless
/// both are set:
///
/// - `MFERENCE_NATIVE_QUICK_PACK`: the Quick `.mfpack`
/// - `MFERENCE_NATIVE_GOLDENS`: one golden backend directory (e.g.
///   `.../goldens/quick-v1/metal`)
///
/// Optional: `MFERENCE_NATIVE_GOLDEN_PROMPTS` (comma-separated prompt ids),
/// `MFERENCE_NATIVE_PREFILL_CHUNK` (default 256) and
/// `MFERENCE_NATIVE_REFERENCE_OUT` (directory for the JSON report).
///
/// D1 here is measured teacher-forced: the golden's generated tokens are fed
/// back, and at every position the reference's logits are compared with the
/// golden's (greedy agreement with `argmax_tokens`, KL(golden ‖ reference)).
@Suite("ReferenceGoldenTests", .serialized)
struct ReferenceGoldenTests {
    static let env = ProcessInfo.processInfo.environment
    static let enabled = env["MFERENCE_NATIVE_QUICK_PACK"] != nil && env["MFERENCE_NATIVE_GOLDENS"] != nil

    struct Golden: Decodable {
        struct Logits: Decodable {
            var file: String
            var rows: Int
            var cols: Int
            var position_ranges: [[Int]]
        }
        struct Tensor: Decodable {
            var key: String
            var base: String
            var layer: Int
            var occurrence: Int
            var cols: Int
            var rows: Int
            var position_ranges: [[Int]]
            var file: String
        }
        var prompt_id: String
        var prompt_tokens: [Int32]
        var generated_tokens: [Int32]
        var argmax_tokens: [Int32]
        var logits: Logits
        var tensors: [Tensor]
    }

    static func goldens() throws -> [(URL, Golden)] {
        let root = URL(fileURLWithPath: try #require(env["MFERENCE_NATIVE_GOLDENS"]))
        let only = env["MFERENCE_NATIVE_GOLDEN_PROMPTS"].map { Set($0.split(separator: ",").map(String.init)) }
        return try FileManager.default.contentsOfDirectory(atPath: root.path).sorted().compactMap { name in
            guard only?.contains(name) ?? true else { return nil }
            let dir = root.appendingPathComponent(name)
            let json = dir.appendingPathComponent("golden.json")
            guard FileManager.default.fileExists(atPath: json.path) else { return nil }
            return (dir, try JSONDecoder().decode(Golden.self, from: Data(contentsOf: json)))
        }
    }

    /// Rows of a golden f32 file.
    final class Matrix {
        let file: Native.ReferenceMappedFile
        let cols: Int
        init(_ url: URL, cols: Int) throws {
            file = try Native.ReferenceMappedFile(url: url)
            self.cols = cols
        }
        var rows: Int { file.size / (cols * 4) }
        func row(_ r: Int) -> UnsafeBufferPointer<Float> {
            UnsafeBufferPointer(start: file.base.advanced(by: r * cols * 4).assumingMemoryBound(to: Float.self),
                                count: cols)
        }
        func rows(_ range: Range<Int>) -> [Float] { range.flatMap { Array(row($0)) } }
    }

    struct PromptReport: Codable {
        var prompt: String
        var promptTokens: Int
        var positions: Int
        var greedyAgreement: Double
        var meanKL: Double
        var maxKL: Double
        var p99KL: Double
        var firstDisagreement: Int?
        var seconds: Double
    }

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
        for (dir, g) in try Self.goldens() {
            let start = Date()
            let positions = g.prompt_tokens.count + g.generated_tokens.count
            let model = try ReferenceQuickTests.makeModel(ctx: ctx, packURL: pack,
                                                          context: (positions + 63) / 64 * 64 + 64, chunk: chunk)
            let logits = try Matrix(dir.appendingPathComponent(g.logits.file), cols: g.logits.cols)
            #expect(logits.rows == g.logits.rows && g.logits.rows == g.generated_tokens.count)
            var kls: [Double] = []
            var agreements = 0
            var firstMiss: Int?
            func compare(_ ours: Native.TensorRef, row: Int) {
                let p = ours.buffer.contents().advanced(by: ours.offset).assumingMemoryBound(to: Float.self)
                let (kl, arg) = Self.klAndArgmax(golden: logits.row(row),
                                                 ours: UnsafeBufferPointer(start: p, count: ours.cols))
                kls.append(kl)
                if Int32(arg) == g.argmax_tokens[row] { agreements += 1 } else if firstMiss == nil { firstMiss = row }
            }
            var position = 0
            while position < g.prompt_tokens.count {
                let n = min(chunk, g.prompt_tokens.count - position)
                let last = position + n == g.prompt_tokens.count
                try model.kv.beginStep(rows: n)
                var out: Native.TensorRef?
                try RefTest.run(ctx) { enc in
                    out = try model.encodeStep(Native.StepPlan(
                        kind: .prefill, startPosition: position, rows: n,
                        input: .host(Array(g.prompt_tokens[position ..< position + n])),
                        logits: last ? .last : .none), encoder: enc)
                }
                if last, let out { compare(out, row: 0) }
                position += n
            }
            for i in 0 ..< g.generated_tokens.count - 1 {
                try model.kv.beginStep(rows: 1)
                var out: Native.TensorRef?
                try RefTest.run(ctx) { enc in
                    out = try model.encodeStep(Native.StepPlan(
                        kind: .decode, startPosition: model.kv.length - 1, rows: 1,
                        input: .host([g.generated_tokens[i]]), logits: .last), encoder: enc)
                }
                compare(try #require(out), row: i + 1)
            }
            let sorted = kls.sorted()
            let report = PromptReport(
                prompt: g.prompt_id, promptTokens: g.prompt_tokens.count, positions: kls.count,
                greedyAgreement: Double(agreements) / Double(kls.count),
                meanKL: kls.reduce(0, +) / Double(kls.count), maxKL: sorted.last ?? 0,
                p99KL: sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.99))],
                firstDisagreement: firstMiss, seconds: Date().timeIntervalSince(start))
            print("reference D1 \(report.prompt): agreement \(report.greedyAgreement) mean KL \(report.meanKL) "
                  + "max KL \(report.maxKL) p99 KL \(report.p99KL) first miss \(String(describing: firstMiss)) "
                  + "(\(Int(report.seconds)) s)")
            reports.append(report)
            agree += agreements
            total += kls.count
        }
        let overall = Double(agree) / Double(max(total, 1))
        print("reference D1 overall: greedy agreement \(overall) over \(total) positions")
        if let out = Self.env["MFERENCE_NATIVE_REFERENCE_OUT"] {
            let url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(reports).write(to: url.appendingPathComponent("d1-teacher-forced.json"))
        }
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

    /// Each reference component fed the golden's own inputs, compared with
    /// the golden's outputs, on prompts captured from position 0 with no gaps.
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
        for (dir, g) in try Self.goldens() {
            let byKey = Dictionary(uniqueKeysWithValues: g.tensors.map { ($0.key, $0) })
            guard let embed = byKey["model.input_embed"], embed.position_ranges == [[0, embed.rows]],
                  embed.rows <= 512 else { continue }
            let rows = embed.rows
            func load(_ key: String) throws -> [Float]? {
                guard let t = byKey[key] else { return nil }
                return try Matrix(dir.appendingPathComponent(t.file), cols: t.cols).rows(0 ..< rows)
            }
            func record(_ component: String, _ layer: Int, _ ours: [Float], _ golden: [Float]) {
                let (a, r, m) = Self.errors(ours, golden)
                reports.append(ComponentReport(prompt: g.prompt_id, component: component, layer: layer,
                                               maxAbsError: a, maxRelativeError: r, meanAbsError: m))
                worst[component] = max(worst[component] ?? 0, r)
            }
            func matmul(_ weight: String, _ input: [Float]) throws -> [Float] {
                let w = try store.tensor(weight)
                let x = RefTest.tensor(ctx.device, floats: input, rows: rows, cols: w.cols)
                let y = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * w.rows), rows: rows,
                                       cols: w.rows)
                try RefTest.run(ctx) { try gemm.encodePrefill($0, weight: w, input: x, output: y, mode: .overwrite) }
                return RefTest.floats(y)
            }
            // Embedding.
            let all = g.prompt_tokens + g.generated_tokens
            let tokens = Array(all.prefix(rows)).map { UInt32(bitPattern: $0) }
            let tokenBuffer = RefTest.buffer(ctx.device, bytes: tokens.withUnsafeBytes { Array($0) })
            let emb = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * c.hiddenSize), rows: rows,
                                     cols: c.hiddenSize)
            try RefTest.run(ctx) {
                try gemm.encodeGetRows($0, weight: try store.tensor("token_embd.weight"),
                                       tokens: Native.TokenSlot(buffer: tokenBuffer, offset: 0, count: rows), output: emb)
            }
            record("get_rows", -1, RefTest.floats(emb), try #require(try load("model.input_embed")))

            let attention = Native.ReferenceAttentionOp(context: ctx)
            let gdn = Native.ReferenceGDNOp(context: ctx)
            let plan = try Native.ReferenceMemoryPlanner().resolve(
                config: c, weightsMapped: 0, contextLength: (rows + 63) / 64 * 64, precision: .f16, maxVerifyRows: 1,
                maxPrefillChunk: rows, activationBytes: 0, prefixCacheRAMBudget: 0, budget: 1 << 34)
            let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
            try attention.prepare(config: c, geometry: kv.geometry)
            try gdn.prepare(config: c, layout: kv.recurrentLayout)
            try kv.beginStep(rows: rows)
            var previous = try #require(try load("model.input_embed"))
            for layer in 0..<c.blockCount {
                let p = "blk.\(layer)."
                guard let normIn = try load("attn_norm-\(layer)") else { continue }
                // Glue: RMSNorm of the previous layer's output.
                let x = RefTest.tensor(ctx.device, floats: previous, rows: rows, cols: c.hiddenSize)
                let h = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * c.hiddenSize), rows: rows,
                                       cols: c.hiddenSize)
                try RefTest.run(ctx) { try glue.encodeRMSNorm($0, input: x, weight: try store.tensor(p + "attn_norm.weight"), output: h) }
                record("rms_norm", layer, RefTest.floats(h), normIn)
                if c.layerKind(layer) == .fullAttention {
                    let qg = try #require(try load("Qcur_full-\(layer)"))
                    let k = try #require(try load("Kcur-\(layer)"))
                    let v = try #require(try load("Vcur-\(layer)"))
                    record("gemm.attn_q", layer, try matmul(p + "attn_q.weight", normIn), qg)
                    record("gemm.attn_k", layer, try matmul(p + "attn_k.weight", normIn), k)
                    record("gemm.attn_v", layer, try matmul(p + "attn_v.weight", normIn), v)
                    let out = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * 4096), rows: rows,
                                             cols: c.headCount * c.valueLength)
                    let attnLayer = c.fullAttentionLayers.firstIndex(of: layer)!
                    try RefTest.run(ctx) {
                        try attention.encodePrefill(
                            $0, attnLayer: attnLayer,
                            weights: .init(qNorm: try store.tensor(p + "attn_q_norm.weight"),
                                           kNorm: try store.tensor(p + "attn_k_norm.weight")),
                            qGate: RefTest.tensor(ctx.device, floats: qg, rows: rows, cols: qg.count / rows),
                            k: RefTest.tensor(ctx.device, floats: k, rows: rows, cols: k.count / rows),
                            v: RefTest.tensor(ctx.device, floats: v, rows: rows, cols: v.count / rows),
                            startPosition: 0, kv: kv.pageTable(), output: out)
                    }
                    let gated = try #require(try load("attn_gated-\(layer)"))
                    record("attention", layer, RefTest.floats(out), gated)
                    record("gemm.attn_output", layer, try matmul(p + "attn_output.weight", gated),
                           try #require(try load("attn_output-\(layer)")))
                } else {
                    let qkv = try #require(try load("linear_attn_qkv_mixed-\(layer)"))
                    let z = try #require(try load("z-\(layer)"))
                    let alpha = try #require(try load("alpha-\(layer)"))
                    let beta = try #require(try load("beta-\(layer)"))
                    record("gemm.attn_qkv", layer, try matmul(p + "attn_qkv.weight", normIn), qkv)
                    record("gemm.attn_gate", layer, try matmul(p + "attn_gate.weight", normIn), z)
                    record("gemm.ssm_alpha", layer, try matmul(p + "ssm_alpha.weight", normIn), alpha)
                    record("gemm.ssm_beta", layer, try matmul(p + "ssm_beta.weight", normIn), beta)
                    let gdnIndex = c.gatedDeltaNetLayers.firstIndex(of: layer)!
                    let banks = kv.recurrentState(gdnLayer: gdnIndex)
                    let out = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * c.ssmInnerSize),
                                             rows: rows, cols: c.ssmInnerSize)
                    try RefTest.run(ctx) {
                        try gdn.encodePrefill(
                            $0, weights: .init(conv1d: try store.tensor(p + "ssm_conv1d.weight"),
                                               dtBias: try store.tensor(p + "ssm_dt.bias"), a: try store.tensor(p + "ssm_a"),
                                               norm: try store.tensor(p + "ssm_norm.weight")),
                            qkv: RefTest.tensor(ctx.device, floats: qkv, rows: rows, cols: c.ssmConvChannels),
                            z: RefTest.tensor(ctx.device, floats: z, rows: rows, cols: c.ssmInnerSize),
                            alpha: RefTest.tensor(ctx.device, floats: alpha, rows: rows, cols: c.ssmTimeStepRank),
                            beta: RefTest.tensor(ctx.device, floats: beta, rows: rows, cols: c.ssmTimeStepRank),
                            stateIn: banks.input, stateOut: banks.output, output: out)
                    }
                    let final = try #require(try load("final_output-\(layer)"))
                    record("gdn", layer, RefTest.floats(out), final)
                    record("gemm.ssm_out", layer, try matmul(p + "ssm_out.weight", final),
                           try #require(try load("linear_attn_out-\(layer)")))
                }
                let post = try #require(try load("attn_post_norm-\(layer)"))
                let gate = try #require(try load("ffn_gate-\(layer)"))
                let up = try #require(try load("ffn_up-\(layer)"))
                record("gemm.ffn_gate", layer, try matmul(p + "ffn_gate.weight", post), gate)
                record("gemm.ffn_up", layer, try matmul(p + "ffn_up.weight", post), up)
                let sw = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: gate.count), rows: rows,
                                        cols: c.ffnSize)
                try RefTest.run(ctx) {
                    try glue.encodeSwiGLU($0, gate: RefTest.tensor(ctx.device, floats: gate, rows: rows, cols: c.ffnSize),
                                          up: RefTest.tensor(ctx.device, floats: up, rows: rows, cols: c.ffnSize), output: sw)
                }
                let swiglu = try #require(try load("ffn_swiglu-\(layer)"))
                record("swiglu", layer, RefTest.floats(sw), swiglu)
                record("gemm.ffn_down", layer, try matmul(p + "ffn_down.weight", swiglu),
                       try #require(try load("ffn_out-\(layer)")))
                previous = try #require(try load("l_out-\(layer)"))
            }
            print("reference components \(g.prompt_id): \(rows) rows checked")
        }
        for (component, value) in worst.sorted(by: { $0.key < $1.key }) {
            print("reference component worst relative error \(component): \(value)")
        }
        if let out = Self.env["MFERENCE_NATIVE_REFERENCE_OUT"] {
            let url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(reports).write(to: url.appendingPathComponent("components.json"))
        }
        #expect(!reports.isEmpty)
        for (component, value) in worst { #expect(value < 2e-2, "\(component) worst relative error \(value)") }
    }
}
