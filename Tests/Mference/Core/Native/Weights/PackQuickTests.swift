import Foundation
import Jinja
import Metal
import Testing
@testable import Mference
@testable import MferenceRepackCore

/// W1 acceptance on the real models, env-gated (none of these run by default):
///
/// - `MFERENCE_PACK_WRITE=<gguf>|<output.mfpack>|<label>`: write a pack with
///   the production writer (`MFERENCE_PACK_RESUME=1` resumes a partial one).
/// - `MFERENCE_PACK_ROUNDTRIP=<pack>|<gguf>`: every tensor unpacked and
///   compared byte for byte with its GGUF blocks; the pack opened with every
///   hash checked; tokenizer data, template and config against the GGUF's
///   metadata. With `MFERENCE_PACK_REFERENCE=<reference pack>`, `weights.bin`
///   and `tokenizer.json` are compared byte for byte with the reference
///   writer's, and the manifests must be equal up to the label.
/// - `MFERENCE_NATIVE_QUICK_PACK=<pack>` with `MFERENCE_NATIVE_GOLDENS`:
///   the pack tokenizer against llama.cpp's ids on the 11 golden prompts and
///   against the reference tokenizer, the template rendered against
///   llama.cpp's rendering, and the store's heap footprint.
/// - `MFERENCE_W1_D1=1` with the two above: D1 of the reference `qwen35` path
///   on the W1 store (`ParityHarness`, `ParityTolerances.d1`), every golden
///   prompt within the 8K evaluation cap; reports to `MFERENCE_W1_OUT`.
@Suite("PackQuickTests", .serialized)
struct PackQuickTests {
    static let env = ProcessInfo.processInfo.environment

    static func parts(_ key: String, _ count: Int) throws -> [String] {
        let p = try #require(env[key]).split(separator: "|").map(String.init)
        try #require(p.count == count, "\(key) needs \(count) |-separated fields")
        return p
    }

    @Test(.enabled(if: env["MFERENCE_PACK_WRITE"] != nil))
    func writePack() throws {
        let p = try Self.parts("MFERENCE_PACK_WRITE", 3)
        let before = RefTest.physicalFootprint()
        let start = Date()
        let result = try MFPackWriter.write(
            gguf: p[0], output: p[1],
            options: MFPackWriteOptions(label: p[2], resume: Self.env["MFERENCE_PACK_RESUME"] == "1",
                                        log: { print("pack writer: \($0)") }))
        let seconds = Date().timeIntervalSince(start)
        let peak = RefTest.physicalFootprint()
        print("pack writer: \(result.manifest.tensors.count) tensors, \(result.receipt.weightsByteCount) bytes, "
              + "written \(result.writtenBytes), resumed \(result.resumedTensors) tensors, \(Int(seconds)) s, "
              + "scratch high water \(result.scratchHighWaterBytes) bytes, footprint \(before >> 20) -> \(peak >> 20) MiB")
        #expect(result.scratchHighWaterBytes <= 64 << 20)
    }

    @Test(.enabled(if: env["MFERENCE_PACK_ROUNDTRIP"] != nil))
    func roundTripAgainstTheGGUF() throws {
        let p = try Self.parts("MFERENCE_PACK_ROUNDTRIP", 2)
        let pack = p[0], gguf = p[1]
        var start = Date()
        let report = try MFPackVerifier.roundTrip(pack: pack, gguf: gguf)
        print("round trip: \(report.tensors) tensors, \(report.bytes) bytes byte-exact to the GGUF "
              + "(\(report.tiledTensors) tiled) in \(Int(Date().timeIntervalSince(start))) s")
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        start = Date()
        let store = try Native.PackWeightStore(directory: URL(fileURLWithPath: pack), device: device,
                                               verification: .packedSHA256)
        print("round trip: store opened with every packed hash checked in \(Int(Date().timeIntervalSince(start))) s")
        let source = try Native.ReferenceGGUFFile(url: URL(fileURLWithPath: gguf))
        #expect(store.manifest.tensors.count == report.tensors)
        #expect(try source.qwen35Config() == store.config)
        #expect(try source.tokenizerData() == store.tokenizer.data)
        #expect(Array((source.string("tokenizer.chat_template") ?? "").utf8) == Array(store.chatTemplate.template.utf8))
        #expect(source.scalarMetadataStrings() == store.manifest.metadata)
        let ggufSHA = Native.ReferencePackWriter.sha256Hex(source.fileBytes)
        #expect(ggufSHA == store.manifest.source.sha256)
        print("round trip: config \(store.config)")

        if let refPath = Self.env["MFERENCE_PACK_REFERENCE"] {
            let ref = URL(fileURLWithPath: refPath), ours = URL(fileURLWithPath: pack)
            let a = try Native.PackMappedFile(url: ours.appendingPathComponent("weights.bin"))
            let b = try Native.PackMappedFile(url: ref.appendingPathComponent("weights.bin"))
            #expect(a.size == b.size)
            var equal = a.size == b.size
            var offset = 0
            while equal && offset < a.size {
                let n = min(64 << 20, a.size - offset)
                equal = memcmp(a.base + offset, b.base + offset, n) == 0
                offset += n
            }
            #expect(equal, "weights.bin differs from the reference writer's")
            let tokA = try Data(contentsOf: ours.appendingPathComponent("tokenizer.json"))
            let tokB = try Data(contentsOf: ref.appendingPathComponent("tokenizer.json"))
            #expect(tokA == tokB)
            let refManifest = try JSONDecoder().decode(Native.PackManifest.self,
                                                       from: Data(contentsOf: ref.appendingPathComponent("manifest.json")))
            var relabelled = store.manifest
            relabelled.label = refManifest.label
            #expect(relabelled == refManifest)
            print("round trip: weights.bin (\(a.size) bytes), tokenizer.json and manifest match the reference writer's")
        }
    }

    // MARK: Tokenizer and template on the real vocabulary

    static let goldenEnabled = env["MFERENCE_NATIVE_QUICK_PACK"] != nil && GoldenStore.root != nil

    /// The fixture's prompt entries (messages, tools, thinking), as JSON objects.
    static func fixturePrompts() throws -> [String: [String: Any]] {
        let url = GoldenStore.fixturesDirectory.appendingPathComponent("prompts/quick-v1.json")
        let root = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let prompts = try #require(root["prompts"] as? [[String: Any]])
        var out: [String: [String: Any]] = [:]
        for p in prompts { out[p["id"] as! String] = p }
        return out
    }

    /// A JSON value as a template value; object keys sorted (llama.cpp's
    /// `nlohmann::ordered_json` keeps the fixture's sorted order).
    static func jinja(_ any: Any) throws -> Value {
        switch any {
        case let s as String: return .string(s)
        case let n as NSNumber:
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return .boolean(n.boolValue) }
            if CFNumberIsFloatType(n) { return .double(n.doubleValue) }
            return .int(n.intValue)
        case let a as [Any]: return .array(try a.map(jinja))
        case let d as [String: Any]:
            var o: OrderedDictionary<String, Value> = [:]
            for k in d.keys.sorted() { o[k] = try jinja(d[k]!) }
            return .object(o)
        case is NSNull: return .null
        default: throw Native.NativeError.unsupported("JSON value \(any)")
        }
    }

    /// Renders a fixture prompt as llama-server does: messages (tool-call
    /// arguments parsed to objects), tools as `{type, function: {name,
    /// description, parameters}}`, `add_generation_prompt`, `enable_thinking`,
    /// BOS/EOS texts; `lstrip_blocks` and `trim_blocks`.
    static func render(_ prompt: [String: Any], template: Native.ChatTemplateSource) throws -> String {
        let t = try Template(template.template, with: .init(lstripBlocks: true, trimBlocks: true))
        let messages = try (prompt["messages"] as? [[String: Any]] ?? []).map { m -> Value in
            var o: OrderedDictionary<String, Value> = [:]
            for key in ["role", "content"] { if let v = m[key] { o[key] = try jinja(v) } }
            if let calls = m["tool_calls"] as? [[String: Any]] {
                o["tool_calls"] = .array(try calls.map { call in
                    let f = call["function"] as! [String: Any]
                    var args = f["arguments"] as Any
                    if let s = args as? String { args = try JSONSerialization.jsonObject(with: Data(s.utf8)) }
                    let function: OrderedDictionary<String, Value> = ["name": try jinja(f["name"]!),
                                                                      "arguments": try jinja(args)]
                    let wrapper: OrderedDictionary<String, Value> = ["id": try jinja(call["id"] ?? ""),
                                                                     "type": .string("function"),
                                                                     "function": .object(function)]
                    return .object(wrapper)
                })
            }
            for key in ["tool_call_id", "name", "reasoning_content"] { if let v = m[key] { o[key] = try jinja(v) } }
            return .object(o)
        }
        var context: [String: Value] = ["messages": .array(messages), "add_generation_prompt": .boolean(true),
                                        "enable_thinking": .boolean(prompt["enable_thinking"] as? Bool ?? false)]
        if let tools = prompt["tools"] as? [[String: Any]], !tools.isEmpty {
            context["tools"] = .array(try tools.map { tool in
                let f = tool["function"] as! [String: Any]
                let function: OrderedDictionary<String, Value> = [
                    "name": try jinja(f["name"]!), "description": try jinja(f["description"] ?? ""),
                    "parameters": try jinja(f["parameters"] ?? [String: Any]())]
                return .object(["type": .string("function"), "function": .object(function)])
            })
        }
        if let bos = template.bosToken { context["bos_token"] = .string(bos) }
        if let eos = template.eosToken { context["eos_token"] = .string(eos) }
        return try t.render(context)
    }

    @Test(.enabled(if: goldenEnabled))
    func tokenizerAndTemplateMatchLlamaCpp() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let pack = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let before = RefTest.physicalFootprint()
        var start = Date()
        let store = try Native.PackWeightStore(directory: pack, device: device)
        let afterOpen = RefTest.physicalFootprint()
        let tokenizer = try (store.tokenizer as! Native.PackTokenizerSource).packTokenizer()
        let built = Date().timeIntervalSince(start)
        let afterTokenizer = RefTest.physicalFootprint()
        print("pack store: opened \(store.manifest.tensors.count) tensors, \(store.mappedByteCount >> 20) MiB mapped; "
              + "footprint \(before >> 20) -> \(afterOpen >> 20) MiB (open), \(afterTokenizer >> 20) MiB (tokenizer), "
              + "\(String(format: "%.2f", built)) s")
        // Bounded heap: the 2.8 GB of weights stay mapped, not copied.
        #expect(afterOpen - min(before, afterOpen) < 256 << 20)
        #expect(afterTokenizer - min(before, afterTokenizer) < 512 << 20)
        #expect(tokenizer.vocabSize == 248_320 && tokenizer.eosTokenID == 248_046)

        start = Date()
        let reference = try Native.ReferenceTokenizer(data: store.tokenizer.data)
        print("reference tokenizer built in \(String(format: "%.2f", Date().timeIntervalSince(start))) s")
        #expect(tokenizer.endOfGenerationTokenIDs == reference.endOfGenerationTokenIDs)
        #expect(tokenizer.specialTokens == reference.specialTokens)

        let fixtures = try Self.fixturePrompts()
        var checked = 0, rendered = 0
        var oursSeconds = 0.0, refSeconds = 0.0, tokens = 0
        for prompt in try GoldenStore.promptSet().prompts {
            let golden = try GoldenStore.open(prompt.id)
            let text = try golden.renderedPrompt()
            start = Date()
            let ids = try tokenizer.encode(text, addSpecial: true, parseSpecial: true)
            oursSeconds += Date().timeIntervalSince(start)
            start = Date()
            let refIDs = try reference.encode(text, addSpecial: true, parseSpecial: true)
            refSeconds += Date().timeIntervalSince(start)
            tokens += ids.count
            #expect(ids == golden.promptTokens, "\(golden.id): \(ids.count) vs \(golden.promptTokens.count) tokens")
            #expect(ids == refIDs, "\(golden.id): differs from the reference tokenizer")
            #expect(ids.flatMap { tokenizer.piece($0, special: true) } == Array(text.utf8), "\(golden.id) pieces")
            #expect(golden.generatedTokens.flatMap { tokenizer.piece($0, special: false) }
                    == golden.generatedTokens.flatMap { reference.piece($0, special: false) }, "\(golden.id) output")
            checked += 1
            let fixture = try #require(fixtures[prompt.id])
            let ours = try Self.render(fixture, template: store.chatTemplate)
            #expect(Array(ours.utf8) == Array(text.utf8), "\(golden.id): rendered template differs from llama.cpp's")
            if Array(ours.utf8) == Array(text.utf8) { rendered += 1 }
        }
        print("pack tokenizer: \(checked) golden prompts match llama.cpp (\(tokens) tokens; "
              + "\(String(format: "%.3f", oursSeconds)) s vs reference \(String(format: "%.3f", refSeconds)) s); "
              + "\(rendered) rendered templates byte-identical")
        #expect(checked == 11 && rendered == 11)

        // Fuzz on the real vocabulary against the reference.
        var mismatches = 0
        for text in PackTokenizerTests.fuzz(400, seed: 99) {
            for (a, p) in [(false, false), (true, true)] where
                try tokenizer.encode(text, addSpecial: a, parseSpecial: p)
                != reference.encode(text, addSpecial: a, parseSpecial: p) {
                mismatches += 1
                print("mismatch: \(text.debugDescription)")
            }
        }
        #expect(mismatches == 0)
        var pieceMismatches = 0
        for t in stride(from: Int32(0), to: Int32(tokenizer.vocabSize), by: 1) {
            if tokenizer.piece(t, special: true) != reference.piece(t, special: true)
                || tokenizer.piece(t, special: false) != reference.piece(t, special: false) {
                pieceMismatches += 1
            }
        }
        #expect(pieceMismatches == 0)
        withExtendedLifetime(store) {}
    }

    // MARK: D1 on the W1 store

    struct PromptReport: Codable {
        var prompt: String
        var promptTokens: Int
        var positions: Int
        var agreements: Int
        var greedyAgreement: Double
        var meanKL: Double
        var p99KL: Double
        var maxKL: Double
        var disagreements: [Int]
        var seconds: Double
    }

    struct D1Summary: Codable {
        var head: String
        var store: String
        var prompts: [String]
        var skipped: [String]
        var positions: Int
        var agreements: Int
        var greedyAgreement: Double
        var meanKL: Double
        var p99KL: Double
        var maxKL: Double
        var failures: [String]
        var passed: Bool
    }

    static func writeReport<T: Encodable>(_ value: T, name: String) throws {
        guard let out = env["MFERENCE_W1_OUT"] else { return }
        try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        try PackMirrorPinTests.encode(value).write(to: URL(fileURLWithPath: out).appendingPathComponent(name))
    }

    /// Teacher-forced D1 of `ModelComponents(weights: PackWeightStore, every
    /// other component reference)`, scored like the Phase 0 reference run.
    /// Prompts whose positions exceed the 8K evaluation cap (A5) are skipped.
    @Test(.enabled(if: goldenEnabled && env["MFERENCE_W1_D1"] == "1"))
    func teacherForcedD1OnTheW1Store() throws {
        let ctx = try MetalContext()
        let pack = URL(fileURLWithPath: try #require(Self.env["MFERENCE_NATIVE_QUICK_PACK"]))
        let chunk = Int(Self.env["MFERENCE_NATIVE_PREFILL_CHUNK"] ?? "") ?? 256
        let cap = 8192
        let only = Self.env["MFERENCE_NATIVE_GOLDEN_PROMPTS"].map { Set($0.split(separator: ",").map(String.init)) }
        var reports: [PromptReport] = []
        var results: [ParityHarness.LogitResult] = []
        var skipped: [String] = []
        for prompt in try GoldenStore.promptSet().prompts where only?.contains(prompt.id) ?? true {
            let golden = try GoldenStore.open(prompt.id)
            let positions = golden.promptTokens.count + golden.generatedTokens.count
            guard positions <= cap else {
                print("W1 D1 \(golden.id): \(positions) positions, over the 8K cap; skipped")
                skipped.append(golden.id)
                continue
            }
            try autoreleasepool {
                let start = Date()
                let store = try Native.PackWeightStore(directory: pack, device: ctx.device)
                let factory = Native.ReferenceForwardModelFactory(context: ctx)
                let c = store.config
                let context = (positions + 63) / 64 * 64 + 64
                let plan = try Native.ReferenceMemoryPlanner().resolve(
                    config: c, weightsMapped: store.mappedByteCount, contextLength: context, precision: .f16,
                    maxVerifyRows: 1, maxPrefillChunk: chunk,
                    activationBytes: factory.activationBytes(config: c, maxPrefillChunk: chunk, maxVerifyRows: 1),
                    prefixCacheRAMBudget: 1 << 28, budget: 16 << 30)
                let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
                let model = try factory.make(Native.ModelComponents(
                    device: ctx.device, weights: store, gemm: Native.ReferenceQuantGEMM(context: ctx),
                    attention: Native.ReferenceAttentionOp(context: ctx), gdn: Native.ReferenceGDNOp(context: ctx),
                    glue: Native.ReferenceGlueOps(context: ctx), kv: kv), plan: plan)
                let promptTokens = golden.promptTokens, generated = golden.generatedTokens
                var position = 0
                func step(_ kind: Native.StepKind, _ tokens: [Int32], logits: Native.LogitRows) throws -> [Float]? {
                    try autoreleasepool {
                        try model.kv.beginStep(rows: tokens.count, kind: kind)
                        var out: Native.TensorRef?
                        try RefTest.run(ctx) { enc in
                            out = try model.encodeStep(Native.StepPlan(kind: kind, startPosition: position,
                                                                       rows: tokens.count, input: .host(tokens),
                                                                       logits: logits), encoder: enc)
                        }
                        position += tokens.count
                        return out.map(RefTest.floats)
                    }
                }
                let result = try ParityHarness.compareLogits(golden: golden) { g in
                    var out: [Float]?
                    if g == 0 {
                        while position < promptTokens.count {
                            let n = min(chunk, promptTokens.count - position)
                            let last = position + n == promptTokens.count
                            out = try step(.prefill, Array(promptTokens[position ..< position + n]),
                                           logits: last ? .last : .none)
                            print("W1 D1 \(golden.id): prefill to \(position)/\(positions), "
                                  + "\(Int(Date().timeIntervalSince(start))) s")
                        }
                    } else {
                        out = try step(.decode, [generated[g - 1]], logits: .last)
                        if g % 32 == 0 {
                            print("W1 D1 \(golden.id): decode \(g)/\(generated.count), "
                                  + "\(Int(Date().timeIntervalSince(start))) s")
                        }
                    }
                    return out
                }
                let d = ParityMetrics.Distribution(result.kld)
                let report = PromptReport(prompt: golden.id, promptTokens: promptTokens.count,
                                          positions: result.positions, agreements: result.agreements,
                                          greedyAgreement: result.agreement, meanKL: d.mean, p99KL: d.p99, maxKL: d.max,
                                          disagreements: result.disagreements, seconds: Date().timeIntervalSince(start))
                print("W1 D1 \(golden.id): agreement \(result.agreements)/\(result.positions) mean KL \(d.mean) "
                      + "p99 \(d.p99) max \(d.max) (\(Int(report.seconds)) s)")
                reports.append(report)
                results.append(result)
                try Self.writeReport(reports, name: "w1-d1-teacher-forced.json")
            }
        }
        let report = ParityHarness.LogitReport(results: results)
        let verdict = report.verdict(.d1)
        let d = report.kld
        print(report.table(title: "reference qwen35 on the W1 store vs quick-v1/metal"))
        try Self.writeReport(D1Summary(
            head: Self.env["MFERENCE_NATIVE_HEAD"] ?? "", store: pack.path, prompts: reports.map(\.prompt),
            skipped: skipped, positions: report.positions, agreements: report.agreements,
            greedyAgreement: report.agreement, meanKL: d.mean, p99KL: d.p99, maxKL: d.max,
            failures: verdict.failures, passed: verdict.passed), name: "w1-d1-summary.json")
        #expect(verdict.passed, "\(verdict.failures)")
    }
}
