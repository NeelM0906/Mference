import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// Which precision costs routing agreement (G6c diagnosis): the CPU
/// reference on the real install, free-running (no teacher forcing) over one
/// of K0's CPU-only traces, once per precision mode. Each mode writes the
/// G6c candidate layout (`<out>/<mode>/tf/<id>/` log-probs at every
/// position after the final softcap, `<out>/<mode>/traces/<id>/` running
/// sums and expert sets), so the same tools compare it with llama.cpp's
/// goldens, and prints how many token-visits chose another expert set than
/// the CPU trace.
///
/// Modes (`MFERENCE_INKLING2_PRECISION_MODES`, comma separated): `fp32`,
/// `llamaCPU`, `llamaExperts`, `metalRunner`, `metalRunner-<site>` (every FP16 site of
/// `.metalRunner` but that one: projectionInputs, bf16Inputs, queries,
/// cache, attentionOutput, expertInput, expertActivation), and `margins`:
/// the reference at `.llamaCPU` teacher-forced on the trace (its records and
/// routes, as G6a runs), writing `<out>/margins/<id>.json` with its own
/// router's gap between the 8th and 9th selection score at every
/// token-visit of the trace's state.
///
/// Loads the install, so it runs only with `MFERENCE_INKLING2_GTURBO`,
/// `MFERENCE_INKLING2_GOLDENS_DIR` and `MFERENCE_INKLING2_PRECISION_OUT` set,
/// and then only inside `model-hold.sh`. `MFERENCE_INKLING2_PRECISION_TRACE`
/// (relative to the goldens folder) picks the prompt, `generate2/cpu/trace02`
/// by default: any golden folder; without a `trace.json` (a teacher-forced
/// prompt, `generate-tf/metal/<id>`) only the log-probs are written and
/// `margins` is refused. `llamaCPU` and `margins` need fewer than 32 tokens,
/// `llamaExperts` 32 or more; at most 512 tokens (the trace's arrays).
@Suite("Inkling2 precision and routing on the real install", .serialized)
struct Inkling2PrecisionRoutingTests {
    static let environment = ProcessInfo.processInfo.environment
    static var install: URL? { environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) } }
    static var goldens: URL? { environment["MFERENCE_INKLING2_GOLDENS_DIR"].map { URL(fileURLWithPath: $0) } }
    static var output: URL? { environment["MFERENCE_INKLING2_PRECISION_OUT"].map { URL(fileURLWithPath: $0) } }

    static let sites: [String: Inkling2ReferenceModel.MetalRunnerSites] = [
        "projectionInputs": .projectionInputs, "bf16Inputs": .bf16Inputs, "queries": .queries,
        "cache": .cache, "attentionOutput": .attentionOutput, "expertInput": .expertInput,
        "expertActivation": .expertActivation,
    ]

    struct Golden: Decodable { let prompt_ids: [Int32] }
    struct Trace: Decodable {
        struct Visit: Decodable { let experts: [[Int]] }
        let hidden_shape: [Int]
        let visits: [Visit]
    }
    struct Margins: Encodable {
        let trace: String
        let precision: String
        /// `[visit][token]`: the 8th minus the 9th selection score.
        let margins: [[Float]]
        /// `[visit][token]`: the trace's expert set below the reference's own top-8 cutoff.
        let shortfalls: [[Float]]
        /// `[visit][token]`: the reference's own top-8 differs from the trace's set.
        let ownDiffers: [[Bool]]
    }

    private func write<T: Encodable>(_ value: T, to url: URL) throws {
        try JSONEncoder().encode(value).write(to: url)
    }

    @Test(.enabled(if: install != nil && goldens != nil && output != nil,
                   "needs MFERENCE_INKLING2_GTURBO, MFERENCE_INKLING2_GOLDENS_DIR and MFERENCE_INKLING2_PRECISION_OUT"))
    func freeRunningRoutesPerPrecision() throws {
        let installURL = try #require(Self.install), goldensURL = try #require(Self.goldens)
        let outURL = try #require(Self.output)
        let traceName = Self.environment["MFERENCE_INKLING2_PRECISION_TRACE"] ?? "generate2/cpu/trace02"
        let modes = Self.environment["MFERENCE_INKLING2_PRECISION_MODES"]?.split(separator: ",").map(String.init)
            ?? ["fp32", "llamaCPU", "metalRunner", "margins"]
        let folder = goldensURL.appendingPathComponent(traceName)
        let id = folder.lastPathComponent
        let ids = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: folder.appendingPathComponent("golden.json")))
            .prompt_ids
        let traceURL = folder.appendingPathComponent("trace.json")
        let trace = FileManager.default.fileExists(atPath: traceURL.path)
            ? try JSONDecoder().decode(Trace.self, from: Data(contentsOf: traceURL)) : nil
        let n = ids.count
        if modes.contains("llamaCPU") || modes.contains("margins") {
            try #require(n < 32, "\(n) tokens take ggml's BLAS path, which .llamaCPU does not model")
        }
        if modes.contains("llamaExperts") {
            try #require(n >= 32, "\(n) tokens: below 32 llama.cpp rounds as .llamaCPU, not .llamaExperts")
        }
        // The reference's trace keeps every visit's running sum as Swift
        // arrays ([48][n][2304] floats, 0.44 MB per token): bounded here.
        try #require(n <= 512, "\(n) tokens: the reference's trace would hold \(n * 48 * 2304 * 4 >> 20) MiB")

        let device = try #require(MTLCreateSystemDefaultDevice())
        // `.pread`: the Model never opens an expert file; the reference maps them.
        // The CLI and server default: the receipt's size checks, not a SHA-256
        // pass over 15.8 GB of layer files through the file cache.
        let model = try Model.load(directoryURL: installURL, device: device, expecting: .inkling2Mini,
                                   integrityPolicy: .trustedReceiptWhenValid)
        let c = model.config
        let d = c.hiddenSize, vocab = c.unpaddedVocabSize
        let softcap = Double(c.finalLogitSoftcap)
        if let trace { try #require(trace.hidden_shape == [48, n, d] && trace.visits.count == 48) }
        let schedule = Inkling2ReferenceConfig.mini.schedule
        let weights = try Inkling2ReferenceWeights(model: model)
        let traceSets = trace?.visits.map { $0.experts.map(Set.init) }

        for mode in modes {
            let reference = try Inkling2ReferenceModel(config: .mini, weights: weights)
            reference.captureTrace = true
            let started = Date()
            if mode == "margins" {
                let golden = try #require(trace, "margins need the golden's trace"), sets = try #require(traceSets)
                let hidden = try Data(contentsOf: folder.appendingPathComponent("hidden_after_visit.f32"))
                let records = hidden.withUnsafeBytes { raw in
                    (0..<48).map { v in (0..<n).map { t in
                        (0..<d).map { raw.load(fromByteOffset: ((v * n + t) * d + $0) * 4, as: Float.self) }
                    } }
                }
                reference.precision = .llamaCPU
                reference.forcedRoutes = golden.visits.map(\.experts)
                reference.forcedVisitOutputs = records
                _ = reference.forward(tokens: ids, logits: .none)
                let own = try #require(reference.lastTrace)
                let differs = (0..<48).map { v in (0..<n).map { t in Set(own.ownRoutes[v][t].experts) != sets[v][t] } }
                try FileManager.default.createDirectory(at: outURL.appendingPathComponent("margins"),
                                                        withIntermediateDirectories: true)
                try write(Margins(trace: traceName, precision: "llamaCPU", margins: own.routeMargins,
                                  shortfalls: own.routeShortfalls, ownDiffers: differs),
                          to: outURL.appendingPathComponent("margins").appendingPathComponent("\(id).json"))
                let all = own.routeMargins.flatMap { $0 }.sorted()
                print("inkling2 precision \(id) margins (teacher-forced, .llamaCPU, "
                      + "\(String(format: "%.0f", Date().timeIntervalSince(started))) s): "
                      + "smallest \(all.prefix(5).map { String(format: "%.2e", $0) }), median \(all[all.count / 2]); "
                      + "own top-8 differs on \(differs.flatMap { $0 }.filter { $0 }.count) of \(48 * n)")
                continue
            }

            if mode == "fp32" {
                reference.precision = .fp32
            } else if mode == "llamaCPU" {
                reference.precision = .llamaCPU
            } else if mode == "llamaExperts" {
                reference.precision = .llamaExperts
            } else if mode == "metalRunner" {
                reference.precision = .metalRunner
            } else if mode.hasPrefix("metalRunner-"), let site = Self.sites[String(mode.dropFirst("metalRunner-".count))] {
                reference.precision = .metalRunner
                reference.metalRunnerSites = Inkling2ReferenceModel.MetalRunnerSites.all.subtracting(site)
            } else {
                Issue.record("unknown mode \(mode)")
                continue
            }
            _ = reference.forward(tokens: ids, logits: .none)
            let own = try #require(reference.lastTrace)
            let seconds = Date().timeIntervalSince(started)

            // Log-probs at every position: the head as the reference's
            // `logits(hidden:)` computes it (Q8_0-rounded input under
            // `.llamaCPU`), through the vectorized product, then the cap.
            let tfDir = outURL.appendingPathComponent(mode).appendingPathComponent("tf").appendingPathComponent(id)
            try FileManager.default.createDirectory(at: tfDir, withIntermediateDirectories: true)
            var records: [Inkling2G6cTests.LogProbRecord] = []
            var argmax = -1
            for t in 0..<n {
                let h = own.finalHidden[t]
                let input = reference.precision == .llamaCPU ? Inkling2ReferenceOps.q8_0ActivationRound(h) : h
                let raw = weights.output.apply(input)
                let row = Inkling2LogProbRow(floats: raw, count: vocab, softcap: softcap)
                try #require(row.rawFinite && row.logSumExp.isFinite, "\(mode): non-finite logits at \(t)")
                if t == n - 1 { argmax = row.top[0] }
                let next = t < n - 1 ? Int(ids[t + 1]) : row.top[0]
                records.append(.init(pos: t, next: next, next_logprob: row.logprob(next),
                                     top: row.top.map { Inkling2G6cTests.Pair(id: $0, logprob: row.logprob($0)) }))
            }
            try write(records, to: tfDir.appendingPathComponent("logprobs.json"))
            try write(Inkling2G6cTests.GoldenRecord(id: id, kind: "ids", prompt_ids: ids, generated_ids: [argmax],
                                                    logprobs_file: "logprobs.json", logprobs_first_pos: 0,
                                                    prefill_s: seconds, runner: "Inkling2ReferenceModel \(mode)"),
                      to: tfDir.appendingPathComponent("golden.json"))

            guard let traceSets else {
                print("inkling2 precision \(id) \(mode) (\(String(format: "%.0f", seconds)) s): "
                      + "\(n) positions written, next \(argmax)")
                continue
            }
            // The running sums and expert sets, as the runner's traces.
            let traceDir = outURL.appendingPathComponent(mode).appendingPathComponent("traces")
                .appendingPathComponent(id)
            try FileManager.default.createDirectory(at: traceDir, withIntermediateDirectories: true)
            var hidden = Data(capacity: 48 * n * d * 4)
            for v in 0..<48 {
                for t in 0..<n { own.visitOutputs[v][t].withUnsafeBytes { hidden.append(contentsOf: $0) } }
            }
            try hidden.write(to: traceDir.appendingPathComponent("hidden_after_visit.f32"))
            let visits = (0..<48).map { v in
                Inkling2G6cTests.TraceVisit(visit: v, slot: schedule[v].cacheSlot, layer: schedule[v].layer,
                                            experts: (0..<n).map { own.routes[v][$0].experts })
            }
            try write(Inkling2G6cTests.TraceRecord(hidden_file: "hidden_after_visit.f32", hidden_shape: [48, n, d],
                                                   hidden_tensor: "the running block sum after each visit (l_out)",
                                                   visits: visits),
                      to: traceDir.appendingPathComponent("trace.json"))
            try write(Inkling2G6cTests.GoldenRecord(id: id, kind: "ids", prompt_ids: ids, generated_ids: [],
                                                    logprobs_file: nil, logprobs_first_pos: nil, prefill_s: seconds,
                                                    runner: "Inkling2ReferenceModel \(mode)"),
                      to: traceDir.appendingPathComponent("golden.json"))

            let flips = (0..<48).reduce(0) { sum, v in
                sum + (0..<n).filter { Set(own.routes[v][$0].experts) != traceSets[v][$0] }.count
            }
            print("inkling2 precision \(id) \(mode) (\(String(format: "%.0f", seconds)) s): expert set differs "
                  + "from the CPU trace on \(flips) of \(48 * n) token-visits, next \(argmax)")
        }
    }
}
