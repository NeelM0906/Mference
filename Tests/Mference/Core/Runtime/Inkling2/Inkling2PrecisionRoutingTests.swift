import CryptoKit
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
/// sums, expert sets and the router's own selection margins), so the same
/// tools compare it with llama.cpp's goldens and with the runner, and prints
/// how many token-visits chose another expert set than the CPU trace.
///
/// Modes (`MFERENCE_INKLING2_PRECISION_MODES`, comma separated): `fp32`,
/// `llamaCPU`, `llamaExperts`, `metalRunner`, `metalRunner-<site>` (every FP16 site of
/// `.metalRunner` but that one: projectionInputs, bf16Inputs, queries,
/// cache, attentionOutput, expertInput, expertActivation), and `margins`:
/// the reference at `.llamaCPU` teacher-forced on the trace (its records and
/// routes, as G6a runs), writing `<out>/margins/<id>.json` with its own
/// router's gap between the 8th and 9th selection score at every
/// token-visit of the trace's state. `forced` is a measurement, not a
/// gate: the reference at `.fp32` teacher-forced on the runner's own trace
/// of the prompt (`MFERENCE_INKLING2_PRECISION_FORCED_FROM/<id>/`, as
/// `Inkling2G6cTests` writes it: its visit outputs and routes, as G6a and
/// G6b force them), writing `<out>/forced/<id>.json`
/// (`inkling2-forced-routing-v1`): at every token-visit of the runner's
/// state, `.fp32`'s own top-8, its selection margin, the runner's set's
/// shortfall below `.fp32`'s 8th score, and each visit's increment error;
/// and every disagreement, near-tie or not (shortfall under rule (a)'s
/// 0.0258). It asserts nothing about how many there are.
/// `MFERENCE_INKLING2_PRECISION_FORCED_PRECISION` (`fp32` by default,
/// `llamaExperts` from 32 tokens, `llamaCPU` below) sets the reference's
/// precision, and `FORCED_FROM` may also be one of K0's llama.cpp CPU trace
/// folders (`<goldens>/generate3/cpu`): its `expert_weights` and
/// `result_norm` then add G6a's route-weight and final-norm errors, so the
/// reference is checked against llama.cpp per visit at any depth up to 512.
///
/// Loads the install, so it runs only with `MFERENCE_INKLING2_GTURBO`,
/// `MFERENCE_INKLING2_GOLDENS_DIR` and `MFERENCE_INKLING2_PRECISION_OUT` set,
/// and then only inside `model-hold.sh`. `MFERENCE_INKLING2_PRECISION_TRACE`
/// (relative to the goldens folder) picks the prompt, `generate2/cpu/trace02`
/// by default: any golden folder; without a `trace.json` (a teacher-forced
/// prompt, `generate-tf/metal/<id>`) the log-probs and the reference's own
/// trace are written, nothing is compared with a CPU trace, and `margins` is
/// refused. `llamaCPU` and `margins` need fewer than 32 tokens,
/// `llamaExperts` 32 or more; at most 512 tokens (the trace's arrays).
@Suite("Inkling2 precision and routing on the real install", .serialized)
struct Inkling2PrecisionRoutingTests {
    static let environment = ProcessInfo.processInfo.environment
    static var install: URL? { environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) } }
    static var goldens: URL? { environment["MFERENCE_INKLING2_GOLDENS_DIR"].map { URL(fileURLWithPath: $0) } }
    static var output: URL? { environment["MFERENCE_INKLING2_PRECISION_OUT"].map { URL(fileURLWithPath: $0) } }
    /// The runner's traces for `forced`: `<this>/<id>/{golden.json,trace.json,hidden_after_visit.f32}`.
    static var forcedFrom: URL? {
        environment["MFERENCE_INKLING2_PRECISION_FORCED_FROM"].map { URL(fileURLWithPath: $0) }
    }
    /// Rule (a)'s bound on an expert-set flip's selection margin (spec v1.12).
    /// `forced` calls a disagreement a near-tie when the runner's shortfall
    /// is under it: the cost of the runner's actual choice, never below
    /// `.fp32`'s margin there, so the stricter reading (agreed with K0).
    /// Each disagreement also carries `.fp32`'s margin for rule (a)'s own.
    static let nearTieBound: Float = 0.0258

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
    /// The runner's trace as `Inkling2G6cTests` writes it.
    struct RunnerTrace: Decodable {
        struct Visit: Decodable {
            let slot: Int
            let layer: Int
            let experts: [[Int]]
            let margins: [Float]?
            /// llama.cpp's traces only.
            let expert_weights: [[Float]]?
        }
        let hidden_file: String
        let hidden_shape: [Int]
        let visits: [Visit]
        /// llama.cpp's traces only: the final norm's last rows.
        let result_norm_file: String?
        let result_norm_shape: [Int]?
    }
    /// An optional written as `null` when absent: the synthesized
    /// `Encodable` would leave the key out, and the reader expects it.
    struct Nullable<Value: Encodable>: Encodable {
        let value: Value?
        init(_ value: Value?) { self.value = value }
        func encode(to encoder: any Encoder) throws {
            var container = encoder.singleValueContainer()
            if let value { try container.encode(value) } else { try container.encodeNil() }
        }
    }
    /// `forced`'s record, `inkling2-forced-routing-v1`.
    struct ForcedRouting: Encodable {
        struct Visit: Encodable {
            let visit: Int
            let layer: Int
            let slot: Int
            /// Per token: the reference's own top-8 at the trace's state,
            /// ascending. The `fp32_*` names are the schema's; at another
            /// `precision` they hold that precision's values.
            let own_experts: [[Int]]
            /// Per token: the reference's 8th minus 9th selection score there.
            let fp32_margins: [Float]
            /// Per token: how far the trace's lowest expert lies below the
            /// reference's 8th selection score (0 when the sets agree; never
            /// below the margin when they differ).
            let shortfalls: [Float]
            let differs: [Bool]
            /// Per token: the relative L2 error of the reference's increment
            /// over this visit (output minus input, the trace's input)
            /// against the trace's, the reference using the trace's experts.
            let increment_rel_l2: [Float]
            /// Per token: G6a's element error, the output's largest element
            /// difference relative to the record's largest magnitude, or to
            /// the increment's, whichever is larger.
            let element_rel_max: [Float]
            /// Per token: the largest difference between the reference's
            /// weights for the forced experts and the trace's
            /// `expert_weights` (G6a's route weights); null when the trace
            /// has none, as the runner's.
            let route_weight_err: Nullable<[Float]>
        }
        struct Disagreement: Encodable {
            let visit: Int
            let layer: Int
            let pos: Int
            let shortfall: Float
            let fp32_margin: Float
            /// The runner's own margin at that token-visit (its trace's), or null.
            let runner_margin: Nullable<Float>
            let runner_only: [Int]
            let fp32_only: [Int]
            /// `shortfall` under `near_tie_bound`.
            let near_tie: Bool
        }
        let schema: String
        let id: String
        /// Provenance, so a reader can refuse a stale file: the candidate
        /// folder's name, the runner's commit and the reference's
        /// (`MFERENCE_INKLING2_PRECISION_RUNNER_TIP` and `_REFERENCE_COMMIT`,
        /// null when unset), and the SHA-256 of the `trace.json` forced from.
        let candidate: String
        let runner_tip: Nullable<String>
        let runner_trace_sha256: String
        let reference_commit: Nullable<String>
        let reference: String
        /// The reference's precision: fp32, llamaExperts or llamaCPU.
        let precision: String
        /// Per row of the trace's `result_norm` (its last rows), G6a's
        /// final-norm error; null when the trace has none, as the runner's.
        let final_norm_rel: Nullable<[Float]>
        /// The runner trace's folder name.
        let forced_from: String
        /// Where `runner_margin` comes from: "trace.json", or null when the
        /// runner's trace has no margins.
        let runner_margins: Nullable<String>
        let n_tokens: Int
        let top_k: Int
        let near_tie_bound: Float
        let token_visits: Int
        let disagreement_count: Int
        let disagreement_rate: Double
        let not_near_tie_count: Int
        let seconds: Double
        let visits: [Visit]
        let disagreements: [Disagreement]
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
            if mode == "forced" {
                try forcedRouting(reference: reference, id: id, ids: ids, d: d, topK: c.topKExperts,
                                  experts: c.numExperts, schedule: schedule, outURL: outURL)
                continue
            }
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

            // The running sums, expert sets and the router's own selection
            // margins, as the runner's traces (rule (a) compares the runner's
            // sets with these and needs each flip's margin).
            let traceDir = outURL.appendingPathComponent(mode).appendingPathComponent("traces")
                .appendingPathComponent(id)
            try FileManager.default.createDirectory(at: traceDir, withIntermediateDirectories: true)
            // Row by row, so no second copy of [48][n][d] is held.
            let hiddenURL = traceDir.appendingPathComponent("hidden_after_visit.f32")
            FileManager.default.createFile(atPath: hiddenURL.path, contents: nil)
            let hidden = try FileHandle(forWritingTo: hiddenURL)
            for v in 0..<48 {
                for t in 0..<n { try own.visitOutputs[v][t].withUnsafeBytes { try hidden.write(contentsOf: $0) } }
            }
            try hidden.close()
            let visits = (0..<48).map { v in
                Inkling2G6cTests.TraceVisit(visit: v, slot: schedule[v].cacheSlot, layer: schedule[v].layer,
                                            experts: (0..<n).map { own.routes[v][$0].experts },
                                            margins: own.routeMargins[v])
            }
            try write(Inkling2G6cTests.TraceRecord(hidden_file: "hidden_after_visit.f32", hidden_shape: [48, n, d],
                                                   hidden_tensor: "the running block sum after each visit (l_out)",
                                                   visits: visits),
                      to: traceDir.appendingPathComponent("trace.json"))
            try write(Inkling2G6cTests.GoldenRecord(id: id, kind: "ids", prompt_ids: ids, generated_ids: [],
                                                    logprobs_file: nil, logprobs_first_pos: nil, prefill_s: seconds,
                                                    runner: "Inkling2ReferenceModel \(mode)"),
                      to: traceDir.appendingPathComponent("golden.json"))

            let peak = Inkling2G6cTests.footprint().peak >> 20
            guard let traceSets else {
                print("inkling2 precision \(id) \(mode) (\(String(format: "%.0f", seconds)) s): "
                      + "\(n) positions and their trace written, next \(argmax); peak footprint \(peak) MiB")
                continue
            }
            let flips = (0..<48).reduce(0) { sum, v in
                sum + (0..<n).filter { Set(own.routes[v][$0].experts) != traceSets[v][$0] }.count
            }
            print("inkling2 precision \(id) \(mode) (\(String(format: "%.0f", seconds)) s): expert set differs "
                  + "from the CPU trace on \(flips) of \(48 * n) token-visits, next \(argmax); peak footprint \(peak) MiB")
        }
    }

    struct ForcedPrecisionError: Error, CustomStringConvertible {
        let name: String
        var description: String { "unknown MFERENCE_INKLING2_PRECISION_FORCED_PRECISION \(name)" }
    }

    /// The `forced` mode: the reference's own router at the trace's state.
    private func forcedRouting(reference: Inkling2ReferenceModel, id: String, ids: [Int32], d: Int, topK: Int,
                               experts: Int, schedule: [Inkling2ReferenceVisit], outURL: URL) throws {
        let root = try #require(Self.forcedFrom, "forced needs MFERENCE_INKLING2_PRECISION_FORCED_FROM")
        let folder = root.appendingPathComponent(id)
        let n = ids.count
        let runnerIDs = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: folder
            .appendingPathComponent("golden.json"))).prompt_ids
        try #require(runnerIDs == ids, "\(id): the runner's trace is of another prompt")
        let traceJSON = try Data(contentsOf: folder.appendingPathComponent("trace.json"))
        let traceSHA = SHA256.hash(data: traceJSON).map { String(format: "%02x", $0) }.joined()
        let runner = try JSONDecoder().decode(RunnerTrace.self, from: traceJSON)
        try #require(runner.hidden_shape == [48, n, d] && runner.visits.count == 48
                     && runner.visits.allSatisfy { $0.experts.count == n && $0.experts.allSatisfy { $0.count == topK } },
                     "\(id): the runner's trace shape")
        // Distinct, in range: the reference's precondition would end the process instead.
        try #require(runner.visits.allSatisfy { $0.experts.allSatisfy { e in
            Set(e).count == topK && e.allSatisfy { (0..<experts).contains($0) }
        } }, "\(id): the runner's expert ids")
        let hasMargins = runner.visits.allSatisfy { $0.margins?.count == n }
        try #require(zip(runner.visits, schedule).allSatisfy { $0.layer == $1.layer && $0.slot == $1.cacheSlot },
                     "\(id): the runner's visit order is the reference's schedule")
        let records: [[[Float]]] = try {
            let hidden = try Data(contentsOf: folder.appendingPathComponent(runner.hidden_file))
            try #require(hidden.count == 48 * n * d * 4, "\(id): the runner's hidden states")
            return hidden.withUnsafeBytes { raw in
                (0..<48).map { v in (0..<n).map { t in
                    (0..<d).map { raw.load(fromByteOffset: ((v * n + t) * d + $0) * 4, as: Float.self) }
                } }
            }
        }()

        let precisionName = Self.environment["MFERENCE_INKLING2_PRECISION_FORCED_PRECISION"] ?? "fp32"
        switch precisionName {
        case "fp32":
            reference.precision = .fp32
        case "llamaExperts":
            try #require(n >= 32, "\(n) tokens: below 32 llama.cpp rounds as .llamaCPU, not .llamaExperts")
            reference.precision = .llamaExperts
        case "llamaCPU":
            try #require(n < 32, "\(n) tokens take ggml's BLAS path, which .llamaCPU does not model")
            reference.precision = .llamaCPU
        default:
            throw ForcedPrecisionError(name: precisionName)
        }
        try #require(runner.visits.allSatisfy { w in
            w.expert_weights.map { $0.count == n && $0.allSatisfy { $0.count == topK } } ?? true
        }, "\(id): the trace's expert weights")
        let finalNorm: [[Float]]? = try runner.result_norm_file.map { file in
            let shape = try #require(runner.result_norm_shape)
            let rows = shape.first ?? 0
            try #require(shape == [rows, d] && (1...n).contains(rows), "\(id): result_norm shape")
            let data = try Data(contentsOf: folder.appendingPathComponent(file))
            try #require(data.count == rows * d * 4, "\(id): result_norm size")
            return data.withUnsafeBytes { raw in
                (0..<rows).map { r in (0..<d).map { raw.load(fromByteOffset: (r * d + $0) * 4, as: Float.self) } }
            }
        }

        let started = Date()
        reference.forcedRoutes = runner.visits.map(\.experts)
        reference.forcedVisitOutputs = records
        _ = reference.forward(tokens: ids, logits: .none)
        let seconds = Date().timeIntervalSince(started)
        let own = try #require(reference.lastTrace)

        var visits: [ForcedRouting.Visit] = []
        var disagreements: [ForcedRouting.Disagreement] = []
        for v in 0..<48 {
            // A block's first visit starts its running sum from zero (as G6a).
            let blockStart = schedule[v].layer % 6 == 0
            var increments: [Float] = [], elements: [Float] = [], weightErrors: [Float] = []
            var differs: [Bool] = []
            for t in 0..<n {
                let got = own.visitOutputs[v][t], want = records[v][t]
                let input: [Float]? = blockStart ? nil : records[v - 1][t]
                var error = 0.0, norm = 0.0, largestDiff = 0.0, largestWant = 0.0, largestIncrement = 0.0
                for i in 0..<d {
                    let base = Double(input?[i] ?? 0)
                    let g = Double(got[i]) - base, w = Double(want[i]) - base
                    error += (g - w) * (g - w)
                    norm += w * w
                    largestDiff = max(largestDiff, abs(g - w))
                    largestWant = max(largestWant, abs(Double(want[i])))
                    largestIncrement = max(largestIncrement, abs(w))
                }
                increments.append(Float((error / max(norm, 1e-30)).squareRoot()))
                elements.append(Float(largestDiff / max(min(largestWant, largestIncrement), 1e-6)))
                if let traced = runner.visits[v].expert_weights?[t] {
                    let mineW = own.routes[v][t].weights
                    weightErrors.append(mineW.count == traced.count
                                        ? zip(mineW, traced).map { abs($0 - $1) }.max()! : .infinity)
                }
                let mine = Set(own.ownRoutes[v][t].experts), theirs = Set(runner.visits[v].experts[t])
                differs.append(mine != theirs)
                if mine != theirs {
                    let shortfall = own.routeShortfalls[v][t]
                    disagreements.append(.init(visit: v, layer: schedule[v].layer, pos: t, shortfall: shortfall,
                                               fp32_margin: own.routeMargins[v][t],
                                               runner_margin: Nullable(hasMargins ? runner.visits[v].margins?[t] : nil),
                                               runner_only: theirs.subtracting(mine).sorted(),
                                               fp32_only: mine.subtracting(theirs).sorted(),
                                               near_tie: shortfall < Self.nearTieBound))
                }
            }
            visits.append(.init(visit: v, layer: schedule[v].layer, slot: schedule[v].cacheSlot,
                                own_experts: (0..<n).map { own.ownRoutes[v][$0].experts.sorted() },
                                fp32_margins: own.routeMargins[v], shortfalls: own.routeShortfalls[v],
                                differs: differs, increment_rel_l2: increments, element_rel_max: elements,
                                route_weight_err: Nullable(runner.visits[v].expert_weights == nil ? nil : weightErrors)))
        }
        let notNearTie = disagreements.filter { !$0.near_tie }
        let finalNormErrors: [Float]? = finalNorm.map { rows in
            rows.indices.map { r in
                let got = own.finalHidden[n - rows.count + r], want = rows[r]
                // As G6a's `relative`: infinite on any non-finite value
                // (`max()` would pass over a NaN), which the check below refuses.
                guard got.allSatisfy(\.isFinite), want.allSatisfy(\.isFinite) else { return Float.infinity }
                let diff = zip(got, want).map { abs($0 - $1) }.max()!
                return diff / max(want.map(abs).max()!, 1e-6)
            }
        }
        // The reader refuses non-finite values; so does JSONEncoder, less clearly.
        try #require(visits.allSatisfy { v in
            (v.fp32_margins + v.shortfalls + v.increment_rel_l2 + v.element_rel_max
             + (v.route_weight_err.value ?? [])).allSatisfy(\.isFinite)
        } && (finalNormErrors ?? []).allSatisfy(\.isFinite),
        "\(id): a non-finite margin, shortfall, increment, element, route-weight or final-norm error")
        try FileManager.default.createDirectory(at: outURL.appendingPathComponent("forced"),
                                                withIntermediateDirectories: true)
        try write(ForcedRouting(schema: "inkling2-forced-routing-v1", id: id,
                                candidate: root.lastPathComponent == "traces"
                                    ? root.deletingLastPathComponent().lastPathComponent
                                    : root.deletingLastPathComponent().lastPathComponent + "/" + root.lastPathComponent,
                                runner_tip: Nullable(Self.environment["MFERENCE_INKLING2_PRECISION_RUNNER_TIP"]),
                                runner_trace_sha256: traceSHA,
                                reference_commit: Nullable(Self.environment["MFERENCE_INKLING2_PRECISION_REFERENCE_COMMIT"]),
                                reference: "Inkling2ReferenceModel \(precisionName)", precision: precisionName,
                                final_norm_rel: Nullable(finalNormErrors), forced_from: folder.lastPathComponent,
                                runner_margins: Nullable(hasMargins ? "trace.json" : nil),
                                n_tokens: n, top_k: topK, near_tie_bound: Self.nearTieBound,
                                token_visits: 48 * n, disagreement_count: disagreements.count,
                                disagreement_rate: Double(disagreements.count) / Double(48 * n),
                                not_near_tie_count: notNearTie.count, seconds: seconds,
                                visits: visits, disagreements: disagreements),
                  to: outURL.appendingPathComponent("forced").appendingPathComponent("\(id).json"))
        let increments = visits.flatMap(\.increment_rel_l2).sorted()
        let weights = visits.flatMap { $0.route_weight_err.value ?? [] }
        print("inkling2 precision \(id) forced (.\(precisionName) at the trace's state, \(String(format: "%.0f", seconds)) s): "
              + "own top-8 differs from the runner's on \(disagreements.count) of \(48 * n) token-visits, "
              + "largest shortfall \(disagreements.map(\.shortfall).max() ?? 0), "
              + "not near-tie (>= \(Self.nearTieBound)) \(notNearTie.count); increment rel L2 median "
              + "\(increments[increments.count / 2]), max \(increments.last!); element max "
              + "\(visits.flatMap(\.element_rel_max).max()!), route weights max \(weights.max().map { "\($0)" } ?? "n/a"), "
              + "final norm max \(finalNormErrors?.max().map { "\($0)" } ?? "n/a"); "
              + "peak footprint \(Inkling2G6cTests.footprint().peak >> 20) MiB")
        for v in visits {
            let worstWeight = v.route_weight_err.value?.max().map { String(format: " w %.2e", $0) } ?? ""
            print(String(format: "inkling2 precision \(id) forced %2d L%02d s%02d incL2 %.2e max %.2e", v.visit, v.layer,
                         v.slot, v.increment_rel_l2.max()!, v.element_rel_max.max()!)
                  + worstWeight + " own-top8 differs \(v.differs.filter { $0 }.count)/\(n) short "
                  + String(format: "%.2e", v.shortfalls.max()!))
        }
        for x in notNearTie {
            print("inkling2 precision \(id) forced NOT A NEAR-TIE: visit \(x.visit) layer \(x.layer) pos \(x.pos) "
                  + "shortfall \(x.shortfall) fp32 margin \(x.fp32_margin) runner margin "
                  + "\(x.runner_margin.value.map { "\($0)" } ?? "none") runner-only \(x.runner_only) fp32-only \(x.fp32_only)")
        }
    }
}
