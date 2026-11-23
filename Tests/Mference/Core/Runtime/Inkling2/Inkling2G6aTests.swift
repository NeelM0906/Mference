import Foundation
import Metal
import Testing
@testable import Mference
import MferenceValidationSupport

/// G6a: the whole-model CPU reference on the real install against K0's
/// CPU-only traces (llama.cpp b12641, `-ngl 0`), teacher-forced visit by
/// visit as K0's recipe describes (`$INK/goldens/README.md`, "Recipe:
/// teacher-forcing a runner from the CPU-only trace"). The reference rounds
/// as that backend does (`.llamaCPU`), follows the trace's routes, and after
/// every visit continues from the trace's record of it, so each visit's error
/// is its own; the final normed hidden state is compared with
/// `result_norm.f32`. The reference reads the experts from the layer files
/// mapped read-only, independently of the runner's store.
///
/// Loads the install, so it runs only with `MFERENCE_INKLING2_GTURBO` and
/// `MFERENCE_INKLING2_GOLDENS_DIR` (K0's `goldens/` folder) set, and then
/// only inside `model-hold.sh`. `MFERENCE_INKLING2_G6A_TRACES` (comma
/// separated, relative to the goldens folder) picks the traces.
///
/// Traces of 32 or more tokens are refused: there ggml's BLAS backend takes
/// the dense products with dequantized F32 weights and unrounded inputs,
/// which `.llamaCPU` does not model.
@Suite("Inkling2 G6a: the reference against the CPU-only traces", .serialized)
struct Inkling2G6aTests {
    static let environment = ProcessInfo.processInfo.environment
    static var install: URL? { environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) } }
    static var goldens: URL? { environment["MFERENCE_INKLING2_GOLDENS_DIR"].map { URL(fileURLWithPath: $0) } }
    static var traces: [String] {
        environment["MFERENCE_INKLING2_G6A_TRACES"]?.split(separator: ",").map(String.init)
            ?? ["generate/cpu/trace01", "generate2/cpu/trace02"]
    }
    /// Per visit: relative L2 of the visit's own increment (output minus
    /// the record it continued; the whole output at a block start), and the
    /// largest element difference relative to the record's largest
    /// magnitude. Provisional, until the first measurement sets them (spec
    /// section 8, G6a).
    static let incrementTolerance: Float = 0.02
    static let visitTolerance: Float = 0.05
    static let routeWeightTolerance: Float = 0.01
    /// Router: the reference's own top-8 may differ from the trace's only at
    /// near-ties, on at most 1% of token-visits.
    static let tieMargin: Float = 0.01
    static let flipFraction: Float = 0.01

    struct Golden: Decodable {
        let prompt_ids: [Int32]
    }
    struct Trace: Decodable {
        struct Visit: Decodable {
            let layer: Int
            let slot: Int
            let experts: [[Int]]
            let expert_weights: [[Float]]
        }
        let hidden_shape: [Int]
        let result_norm_shape: [Int]
        let visits: [Visit]
    }

    /// Infinite when either side holds a non-finite value.
    private func relative(_ got: [Float], _ want: [Float]) -> Float {
        guard got.allSatisfy(\.isFinite), want.allSatisfy(\.isFinite) else { return .infinity }
        return zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, 1e-6)
    }

    /// ||got - want|| / ||want||; infinite on a non-finite value.
    private func relativeL2(_ got: [Float], _ want: [Float]) -> Float {
        guard got.allSatisfy(\.isFinite), want.allSatisfy(\.isFinite) else { return .infinity }
        let num = zip(got, want).reduce(0.0) { $0 + Double($1.0 - $1.1) * Double($1.0 - $1.1) }
        let den = want.reduce(0.0) { $0 + Double($1) * Double($1) }
        return Float((num / max(den, 1e-30)).squareRoot())
    }

    private func floats(_ data: Data, from: Int, count: Int) -> [Float] {
        data.withUnsafeBytes { raw in
            (0..<count).map { raw.load(fromByteOffset: (from + $0) * 4, as: Float.self) }
        }
    }

    @Test(.enabled(if: install != nil && goldens != nil,
                   "needs MFERENCE_INKLING2_GTURBO and MFERENCE_INKLING2_GOLDENS_DIR"))
    func referenceMatchesTheCPUTracesVisitByVisit() throws {
        let installURL = try #require(Self.install), goldensURL = try #require(Self.goldens)
        let traces = Self.traces
        try #require(!traces.isEmpty, "no trace selected")
        let device = try #require(MTLCreateSystemDefaultDevice())
        // `.pread`: the Model never opens an expert file; the reference maps them.
        let model = try Model.load(directoryURL: installURL, device: device, expecting: .inkling2Mini)
        let d = model.config.hiddenSize
        let schedule = Inkling2ReferenceConfig.mini.schedule

        for name in traces {
            let folder = goldensURL.appendingPathComponent(name)
            let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: folder.appendingPathComponent("golden.json")))
            let trace = try JSONDecoder().decode(Trace.self, from: Data(contentsOf: folder.appendingPathComponent("trace.json")))
            let hidden = try Data(contentsOf: folder.appendingPathComponent("hidden_after_visit.f32"))
            let finalNorm = try Data(contentsOf: folder.appendingPathComponent("result_norm.f32"))
            let n = golden.prompt_ids.count
            try #require(n < 32, "\(name): \(n) tokens take ggml's BLAS path, which .llamaCPU does not model")
            try #require(trace.hidden_shape == [48, n, d] && hidden.count == 48 * n * d * 4, "\(name) shape")
            let normRows = trace.result_norm_shape.first ?? 0
            try #require(trace.result_norm_shape == [normRows, d] && (1...n).contains(normRows)
                         && finalNorm.count == normRows * d * 4, "\(name) result_norm shape")
            try #require(trace.visits.count == 48 && trace.visits.allSatisfy { $0.experts.count == n })
            #expect(zip(trace.visits, schedule).allSatisfy { $0.layer == $1.layer && $0.slot == $1.cacheSlot },
                    "\(name): the trace's visit order is the reference's schedule")
            let records = (0..<48).map { v in (0..<n).map { t in floats(hidden, from: (v * n + t) * d, count: d) } }

            // Weights per trace: the read-only mappings of the expert files
            // go when the trace is done.
            let reference = try Inkling2ReferenceModel(config: .mini, weights: Inkling2ReferenceWeights(model: model))
            reference.captureTrace = true
            reference.precision = .llamaCPU
            reference.forcedRoutes = trace.visits.map(\.experts)
            reference.forcedVisitOutputs = records
            let started = Date()
            _ = reference.forward(tokens: golden.prompt_ids, logits: .none)
            let seconds = Date().timeIntervalSince(started)
            let own = try #require(reference.lastTrace)

            var lines: [String] = []
            var worstL2: Float = 0, worstMax: Float = 0, worstShortfall: Float = 0, worstWeight: Float = 0
            var flips = 0
            for v in 0..<48 {
                let blockStart = schedule[v].layer % 6 == 0
                var l2: Float = 0, maxErr: Float = 0, weightErr: Float = 0, shortfall: Float = 0
                var differ = 0, symmetric = 0
                for t in 0..<n {
                    let got = own.visitOutputs[v][t], want = records[v][t]
                    let input = blockStart ? [Float](repeating: 0, count: d) : records[v - 1][t]
                    let gotInc = zip(got, input).map { $0 - $1 }, wantInc = zip(want, input).map { $0 - $1 }
                    l2 = max(l2, relativeL2(gotInc, wantInc))
                    maxErr = max(maxErr, relative(got, want), relative(gotInc, wantInc))
                    shortfall = max(shortfall, own.routeShortfalls[v][t])
                    let mine = Set(own.ownRoutes[v][t].experts), theirs = Set(trace.visits[v].experts[t])
                    if mine != theirs {
                        differ += 1
                        symmetric += mine.symmetricDifference(theirs).count
                    }
                    let w = own.routes[v][t].weights, k = trace.visits[v].expert_weights[t]
                    weightErr = max(weightErr, w.count == k.count ? zip(w, k).map { abs($0 - $1) }.max()! : .infinity)
                }
                flips += differ
                worstL2 = max(worstL2, l2)
                worstMax = max(worstMax, maxErr)
                worstShortfall = max(worstShortfall, shortfall)
                worstWeight = max(worstWeight, weightErr)
                lines.append(String(format: "%2d L%02d s%02d incL2 %.2e max %.2e w %.2e own-top8 differs %d/%d (sym %d) short %.2e",
                                    v, schedule[v].layer, schedule[v].cacheSlot, l2, maxErr, weightErr,
                                    differ, n, symmetric, shortfall))
            }
            var normErr: Float = 0
            for r in 0..<normRows {
                normErr = max(normErr, relative(own.finalHidden[n - normRows + r], floats(finalNorm, from: r * d, count: d)))
            }
            print("inkling2 G6a \(name) (\(n) tokens, \(String(format: "%.0f", seconds)) s): "
                  + "worst increment L2 \(worstL2), worst max \(worstMax), final norm (\(normRows) rows) \(normErr), "
                  + "route weights \(worstWeight), own top-8 differs on \(flips) of \(48 * n) token-visits, "
                  + "largest shortfall \(worstShortfall)")
            lines.forEach { print("inkling2 G6a \(name) " + $0) }
            #expect(worstL2 < Self.incrementTolerance, "\(name)")
            #expect(worstMax < Self.visitTolerance, "\(name)")
            #expect(normErr < Self.visitTolerance, "\(name)")
            #expect(worstWeight < Self.routeWeightTolerance, "\(name)")
            #expect(worstShortfall <= Self.tieMargin, "\(name)")
            #expect(Float(flips) <= Self.flipFraction * Float(48 * n), "\(name)")
        }
    }
}
