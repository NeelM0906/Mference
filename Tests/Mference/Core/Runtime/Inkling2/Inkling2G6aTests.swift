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
@Suite("Inkling2 G6a: the reference against the CPU-only traces", .serialized)
struct Inkling2G6aTests {
    static let environment = ProcessInfo.processInfo.environment
    static var install: URL? { environment["MFERENCE_INKLING2_GTURBO"].map { URL(fileURLWithPath: $0) } }
    static var goldens: URL? { environment["MFERENCE_INKLING2_GOLDENS_DIR"].map { URL(fileURLWithPath: $0) } }
    static var traces: [String] {
        environment["MFERENCE_INKLING2_G6A_TRACES"]?.split(separator: ",").map(String.init)
            ?? ["generate/cpu/trace01", "generate2/cpu/trace02"]
    }
    /// Provisional, until the first measurement sets them (spec v1.5, G6a).
    static let visitTolerance: Float = 0.05
    static let tieMargin: Float = 0.01

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
        let visits: [Visit]
    }

    /// Infinite when either side holds a non-finite value.
    private func relative(_ got: [Float], _ want: [Float]) -> Float {
        guard got.allSatisfy(\.isFinite), want.allSatisfy(\.isFinite) else { return .infinity }
        return zip(got, want).map { abs($0 - $1) }.max()! / max(want.map(abs).max()!, 1e-6)
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
        let device = try #require(MTLCreateSystemDefaultDevice())
        // `.pread`: the Model never opens an expert file; the reference maps them.
        let model = try Model.load(directoryURL: installURL, device: device, expecting: .inkling2Mini)
        let weights = try Inkling2ReferenceWeights(model: model)
        let d = model.config.hiddenSize
        let schedule = Inkling2ReferenceConfig.mini.schedule

        for name in Self.traces {
            let folder = goldensURL.appendingPathComponent(name)
            let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: folder.appendingPathComponent("golden.json")))
            let trace = try JSONDecoder().decode(Trace.self, from: Data(contentsOf: folder.appendingPathComponent("trace.json")))
            let hidden = try Data(contentsOf: folder.appendingPathComponent("hidden_after_visit.f32"))
            let finalNorm = try Data(contentsOf: folder.appendingPathComponent("result_norm.f32"))
            let n = golden.prompt_ids.count
            #expect(trace.hidden_shape == [48, n, d] && hidden.count == 48 * n * d * 4, "\(name) shape")
            #expect(trace.visits.count == 48)
            #expect(zip(trace.visits, schedule).allSatisfy { $0.layer == $1.layer && $0.slot == $1.cacheSlot },
                    "\(name): the trace's visit order is the reference's schedule")
            let records = (0..<48).map { v in (0..<n).map { t in floats(hidden, from: (v * n + t) * d, count: d) } }

            let reference = try Inkling2ReferenceModel(config: .mini, weights: weights)
            reference.captureTrace = true
            reference.precision = .llamaCPU
            reference.forcedRoutes = trace.visits.map(\.experts)
            reference.forcedVisitOutputs = records
            let started = Date()
            _ = reference.forward(tokens: golden.prompt_ids, logits: .none)
            let seconds = Date().timeIntervalSince(started)
            let own = try #require(reference.lastTrace)

            var lines: [String] = []
            var worst: Float = 0, worstShortfall: Float = 0, worstWeight: Float = 0, flips = 0
            for v in 0..<48 {
                let blockStart = schedule[v].layer % 6 == 0
                var err: Float = 0, increment: Float = 0, weightErr: Float = 0, shortfall: Float = 0
                for t in 0..<n {
                    let got = own.visitOutputs[v][t], want = records[v][t]
                    err = max(err, relative(got, want))
                    let input = blockStart ? [Float](repeating: 0, count: d) : records[v - 1][t]
                    increment = max(increment, relative(zip(got, input).map { $0 - $1 },
                                                        zip(want, input).map { $0 - $1 }))
                    let s = own.routeShortfalls[v][t]
                    if s > 0 { flips += 1 }
                    shortfall = max(shortfall, s)
                    let w = own.routes[v][t].weights, k = trace.visits[v].expert_weights[t]
                    weightErr = max(weightErr, zip(w, k).map { abs($0 - $1) }.max() ?? .infinity)
                }
                worst = max(worst, err, increment)
                worstShortfall = max(worstShortfall, shortfall)
                worstWeight = max(worstWeight, weightErr)
                lines.append(String(format: "%2d L%02d s%02d out %.2e inc %.2e w %.2e short %.2e",
                                    v, schedule[v].layer, schedule[v].cacheSlot, err, increment, weightErr, shortfall))
            }
            let normErr = relative(try #require(own.finalHidden.last), floats(finalNorm, from: 0, count: d))
            print("inkling2 G6a \(name) (\(n) tokens, \(String(format: "%.0f", seconds)) s): worst visit \(worst), "
                  + "final norm \(normErr), route weights \(worstWeight), \(flips) near-tie flips, "
                  + "largest shortfall \(worstShortfall)")
            lines.forEach { print("inkling2 G6a \(name) " + $0) }
            #expect(worst < Self.visitTolerance, "\(name)")
            #expect(normErr < Self.visitTolerance, "\(name)")
            #expect(worstShortfall <= Self.tieMargin, "\(name)")
        }
    }
}
