import Foundation
import Testing
@testable import Mference

/// The parity metrics and harness, model-free.
@Suite("ParityHarnessTests")
struct ParityHarnessTests {
    @Test func logSoftmaxNormalizes() {
        let l = ParityMetrics.logSoftmax([1, 2, 3, 4])
        let total = l.map { exp($0) }.reduce(0, +)
        #expect(abs(total - 1) < 1e-12)
        #expect(abs(l[3] - l[2] - 1) < 1e-12)
        // large logits do not overflow
        let big = ParityMetrics.logSoftmax([1000, 1000])
        #expect(abs(big[0] - log(0.5)) < 1e-12)
    }

    @Test func klDivergenceMatchesTheClosedForm() {
        // P = softmax([0, 0]) = (0.5, 0.5); Q = softmax([log 3, 0]) = (0.75, 0.25)
        let kl = ParityMetrics.klDivergence(golden: [0, 0], candidate: [Float(log(3.0)), 0])
        let expected = 0.5 * log(0.5 / 0.75) + 0.5 * log(0.5 / 0.25)
        #expect(abs(kl - expected) < 1e-7)
        #expect(ParityMetrics.klDivergence(golden: [1, 2, 3], candidate: [1, 2, 3]) == 0)
        // a constant shift of the logits changes nothing
        #expect(ParityMetrics.klDivergence(golden: [1, 2, 3], candidate: [11, 12, 13]) < 1e-12)
        #expect(ParityMetrics.klDivergence(golden: [5, 0, 0], candidate: [0, 5, 0]) > 4)
    }

    @Test func argmaxTiesGoToTheLowestIndex() {
        #expect(ParityMetrics.argmax([1, 3, 3, 2]) == 1)
        #expect(ParityMetrics.argmax([-1]) == 0)
    }

    @Test func errorStats() {
        var s = ParityMetrics.ErrorStats()
        s.add(golden: [1, -2, 0], candidate: [1, -2.5, 0.5])
        #expect(s.count == 3)
        #expect(s.maxAbs == 0.5)
        #expect(abs(s.meanAbs - 1.0 / 3.0) < 1e-12)
        #expect(abs(s.relativeL2 - (0.5 / 5.0).squareRoot()) < 1e-12)
        var t = ParityMetrics.ErrorStats()
        t.add(golden: [3], candidate: [3])
        s.merge(t)
        #expect(s.count == 4 && s.maxGolden == 3)
    }

    @Test func distribution() {
        let d = ParityMetrics.Distribution((1...100).map(Double.init))
        #expect(d.count == 100 && d.max == 100 && d.median == 50 && d.p99 == 99)
        #expect(abs(d.mean - 50.5) < 1e-12)
        #expect(ParityMetrics.Distribution([]).max == 0)
    }

    @Test func toleranceClasses() {
        #expect(OpToleranceClass.of(base: "model.input_embed") == .exact)
        #expect(OpToleranceClass.of(base: "attn_norm") == .norm)
        #expect(OpToleranceClass.of(base: "attn_gated") == .mixer)
        #expect(OpToleranceClass.of(base: "l_out") == .residual)
        #expect(OpToleranceClass.of(base: "state_predelta") == .state)
        #expect(ParityTolerances.d1 == ParityTolerances(minimumAgreement: 0.99, maxTokenKLD: 0.01, meanKLD: 0.001))
        var s = ParityMetrics.ErrorStats()
        s.add(golden: [1, 0], candidate: [1.005, 0])
        #expect(ParityHarness.localVerdict(s, class: .gemm, name: "x") == nil)
        #expect(ParityHarness.localVerdict(s, class: .norm, name: "x") != nil)
        #expect(ParityHarness.localVerdict(ParityMetrics.ErrorStats(), class: .gemm, name: "x") != nil)
    }

    /// The synthetic golden of `GoldenStoreTests`: generated tokens [3, 0],
    /// logits rows [0,1,2,3] and [3,2,1,0].
    @Test func comparesLogitsAgainstAGolden() throws {
        let root = try GoldenStoreTests.syntheticRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let g = try GoldenStore.open("p", root: root)

        let same = try ParityHarness.compareLogits(golden: g) { try g.logits().row($0) }
        #expect(same.positions == 2 && same.agreements == 2 && same.kld == [0, 0])

        // row 1 flips its argmax to token 1 (golden margin 1.0)
        let flipped = try ParityHarness.compareLogits(golden: g) { r in r == 0 ? [0, 1, 2, 3] : [2, 4, 1, 0] }
        #expect(flipped.agreements == 1 && flipped.disagreements == [1])
        #expect(flipped.disagreementMargins == [1])
        #expect(flipped.kld[1] > 0)

        let report = ParityHarness.LogitReport(results: [same, flipped])
        #expect(report.positions == 4 && report.agreement == 0.75)
        let verdict = report.verdict()
        #expect(!verdict.passed)
        #expect(verdict.failures.contains { $0.hasPrefix("greedy agreement") })
        #expect(ParityHarness.LogitReport(results: [same]).verdict().passed)
        #expect(report.table(title: "run").contains("| **all** | 4 |"))

        // a wrong vocabulary size is an error, not a silent pass
        #expect(throws: GoldenError.self) { try ParityHarness.compareLogits(golden: g) { _ in [1, 2] } }
    }

    /// A candidate that skips rows, or is shorter than the golden, fails:
    /// missing positions never count as agreement.
    @Test func incompleteCoverageFails() throws {
        let root = try GoldenStoreTests.syntheticRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let g = try GoldenStore.open("p", root: root)
        let none = try ParityHarness.compareLogits(golden: g) { _ in nil }
        #expect(none.positions == 0 && none.missing == [0, 1] && none.expectedPositions == 2)
        let v = ParityHarness.LogitReport(results: [none]).verdict()
        #expect(!v.passed && v.failures.contains { $0.hasPrefix("incomplete coverage") })

        let prefix = try ParityHarness.compareLogits(golden: g, rows: 1) { try g.logits().row($0) }
        #expect(prefix.missing == [1])
        #expect(!ParityHarness.LogitReport(results: [prefix]).verdict().passed)
        #expect(!ParityHarness.LogitReport(results: []).verdict().passed)
    }

    /// A NaN or infinite candidate logit is never a perfect score, even when
    /// the argmax still agrees.
    @Test func nonFiniteLogitsFail() throws {
        #expect(ParityMetrics.klDivergence(golden: [0, 1, 2, 3], candidate: [0, .nan, 2, 3]) == .infinity)
        #expect(ParityMetrics.klDivergence(golden: [0, 1, 2, 3], candidate: [0, 1, .infinity, 3]) == .infinity)
        let root = try GoldenStoreTests.syntheticRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let g = try GoldenStore.open("p", root: root)
        let r = try ParityHarness.compareLogits(golden: g) { row in row == 0 ? [0, .nan, 2, 3] : [3, 2, 1, 0] }
        #expect(r.agreements == 2)
        let v = ParityHarness.LogitReport(results: [r]).verdict()
        #expect(!v.passed && v.failures.contains { $0.hasPrefix("non-finite KL") })
    }

    @Test func comparesTensors() throws {
        let root = try GoldenStoreTests.syntheticRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let g = try GoldenStore.open("p", root: root)
        let lout = try g.tensor(named: "l_out-3")
        // candidate has positions 0 and 3 only; position 3 is off by 1 in one element
        let stats = try ParityHarness.compareTensor(lout) { p in
            switch p {
            case 0: return [10, 11]
            case 3: return [40, 42]
            default: return nil
            }
        }
        #expect(stats.count == 4 && stats.maxAbs == 1)
        #expect(try ParityHarness.compareTensor(named: "l_out-3", golden: g, run: g).maxAbs == 0)
    }

    /// The committed slice's top-8 logits are real llama.cpp rows: the metrics
    /// agree with themselves on them (top-1 is the greedy token, KL of a row
    /// with itself is 0).
    @Test func sliceRowsAreConsistent() throws {
        let slice = try #require(try GoldenSlice.load(), "the committed slice is missing")
        for (id, p) in slice.prompts {
            for (g, top) in p.logitsTop.prefix(4).enumerated() {
                #expect(ParityMetrics.argmax(top.values) == 0, "\(id) row \(g)")
                #expect(ParityMetrics.klDivergence(golden: top.values, candidate: top.values) == 0)
                #expect(top.logSumExp >= Double(top.values[0]), "\(id) row \(g)")
            }
        }
    }
}

/// llama.cpp against itself: the noise floor that justifies ε and the per-op
/// tolerances. `MFERENCE_NATIVE_GOLDENS` is the golden run;
/// `MFERENCE_NATIVE_NOISE_RUNS` a colon-separated list of other run
/// directories captured with `mfe-golden --teacher <golden>`. Prints the
/// tables; with `MFERENCE_PARITY_REPORT=<file>` also writes them there.
@Suite("ParityNoiseFloorTests", .enabled(if: GoldenStore.root != nil
    && ProcessInfo.processInfo.environment["MFERENCE_NATIVE_NOISE_RUNS"] != nil))
struct ParityNoiseFloorTests {
    @Test func noiseFloor() throws {
        let env = ProcessInfo.processInfo.environment
        let runs = (env["MFERENCE_NATIVE_NOISE_RUNS"] ?? "").split(separator: ":").map(String.init)
        let set = try GoldenStore.promptSet()
        var out = "# llama.cpp noise floor against the golden\n\n"
        for run in runs {
            let runURL = URL(fileURLWithPath: run, isDirectory: true)
            var results: [ParityHarness.LogitResult] = []
            var perClass: [OpToleranceClass: ParityMetrics.ErrorStats] = [:]
            var perBase: [String: ParityMetrics.ErrorStats] = [:]
            for p in set.prompts {
                let golden = try GoldenStore.open(p.id)
                let other = try GoldenStore.open(p.id, root: runURL)
                results.append(try ParityHarness.compareLogits(golden: golden, run: other))
                guard other.manifest.mode == "golden", golden.manifest.mode == "golden" else { continue }
                for t in golden.tensors where other.entry(named: t.name, occurrence: t.occurrence) != nil {
                    let s = try ParityHarness.compareTensor(named: t.name, occurrence: t.occurrence, golden: golden, run: other)
                    perClass[OpToleranceClass.of(base: t.base), default: .init()].merge(s)
                    perBase["\(t.base)#\(t.occurrence)", default: .init()].merge(s)
                }
            }
            let report = ParityHarness.LogitReport(results: results)
            out += "## \(runURL.lastPathComponent)\n\n" + report.table(title: "prompt")
            // the justification of ε: llama.cpp's Metal variants pass the bar,
            // its CPU backend (Q8_K activations) does not
            let verdict = report.verdict()
            if runURL.lastPathComponent.hasPrefix("metal") {
                #expect(verdict.passed, "\(runURL.lastPathComponent): \(verdict.failures)")
            } else if runURL.lastPathComponent == "cpu" {
                #expect(!verdict.passed, "the CPU backend now passes D1; revisit ε")
            }
            out += "\nD1 verdict: \(verdict.passed ? "pass" : "fail: " + verdict.failures.joined(separator: "; "))\n"
            let flips = results.flatMap { r in zip(r.disagreements, r.disagreementMargins).map { "\(r.promptID)@\($0.0) margin \(String(format: "%.3f", $0.1))" } }
            out += "\nDisagreements: \(flips.isEmpty ? "none" : flips.joined(separator: ", "))\n\n"
            if !perClass.isEmpty {
                out += "| class | elements | max abs | mean abs | rel L2 |\n|---|---:|---:|---:|---:|\n"
                for c in OpToleranceClass.allCases {
                    guard let s = perClass[c] else { continue }
                    out += "| \(c.rawValue) | \(s.count) | " + String(format: "%.3g | %.3g | %.3g |\n", s.maxAbs, s.meanAbs, s.relativeL2)
                }
                out += "\n| tensor | max abs | mean abs | rel L2 |\n|---|---:|---:|---:|\n"
                for (k, s) in perBase.sorted(by: { $0.key < $1.key }) {
                    out += String(format: "| %@ | %.3g | %.3g | %.3g |\n", k, s.maxAbs, s.meanAbs, s.relativeL2)
                }
                out += "\n"
            }
        }
        print(out)
        if let path = env["MFERENCE_PARITY_REPORT"] {
            try out.write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
