import Foundation
@testable import Mference

/// Compares a candidate (a backend, the all-reference path, a single
/// component) with the llama.cpp golden under the D1 bar
/// (`docs/NATIVE_PARITY.md`). The candidate supplies numbers; the harness
/// scores them. Model-free: tests feed it goldens, other golden runs, or
/// synthetic rows.
enum ParityHarness {

    // MARK: logits (D1: greedy agreement and per-token KL)

    /// One prompt's logit comparison, teacher-forced on the golden tokens:
    /// row `g` is the candidate's logits after the prompt and the first `g`
    /// golden generated tokens.
    struct LogitResult {
        var promptID: String
        /// Positions the golden has (one per generated token). A candidate
        /// must cover every one of them.
        var expectedPositions: Int = 0
        /// Rows (index into the generated tokens) the candidate did not
        /// supply. Any missing row fails the bar.
        var missing: [Int] = []
        /// Positions compared.
        var positions: Int
        /// Positions where the candidate's argmax equals the golden token.
        var agreements: Int
        /// Per-position KL(golden || candidate), in nats.
        var kld: [Double]
        /// Rows (index into the generated tokens) where the argmax differs.
        var disagreements: [Int]
        /// Golden top-1 minus top-2 logit at each disagreement (how close a
        /// call it was).
        var disagreementMargins: [Double]

        var agreement: Double { positions == 0 ? 1 : Double(agreements) / Double(positions) }
    }

    /// Compares `candidate(g)` (the candidate's F32 logits row `g`, or nil
    /// when it has none) with the golden's pass A logits.
    static func compareLogits(golden: GoldenPrompt, rows: Int? = nil,
                              candidate: (Int) throws -> [Float]?) throws -> LogitResult {
        let logits = try golden.logits()
        let tokens = golden.generatedTokens
        let n = min(rows ?? logits.rows, logits.rows)
        var result = LogitResult(promptID: golden.id, positions: 0, agreements: 0, kld: [],
                                 disagreements: [], disagreementMargins: [])
        result.expectedPositions = logits.rows
        // rows past `n` (a shorter candidate) are missing too
        result.missing = Array(n..<logits.rows)
        for g in 0..<n {
            guard let c = try candidate(g) else {
                result.missing.append(g)
                continue
            }
            try logits.withRow(g) { gold in
                guard c.count == gold.count else {
                    throw GoldenError.sizeMismatch("\(golden.id) row \(g): candidate \(c.count) vs golden \(gold.count)")
                }
                let kl = c.withUnsafeBufferPointer { ParityMetrics.klDivergence(golden: gold, candidate: $0) }
                result.kld.append(kl)
                result.positions += 1
                let choice = c.withUnsafeBufferPointer { ParityMetrics.argmax($0) }
                if Int32(choice) == tokens[g] {
                    result.agreements += 1
                } else {
                    result.disagreements.append(g)
                    result.disagreementMargins.append(Self.topMargin(gold))
                }
            }
        }
        return result
    }

    /// Another golden run (a noise-floor run, teacher-forced on the golden)
    /// as the candidate.
    static func compareLogits(golden: GoldenPrompt, run: GoldenPrompt) throws -> LogitResult {
        guard run.promptTokens == golden.promptTokens else {
            throw GoldenError.badFormat("\(golden.id): prompt tokens differ between runs")
        }
        let other = try run.logits()
        let n = min(other.rows, golden.generatedTokens.count)
        if n > 1 {
            // teacher forcing: the runs must have fed the same tokens
            guard Array(run.generatedTokens.prefix(n - 1)) == Array(golden.generatedTokens.prefix(n - 1)) else {
                throw GoldenError.badFormat("\(golden.id): run \(run.directory.path) was not teacher-forced on the golden")
            }
        }
        return try compareLogits(golden: golden, rows: n) { g in other.row(g) }
    }

    static func topMargin(_ x: UnsafeBufferPointer<Float>) -> Double {
        var a = -Float.infinity, b = -Float.infinity
        for v in x {
            if v > a { b = a; a = v } else if v > b { b = v }
        }
        return Double(a - b)
    }

    /// The prompt set's aggregate under the bar.
    struct LogitReport {
        var results: [LogitResult]
        var positions: Int { results.reduce(0) { $0 + $1.positions } }
        var expectedPositions: Int { results.reduce(0) { $0 + $1.expectedPositions } }
        var missing: Int { results.reduce(0) { $0 + $1.missing.count } }
        var agreements: Int { results.reduce(0) { $0 + $1.agreements } }
        var agreement: Double { positions == 0 ? 1 : Double(agreements) / Double(positions) }
        var kld: ParityMetrics.Distribution { ParityMetrics.Distribution(results.flatMap(\.kld)) }

        /// D1: agreement and per-token KL against `tolerances`.
        func verdict(_ tolerances: ParityTolerances = .d1) -> Verdict {
            var failures: [String] = []
            if missing > 0 || positions < expectedPositions || expectedPositions == 0 {
                failures.append("incomplete coverage: \(positions) of \(expectedPositions) positions compared, \(missing) missing")
            }
            if agreement < tolerances.minimumAgreement {
                failures.append(String(format: "greedy agreement %.4f < %.4f (%ld of %ld)", agreement,
                                       tolerances.minimumAgreement, agreements, positions))
            }
            let d = kld
            if results.contains(where: { $0.kld.contains { !$0.isFinite } }) {
                failures.append("non-finite KL (NaN or infinite candidate logits)")
            }
            if d.max > tolerances.maxTokenKLD {
                failures.append(String(format: "max per-token KL %.3g > ε %.3g", d.max, tolerances.maxTokenKLD))
            }
            if d.mean > tolerances.meanKLD {
                failures.append(String(format: "mean KL %.3g > %.3g", d.mean, tolerances.meanKLD))
            }
            return Verdict(failures: failures)
        }

        /// A Markdown table: one row per prompt plus the total.
        func table(title: String) -> String {
            var s = "| \(title) | positions | agreement | mean KL | p99 KL | max KL |\n|---|---:|---:|---:|---:|---:|\n"
            for r in results {
                let d = ParityMetrics.Distribution(r.kld)
                s += String(format: "| %@ | %ld | %.4f | %.3g | %.3g | %.3g |\n", r.promptID, r.positions, r.agreement, d.mean, d.p99, d.max)
            }
            let d = kld
            s += String(format: "| **all** | %ld | %.4f | %.3g | %.3g | %.3g |\n", positions, agreement, d.mean, d.p99, d.max)
            return s
        }
    }

    struct Verdict: Equatable {
        var failures: [String]
        var passed: Bool { failures.isEmpty }
    }

    // MARK: per-op tensors

    /// Compares a candidate's rows for one golden tensor at the positions both
    /// have. `candidate(position)` returns the candidate's row or nil.
    static func compareTensor(_ golden: GoldenMatrix, candidate: (Int) throws -> [Float]?) throws -> ParityMetrics.ErrorStats {
        var stats = ParityMetrics.ErrorStats()
        for (i, p) in golden.positions.enumerated() {
            guard let c = try candidate(p) else { continue }
            guard c.count == golden.cols else {
                throw GoldenError.sizeMismatch("position \(p): candidate \(c.count) vs golden \(golden.cols)")
            }
            golden.withRow(i) { g in c.withUnsafeBufferPointer { stats.add(golden: g, candidate: $0) } }
        }
        return stats
    }

    /// A component's output, fed the golden inputs, against its class's local
    /// tolerance: nil when within it, else the failure.
    static func localVerdict(_ stats: ParityMetrics.ErrorStats, class c: OpToleranceClass, name: String) -> String? {
        guard stats.count > 0 else { return "\(name): nothing compared" }
        let r = stats.relativeL2
        return r <= c.maxLocalRelativeL2 && !r.isNaN
            ? nil : String(format: "%@: rel L2 %.3g > %.3g (%@)", name, r, c.maxLocalRelativeL2, c.rawValue)
    }

    /// Compares two golden runs' tensor `name` at their common positions.
    static func compareTensor(named name: String, occurrence: Int = 0,
                              golden: GoldenPrompt, run: GoldenPrompt) throws -> ParityMetrics.ErrorStats {
        let g = try golden.tensor(named: name, occurrence: occurrence)
        let r = try run.tensor(named: name, occurrence: occurrence)
        return try compareTensor(g) { p in r.row(atPosition: p) }
    }
}
