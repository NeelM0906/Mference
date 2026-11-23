import Foundation
import Testing
@testable import Mference

/// The cost of one `push` by position, for the incremental `MFDetokenizer`
/// and the full-decode `QuadraticDetokenizerOracle`, on the same token
/// stream. Opt-in (`MFERENCE_DETOK_BENCH=1`). `MFERENCE_DETOK_BENCH_TOKENIZER`
/// names a tokenizer folder (an install's, files only); the default is the
/// Inkling 2 fixture. It asserts nothing about timing, only that the two
/// emit the same chunks.
@Suite(.serialized) struct DetokenizerCostBench {
    static let environment = ProcessInfo.processInfo.environment

    @Test(.enabled(if: environment["MFERENCE_DETOK_BENCH"] == "1"))
    func pushCostByPosition() async throws {
        let folder = try Self.environment["MFERENCE_DETOK_BENCH_TOKENIZER"].map { URL(fileURLWithPath: $0) }
            ?? DetokenizerLinearOracleTests.fixture("Inkling2Tokenizer")
        let tok = try await MFTokenizer.load(from: folder, family: nil)
        let paragraph = "The user wants the total of three invoices. First, 128.40 plus 75.10 is 203.50; "
            + "adding 46.25 gives 249.75. Check the rounding, and confirm no line was counted twice. "
            + "数字を確認します。 Überprüfe die Summe. "
        var ids: [Int32] = []
        while ids.count < 32_100 { ids += tok.encode(paragraph, addBOS: false) }
        let marks = [100, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
        let oracleLimit = Int(Self.environment["MFERENCE_DETOK_BENCH_ORACLE_LIMIT"] ?? "16000") ?? 16_000
        func log(_ s: String) { FileHandle.standardError.write(Data(("[detok bench] " + s + "\n").utf8)) }
        log("tokenizer \(folder.lastPathComponent) (\(folder.deletingLastPathComponent().lastPathComponent)), "
            + "\(ids.count) tokens")

        func measure(_ name: String, limit: Int, push: (Int32) -> String) -> [Int: Double] {
            var result: [Int: Double] = [:]
            var windowStart = ContinuousClock.now
            for (i, id) in ids.prefix(limit + 50).enumerated() {
                if marks.contains(i) { windowStart = .now }
                _ = push(id)
                if marks.contains(i - 49) {
                    let d = windowStart.duration(to: .now)
                    result[i - 49] = (Double(d.components.attoseconds) / 1e18 + Double(d.components.seconds)) * 1000 / 50
                }
            }
            for mark in marks where result[mark] != nil {
                log(String(format: "%@ push at position %d: %.4f ms", name, mark, result[mark]!))
            }
            return result
        }
        var linear = MFDetokenizer(tokenizer: tok)
        _ = measure("incremental", limit: 32_000) { linear.push($0) }
        var oracle = QuadraticDetokenizerOracle(tokenizer: tok)
        _ = measure("full decode", limit: oracleLimit) { oracle.push($0) }

        // Same chunks over the oracle's range.
        var a = QuadraticDetokenizerOracle(tokenizer: tok), b = MFDetokenizer(tokenizer: tok)
        for id in ids.prefix(min(oracleLimit, 4_000)) {
            #expect(a.push(id).utf8.elementsEqual(b.push(id).utf8))
        }
    }
}
