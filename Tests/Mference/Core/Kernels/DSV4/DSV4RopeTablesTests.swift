import Testing

@testable import Mference

/// Reference values computed with the transformers `_compute_yarn_parameters`
/// implementation (truncate = true) at the production compress-rope shape:
/// base 160000, dim 64, factor 16, beta_fast 32, beta_slow 1,
/// original_max_position_embeddings 65536. Correction range is dims 15...25:
/// pairs 0-15 keep the plain frequency, 25-31 are fully interpolated
/// (freq / 16), and the span between ramps linearly.
struct DSV4RopeTablesTests {

    @Test func plainTableMatchesThetaPowers() {
        let t = DSV4RopeTables.plainInvFreq(theta: 10_000.0, ropeDim: 64)
        #expect(t.count == 32)
        #expect(abs(t[0] - 1.0) < 1e-7)
        #expect(abs(t[16] - 1e-2) < 1e-8)
        #expect(abs(t[31] - Float(1.0 / 7_498.94)) < 1e-9)
    }

    @Test func yarnTableMatchesReference() {
        let t = DSV4RopeTables.yarnInvFreq(
            theta: 160_000.0, ropeDim: 64,
            factor: 16.0, originalMaxPositions: 65_536,
            betaFast: 32.0, betaSlow: 1.0)
        #expect(t.count == 32)
        let expected: [Int: Float] = [
            0: 1.0000000000e+00,
            7: 7.2710771672e-02,
            8: 5.0000000000e-02,
            15: 3.6355385836e-03,
            16: 2.2656250000e-03,
            24: 1.9531250000e-05,
            25: 5.3723126714e-06,
            31: 5.6805290369e-07,
        ]
        for (i, want) in expected {
            let got = t[i]
            #expect(abs(got - want) <= abs(want) * 1e-5,
                    "pair \(i): got \(got), want \(want)")
        }
        // The unramped low pairs equal the plain table; the fully
        // interpolated tail is exactly plain / factor.
        let plain = DSV4RopeTables.plainInvFreq(theta: 160_000.0, ropeDim: 64)
        for i in 0...15 { #expect(t[i] == plain[i]) }
        for i in 25...31 {
            #expect(abs(t[i] - plain[i] / 16.0) <= plain[i] * 1e-6)
        }
    }
}
