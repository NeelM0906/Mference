import Testing
@testable import Mference

/// Inkling 2 plans its prefill chunks balanced (`PrefillChunkPlanner
/// .balancedSpans`, through `Inkling2ForwardRunner.prefillSpans`): the same
/// number of chunks as before, sizes within one token of each other. Every
/// other family's runner keeps `PrefillChunkPlanner.spans`, which must plan
/// exactly as it did, for every length (the byte-identity runs go through it).
@Suite struct BalancedPrefillSpansTests {
    /// Every allowed size and one that is not, up to 10,000 tokens; the tiny
    /// sizes 1 and 3 up to 600 (their span lists grow with the length).
    static let cases: [(chunk: Int, lengths: ClosedRange<Int>)] =
        (RuntimeConfiguration.allowedPrefillChunkTokens + [1_000]).map { ($0, 0...10_000) } + [(1, 0...600), (3, 0...600)]

    /// `spans` as it was written at trunk eb8eb1e5, verbatim but for the name.
    static func oldSpans(tokenCount: Int, startPosition: Int, chunkTokens: Int) -> [PrefillChunkSpan] {
        let chunk = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        guard tokenCount > 0 else { return [] }
        var spans: [PrefillChunkSpan] = []
        var offset = 0
        while offset < tokenCount {
            let count = min(chunk, tokenCount - offset)
            let completed = offset + count
            spans.append(PrefillChunkSpan(tokenOffset: offset, tokenCount: count,
                                          startPosition: startPosition + offset, completedCount: completed))
            offset = completed
        }
        return spans
    }

    static func shape(_ spans: [PrefillChunkSpan]) -> [[Int]] {
        spans.map { [$0.tokenOffset, $0.tokenCount, $0.startPosition, $0.completedCount] }
    }

    @Test func everyOtherFamilysPlanIsUnchanged() {
        for (chunk, lengths) in Self.cases {
            for n in lengths {
                let start = n % 7 == 0 ? 0 : 37
                let now = PrefillChunkPlanner.spans(tokenCount: n, startPosition: start, chunkTokens: chunk)
                let before = Self.oldSpans(tokenCount: n, startPosition: start, chunkTokens: chunk)
                if Self.shape(now) != Self.shape(before) { Issue.record("spans changed: \(n) tokens at \(chunk)") }
            }
        }
    }

    @Test func balancedSpansKeepTheCountAndEvenOutTheSizes() {
        for (chunk, lengths) in Self.cases {
            let limit = min(chunk, PrefillRuntimeConfig.maxChunkTokens)
            for n in lengths {
                let start = n % 5 == 0 ? 0 : 1_234
                let spans = PrefillChunkPlanner.balancedSpans(tokenCount: n, startPosition: start, chunkTokens: chunk)
                let sizes = spans.map(\.tokenCount)
                let expectedCount = n == 0 ? 0 : (n + limit - 1) / limit
                var ok = spans.count == expectedCount
                    && sizes.reduce(0, +) == n
                    && (sizes.max() ?? 0) <= limit
                    && (sizes.max() ?? 0) - (sizes.min() ?? 0) <= 1
                    && sizes == sizes.sorted(by: >)
                var offset = 0
                for span in spans {
                    ok = ok && span.tokenOffset == offset && span.startPosition == start + offset
                        && span.completedCount == offset + span.tokenCount && span.tokenCount > 0
                    offset += span.tokenCount
                }
                if !ok { Issue.record("balanced spans wrong: \(n) tokens at \(chunk): \(sizes)") }
                // One chunk, the same as before.
                if n <= limit, Self.shape(spans) != Self.shape(Self.oldSpans(tokenCount: n, startPosition: start,
                                                                                chunkTokens: chunk)) {
                    Issue.record("single chunk differs: \(n) at \(chunk)")
                }
            }
        }
    }

    @Test func k0sServerPromptBalances() {
        #expect(PrefillChunkPlanner.balancedSpans(tokenCount: 4_111, startPosition: 0, chunkTokens: 2_048)
                    .map(\.tokenCount) == [1_371, 1_370, 1_370])
        #expect(PrefillChunkPlanner.spans(tokenCount: 4_111, startPosition: 0, chunkTokens: 2_048)
                    .map(\.tokenCount) == [2_048, 2_048, 15])
        #expect(PrefillChunkPlanner.balancedSpans(tokenCount: 4_097, startPosition: 0, chunkTokens: 2_048)
                    .map(\.tokenCount) == [1_366, 1_366, 1_365])
        #expect(PrefillChunkPlanner.balancedSpans(tokenCount: 4_096, startPosition: 0, chunkTokens: 2_048)
                    .map(\.tokenCount) == [2_048, 2_048])
        #expect(PrefillChunkPlanner.balancedSpans(tokenCount: 20_000, startPosition: 0, chunkTokens: 2_048)
                    .map(\.tokenCount) == Array(repeating: 2_000, count: 10))
    }

    /// The chunk is clamped as `spans` clamps it: above the planner's maximum
    /// to 4,096, at zero or below to 1.
    @Test func theChunkIsClampedAsBefore() {
        for (chunk, n) in [(5_000, 9_000), (4_097, 8_193), (0, 7), (-3, 5)] {
            let balanced = PrefillChunkPlanner.balancedSpans(tokenCount: n, startPosition: 0, chunkTokens: chunk)
            let clamped = max(1, min(chunk, PrefillRuntimeConfig.maxChunkTokens))
            #expect(balanced.count == (n + clamped - 1) / clamped, "\(chunk)")
            #expect(balanced.allSatisfy { $0.tokenCount <= clamped }, "\(chunk)")
            #expect(Self.shape(PrefillChunkPlanner.spans(tokenCount: n, startPosition: 0, chunkTokens: chunk))
                    == Self.shape(Self.oldSpans(tokenCount: n, startPosition: 0, chunkTokens: chunk)), "\(chunk)")
        }
        #expect(PrefillChunkPlanner.balancedSpans(tokenCount: 9_000, startPosition: 0, chunkTokens: 5_000)
                    .map(\.tokenCount) == [3_000, 3_000, 3_000])
    }

    /// The Inkling runner plans through `prefillSpans`, which is the balanced
    /// planner for every length and start.
    @Test func inkling2PlansBalanced() {
        for n in [1, 15, 2_048, 2_049, 4_111, 16_384, 40_960] {
            for start in [0, 100] {
                #expect(Self.shape(Inkling2ForwardRunner.prefillSpans(tokenCount: n, startPosition: start,
                                                                       chunkTokens: 2_048))
                        == Self.shape(PrefillChunkPlanner.balancedSpans(tokenCount: n, startPosition: start,
                                                                         chunkTokens: 2_048)))
            }
        }
    }
}
