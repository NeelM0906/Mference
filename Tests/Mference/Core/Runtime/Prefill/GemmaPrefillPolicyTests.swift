import Testing
@testable import Mference

@Suite struct GemmaPrefillPolicyTests {
    private static let originalID = "mlx-community/gemma-4-26b-a4b-it-4bit"

    @Test func qatPrefillMatmulsAndFullAttentionUseNormalKernels() {
        let policy = GemmaPrefillPolicy(modelID: CheckpointIdentity.gemma4QAT, environment: [:])
        #expect(policy.sourceFP16)
        #expect(!policy.prefillMatmulSourceFP16)
        #expect(!policy.prefillAttentionSourceFP16)
        #expect(policy.batchedExperts)
    }

    @Test func exactPrefillSwitchRestoresTheShippedQATProfile() {
        let policy = GemmaPrefillPolicy(modelID: CheckpointIdentity.gemma4QAT,
                                        environment: ["MFERENCE_QAT_EXACT_PREFILL": "1"])
        #expect(policy.sourceFP16)
        #expect(policy.prefillMatmulSourceFP16)
        #expect(policy.prefillAttentionSourceFP16)
        // Source arithmetic has no grouped-GEMM form; the switch restores
        // the whole shipped prefill, including per-row experts.
        #expect(!policy.batchedExperts)
    }

    @Test func originalGemmaKeepsNormalArithmeticAndBatchesExperts() {
        let policy = GemmaPrefillPolicy(modelID: Self.originalID,
                                        environment: ["MFERENCE_QAT_EXACT_PREFILL": "1"])
        #expect(!policy.sourceFP16)
        #expect(!policy.prefillMatmulSourceFP16)
        #expect(!policy.prefillAttentionSourceFP16)
        #expect(policy.batchedExperts)
    }

    @Test(arguments: [CheckpointIdentity.gemma4QAT, originalID])
    func legacySwitchKeepsPerRowExpertsForBothCheckpoints(modelID: String) {
        let policy = GemmaPrefillPolicy(modelID: modelID,
                                        environment: ["MFERENCE_GEMMA_PREFILL_LEGACY": "1"])
        #expect(!policy.batchedExperts)
        #expect(!policy.prefillMatmulSourceFP16)
        #expect(!policy.prefillAttentionSourceFP16)
    }
}

/// Grouped GEMM computes whole 64-row tiles per expert, so it only pays while
/// padding stays small. Measured on M2: ~17.5 us per padded row against
/// ~29.6 us per real row for the row kernel (break-even padding 1.69x).
@Suite struct PrefillGroupedExpertGateTests {
    @Test(arguments: [48, 64, 96, 188])
    func wellFilledTilesUseGroupedGEMM(rows: Int) {
        #expect(PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: Array(repeating: rows, count: 8)))
    }

    @Test(arguments: [1, 8, 27, 40])
    func sparseTilesKeepTheRowKernel(rows: Int) {
        #expect(!PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: Array(repeating: rows, count: 8)))
    }

    @Test func rowsJustPastATileBoundaryKeepTheRowKernel() {
        // 70 rows pad to 128: 1.83x the real work.
        #expect(!PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: Array(repeating: 70, count: 8)))
    }

    @Test func skewedTilesAreJudgedByTheirTotalPadding() {
        #expect(!PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: [200, 10, 10, 10, 10, 10, 10, 10]))
        #expect(PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: [200, 190, 180, 60, 64, 128, 100, 90]))
    }

    @Test func emptyTilesKeepTheRowKernel() {
        #expect(!PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: []))
        #expect(!PrefillGroupedExpertGate.usesGroupedGEMM(pairCounts: [0, 0]))
    }
}
