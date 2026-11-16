import Testing
@testable import Mference

/// GGUF Flash-Next prefill tiles take the grouped LUT4 GEMM only where the
/// Apple10 TensorOps path is absent (D8), the shape is whole tiles, and the
/// tile passes the 8x padding rule (shared with Qwen 3.6's INT4 kernel), and
/// only in prefill chunks of at most 1,024 tokens.
@Suite struct FlashNextGroupedLUT4SelectionTests {
    @Test func selectedOnlyOffApple10() {
        #expect(FlashNextForwardRunner.usesSimdgroupLUT4RoutedExperts(supportsApple10TensorOps: false))
        #expect(!FlashNextForwardRunner.usesSimdgroupLUT4RoutedExperts(supportsApple10TensorOps: true))
    }

    @Test func unslothShapeIsWholeTiles() {
        let cfg = ArchConfig.qwen38FlashNextUnslothUDIQ3XXS
        #expect(SimdgroupGroupedRoutedMoE.supportsLUT4(d: cfg.hiddenSize, routedIntermediate: cfg.moeIntermediateSize))
    }

    /// Short prompts never qualify: 128 tokens x top-10 over 512 experts is
    /// about 2.5 rows per expert, 25x padding. About 410 tokens in one chunk
    /// reach the 8x rule.
    @Test func fillRuleAtFlashNextChunkSizes() {
        let rowsAt128 = [Int](repeating: 2, count: 8) + [Int](repeating: 3, count: 8)
        #expect(!PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM(pairCounts: rowsAt128))
        let rowsAt2048 = [Int](repeating: 40, count: 16)
        #expect(PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM(pairCounts: rowsAt2048))
    }

    /// Chunks above 1,024 tokens keep the row kernel; the cap is inclusive.
    @Test func groupedPathCappedAt1024TokenChunks() {
        #expect(FlashNextForwardRunner.groupedLUT4MaxPrefillChunkTokens == 1024)
        #expect(FlashNextForwardRunner.usesSimdgroupLUT4RoutedPrefill(chunkTokens: 128))
        #expect(FlashNextForwardRunner.usesSimdgroupLUT4RoutedPrefill(chunkTokens: 1024))
        #expect(!FlashNextForwardRunner.usesSimdgroupLUT4RoutedPrefill(chunkTokens: 1025))
        #expect(!FlashNextForwardRunner.usesSimdgroupLUT4RoutedPrefill(chunkTokens: 2048))
    }
}
