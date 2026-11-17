import Testing
@testable import Mference

/// GGUF Flash-Next prefill tiles take the grouped LUT4 GEMM only where the
/// Apple10 TensorOps path is absent (D8), the shape is whole tiles, and the
/// tile passes the 8x padding rule (shared with Qwen 3.6's INT4 kernel), and
/// only in 512- and 1,024-token prefill chunks.
@Suite struct FlashNextGroupedLUT4SelectionTests {
    @Test func selectedOnlyOffApple10() {
        #expect(FlashNextForwardRunner.usesSimdgroupLUT4RoutedExperts(supportsApple10TensorOps: false))
        #expect(!FlashNextForwardRunner.usesSimdgroupLUT4RoutedExperts(supportsApple10TensorOps: true))
    }

    @Test func unslothShapeIsWholeTiles() {
        let cfg = ArchConfig.qwen38FlashNextUnslothUDIQ3XXS
        #expect(SimdgroupGroupedRoutedMoE.supportsLUT4(d: cfg.hiddenSize, routedIntermediate: cfg.moeIntermediateSize))
    }

    /// The 8x rule on its own: an evenly spread 128-token tile (about 2.5 rows
    /// per expert, 25x padding) fails and a 2,048-token one passes. The rule
    /// is per tile, so an uneven sparse chunk can still pass on one tile; the
    /// chunk range below is what keeps small chunks on the row kernel.
    @Test func fillRuleAtFlashNextChunkSizes() {
        let rowsAt128 = [Int](repeating: 2, count: 8) + [Int](repeating: 3, count: 8)
        #expect(!PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM(pairCounts: rowsAt128))
        let rowsAt2048 = [Int](repeating: 40, count: 16)
        #expect(PrefillGroupedExpertGate.usesSimdgroupGroupedGEMM(pairCounts: rowsAt2048))
    }

    /// Only the gated 512 and 1,024 chunks may take the grouped path. Every
    /// other allowed chunk keeps the row kernel, including the 32 and 64
    /// chunks the CLI picks for the 5-token and 62-token class E prompts and
    /// the server's 128 and 4,096 tiers.
    @Test func groupedPathOnlyAt512And1024TokenChunks() {
        #expect(FlashNextForwardRunner.groupedLUT4MinPrefillChunkTokens == 512)
        #expect(FlashNextForwardRunner.groupedLUT4MaxPrefillChunkTokens == 1024)
        for chunk in RuntimeConfiguration.allowedPrefillChunkTokens {
            #expect(FlashNextForwardRunner.usesSimdgroupLUT4RoutedPrefill(chunkTokens: chunk)
                        == (chunk == 512 || chunk == 1024), "chunk \(chunk)")
        }
        #expect(!FlashNextForwardRunner.usesSimdgroupLUT4RoutedPrefill(chunkTokens: 511))
        #expect(!FlashNextForwardRunner.usesSimdgroupLUT4RoutedPrefill(chunkTokens: 1025))
    }
}
