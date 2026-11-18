import Foundation

/// Grouped GEMM computes whole 64-row matrix tiles per expert, so it pays only
/// while padding stays small. Measured on M2: ~17.5 us per padded row against
/// ~29.6 us per real row for the row kernel, a break-even near 1.69x padding.
enum PrefillGroupedExpertGate {
    static let tileRows = 64

    static func usesGroupedGEMM(pairCounts: [Int]) -> Bool {
        let real = pairCounts.reduce(0, +)
        guard real > 0 else { return false }
        let padded = pairCounts.reduce(0) { $0 + ($1 + tileRows - 1) / tileRows * tileRows }
        return padded * 2 <= real * 3
    }

    /// The simdgroup grouped kernel (`SimdgroupGroupedRoutedMoE`) costs about
    /// the same per 64-row block at any fill; on M1 Max it matches the row
    /// kernel near 8 real rows per block (16 experts: 1.30 ms per block
    /// against 1.01 ms at 7 rows and 1.46 ms at 10). So it takes tiles whose
    /// padding stays within 8x.
    static func usesSimdgroupGroupedGEMM(pairCounts: [Int]) -> Bool {
        let real = pairCounts.reduce(0, +)
        guard real > 0 else { return false }
        let padded = pairCounts.reduce(0) { $0 + ($1 + tileRows - 1) / tileRows * tileRows }
        return padded <= real * 8
    }
}
