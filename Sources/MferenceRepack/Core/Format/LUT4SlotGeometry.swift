import Foundation

/// Slot geometry of the LUT4 routed-expert execution format (WS4 U1 §3a).
/// The installer publishes it per layer beside the disk stride (ruling R2):
/// a slot holds LUT4, not the GGUF bytes, so it is sized from this geometry,
/// never from the disk stride (a LUT4 slot is larger than every disk stride
/// of the UD-IQ3_XXS file).
///
/// Per projection of `n` weights: 4-bit indices (`n / 2` bytes), an FP16 `D`
/// per `BD` weights and, where the source has one, an int8 `M` per `BM`
/// weights. The 16-entry tables and `K` are per tensor (per layer), not per
/// expert, so they are not in the slot.
enum LUT4SlotGeometry {
    /// `(BD, BM)` per source type; `BM == 0` means no `M` (M ≡ 1).
    static func blocking(_ t: GGMLType) -> (bd: UInt64, bm: UInt64)? {
        switch t {
        case .iq2_s: (256, 16)
        case .iq3_xxs: (256, 32)
        case .iq4_xs: (256, 32)
        case .iq4_nl: (32, 0)
        case .q2_0: (64, 0)
        default: nil
        }
    }

    static func projectionBytes(_ t: GGMLType, weights n: UInt64) -> UInt64? {
        guard let (bd, bm) = blocking(t), n % 2 == 0, n % bd == 0, bm == 0 || n % bm == 0 else { return nil }
        return n / 2 + 2 * (n / bd) + (bm == 0 ? 0 : n / bm)
    }

    /// Page-aligned slot stride of one expert of `layer`.
    static func slotStride(_ layer: LayerFilePlan) throws -> UInt64 {
        var total: UInt64 = 0
        for slice in layer.subTensors {
            let id = UInt32(slice.dtype & ~FlashNextGGUFPlanner.quantizedDtypeBase)
            let n = slice.logicalShape.reduce(1, *)
            guard slice.dtype & FlashNextGGUFPlanner.quantizedDtypeBase != 0,
                  let t = GGMLType(rawValue: id), let bytes = projectionBytes(t, weights: n) else {
                throw RepackError.configurationInvalid(
                    detail: "layer \(layer.layerIndex) \(slice.role) has no LUT4 slot geometry")
            }
            total += bytes
        }
        return RepackPlanner.roundUpToPage(total)
    }
}
