import Foundation

/// The GGML tensor formats an install copied from a GGUF may carry, with the
/// block geometry of `ggml/src/ggml-common.h` (llama.cpp tag b12635). The
/// installer copies GGUF bytes unchanged, so a tensor's bytes are exactly
/// these blocks, row after row, in the GGUF's own order.
///
/// - `q8_0`: 32 values per 34-byte block, an F16 scale `d` then 32 int8 codes;
///   value = `d * q`.
/// - `nvfp4`: 64 values per 36-byte block, four UE4M3 scales (one per 16
///   values) then 32 bytes of E2M1 codes, two per byte, the even value in the
///   low nibble. A tensor-level or per-expert F32 factor (`*.scale`) multiplies
///   the product, as llama.cpp does.
public enum GGMLBlockFormat: UInt32, Sendable, Hashable, CaseIterable {
    case f32 = 0
    case f16 = 1
    case q8_0 = 8
    case bf16 = 30
    case nvfp4 = 40

    /// Values per block (`QK*`; 1 for the plain float formats).
    public var blockElements: Int {
        switch self {
        case .f32, .f16, .bf16: 1
        case .q8_0: 32
        case .nvfp4: 64
        }
    }

    /// Bytes per block (`sizeof(block_*)`, or the element size).
    public var blockBytes: Int {
        switch self {
        case .f32: 4
        case .f16, .bf16: 2
        case .q8_0: 34
        case .nvfp4: 36
        }
    }

    /// The name `layout.json` and `arch.routedExpertFormats` use.
    public var name: String {
        switch self {
        case .f32: "F32"
        case .f16: "F16"
        case .q8_0: "Q8_0"
        case .bf16: "BF16"
        case .nvfp4: "NVFP4"
        }
    }

    public init?(name: String) {
        guard let format = Self.allCases.first(where: { $0.name == name }) else { return nil }
        self = format
    }

    /// Marker bit of a resident index dtype that holds native GGML blocks:
    /// `0x40 | ggml type id`. The plain float formats keep the codes every
    /// install uses (3 = F32, 2 = F16, 1 = BF16).
    public static let blockDtypeBase: UInt8 = 0x40

    /// The resident index dtype byte for a tensor in this format.
    public var residentDtype: UInt8 {
        switch self {
        case .f32: 3
        case .f16: 2
        case .bf16: 1
        case .q8_0, .nvfp4: Self.blockDtypeBase | UInt8(rawValue)
        }
    }

    /// The format a resident index dtype byte names, or nil for a dtype that
    /// is not one of these (an affine INT4 `U32` tensor, an integer table).
    public init?(residentDtype dtype: UInt8) {
        guard let format = Self.allCases.first(where: { $0.residentDtype == dtype }) else { return nil }
        self = format
    }

    /// Bytes of one row of `elements` values, or nil when the row is not a
    /// whole number of blocks.
    public func rowBytes(elements: Int) -> Int? {
        guard elements > 0, elements % blockElements == 0 else { return nil }
        return elements / blockElements * blockBytes
    }
}
