import Foundation
import Metal

/// Resident tensors of a `qwen38flashnextudiq3xxs` install (WS4 U4): the
/// Unsloth UD-IQ3_XXS GGUF of Qwen3.8-Flash-Next under its GGUF names, with
/// llama.cpp's converter transforms kept in the bytes (U2 §7).
///
/// The accessors here back the shared Flash-Next accessors (`Model.qProj`,
/// `hcMixDown`, `linearALog`, ...) for this family, so
/// `FlashNextForwardRunner` sees the same API as for the vendor install. What
/// differs, and how each is consumed:
/// - INT8-2L (`0x80 | ggml type`) and BF16 matrices pass through untouched;
///   `FlashNextWeightMatrix.fromGGUF` turns them into matrices.
/// - F32 2-D matrices and conv taps, and the 1-D vectors U2's R8 scan found
///   BF16-exact, are narrowed to BF16 at load. The narrowing re-proves
///   exactness and throws on the first value that is not (`.exact`).
/// - The `+1` norms are not BF16-exact; they stay F32 and the norm kernels
///   read them through their F32-weight function constants (U5 step 1).
/// - `ssm_a` (`-exp(A_log)` in F32) is returned as stored; the GDN kernels
///   read it directly (`GDN.CheckpointLayout.gguf`).
extension Model {

    /// Whether this load is the WS4 GGUF Flash-Next install.
    var isFlashNextGGUF: Bool { config.family == .qwen38flashnextudiq3xxs }

    enum GGUFNarrowing: String {
        /// Every F32 value must already be a BF16 value; throw otherwise.
        case exact
        /// Round to nearest even.
        case nearestEven
    }

    /// `blk.L.<suffix>`.
    static func ggufLayerName(_ L: Int, _ suffix: String) -> String { "blk.\(L).\(suffix)" }

    /// A GGUF resident tensor as BF16: BF16 passes through; F32 is narrowed
    /// into a cached shared buffer (`mode` decides exact vs rounded). `shape`
    /// overrides the stored shape (a `[n]` vector used as a `[1, n]` matrix).
    func ggufBF16(name: String, mode: GGUFNarrowing,
                  shape: (UInt32, UInt32, UInt32, UInt32)? = nil) throws -> TensorView {
        let source = try resident(name: name)
        switch source.dtype {
        case 1:
            guard let shape else { return source }
            return TensorView(buffer: source.buffer, offset: source.offset, length: source.length,
                              scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                              shape: shape, dtype: 1)
        case 3:
            break
        default:
            throw ModelError.indexCorrupt(
                detail: "\(name) is dtype \(source.dtype); expected F32 (3) or BF16 (1)")
        }
        let key = "gguf-bf16-\(mode.rawValue):" + name
        if let cached = streamersQueue.sync(execute: { convertedBox.views[key] }) {
            return cached
        }
        let count = Int(source.length) / MemoryLayout<Float>.stride
        guard count > 0, UInt64(count * MemoryLayout<Float>.stride) == source.length,
              let buffer = device.makeBuffer(length: count * MemoryLayout<UInt16>.stride,
                                             options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        try Self.narrowF32ToBF16(UnsafeRawPointer(source.buffer.contents().advanced(by: Int(source.offset))),
                                 count: count,
                                 into: buffer.contents().assumingMemoryBound(to: UInt16.self),
                                 mode: mode, name: name)
        let view = TensorView(
            buffer: buffer, offset: 0, length: UInt64(count * MemoryLayout<UInt16>.stride),
            scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
            shape: shape ?? source.shape, dtype: 1)
        streamersQueue.sync { convertedBox.views[key] = view }
        return view
    }

    /// F32 → BF16, element by element. `.exact` throws on the first value
    /// whose low 16 bits are not zero; `.nearestEven` rounds.
    static func narrowF32ToBF16(_ src: UnsafeRawPointer, count: Int,
                                into dst: UnsafeMutablePointer<UInt16>,
                                mode: GGUFNarrowing, name: String) throws {
        for i in 0..<count {
            let bits = src.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self)
            switch mode {
            case .exact:
                guard bits & 0xFFFF == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(name)[\(i)] = \(Float(bitPattern: bits)) is not BF16-exact")
                }
                dst[i] = UInt16(truncatingIfNeeded: bits >> 16)
            case .nearestEven:
                dst[i] = Quantization.bf16Bits(Float(bitPattern: bits))
            }
        }
    }

    /// A `+1` norm vector as stored: F32 `[n]`, not BF16-exact, read directly
    /// by the F32-weight norm kernels (`RMSNorm(groupedWeightF32:)`,
    /// `PrefillQKVEpilogue(normWeightsF32:)`, `FlashNextIndexer(normWeightsF32:)`).
    func ggufF32Norm(name: String) throws -> TensorView {
        let view = try resident(name: name)
        guard view.dtype == 3, view.length > 0, view.length % 4 == 0, view.offset % 4 == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) must be an F32 vector at a 4-byte offset, got dtype \(view.dtype), "
                    + "\(view.length) B at \(view.offset)")
        }
        return view
    }

    /// `ssm_a` as stored: F32 `[numVHeads]`, `-exp(A_log)`.
    func ggufSSMA(layer L: Int) throws -> TensorView {
        let name = Self.ggufLayerName(L, "ssm_a")
        let view = try resident(name: name)
        let heads = config.linearAttention.numVHeads
        guard view.dtype == 3, view.length == UInt64(heads * 4), view.offset % 4 == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) must be F32 [\(heads)] at a 4-byte offset, got dtype \(view.dtype), "
                    + "\(view.length) B at \(view.offset)")
        }
        return view
    }

    /// The hyper-connection site's GGUF stem.
    static func ggufHCStem(_ site: HyperConnectionSite) -> String {
        switch site {
        case .attention: "hc_attn"
        case .mlp: "hc_ffn"
        }
    }

    /// The PLE n-gram hash constants of a GGUF install: from the manifest
    /// (R6), with no resident I64 copy allowed alongside.
    func ggufPleHash(layer L: Int) throws -> ManifestPleHash {
        guard let entry = manifest.plePool?.layers.first(where: { $0.layer == L }) else {
            throw ModelError.plePoolMissing(layer: L)
        }
        guard let hash = entry.hash else {
            throw ModelError.plePoolInvalid(detail: "layer \(L) carries no manifest hash constants")
        }
        // Exactly one source: the only I64 tensors an install has are these
        // constants, so any resident I64 entry is a second copy.
        if residentIndex.entries.values.contains(where: { $0.dtype == 4 }) {
            throw ModelError.plePoolInvalid(
                detail: "layer \(L) carries hash constants in both the manifest and the resident index")
        }
        guard hash.layerMultipliers.count == hash.ngramSize,
              hash.headOffsets.count == hash.headVocabSizes.count,
              !hash.headOffsets.isEmpty else {
            throw ModelError.plePoolInvalid(detail: "layer \(L) hash constants have inconsistent lengths")
        }
        return hash
    }
}
