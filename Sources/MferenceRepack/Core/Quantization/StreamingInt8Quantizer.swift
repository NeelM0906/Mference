import Foundation

/// Quantize-in-flight at INT8: BF16 source bytes sitting in bounded scratch,
/// rewritten in place into the MLX INT8 affine group-64 layout (one unsigned
/// byte per weight plus BF16 scales and BF16 biases).
///
/// The INT8 sibling of `StreamingInt4Quantizer`, and it exists for one
/// concrete reason: a community conversion of a vendor checkpoint is commonly
/// **not** uniform in width. mlx-community's Qwen 3.6 conversion keeps every
/// layer's `mlp.gate` (the MoE router) and `mlp.shared_expert_gate` at INT8
/// group-64 while everything else is INT4, and the extra bits are
/// load-bearing rather than ceremonial: an INT4 router changes roughly 13-18 %
/// of the selected top-8 expert set (§6 of `docs/QUANTIZER_QUALITY.md`).
/// Without this component the repacker could only emit a uniform-INT4 install,
/// which the Qwen 3.6 runner refuses at load and which would in any case make
/// a quantizer-quality comparison against that control measure routing
/// divergence rather than weight fidelity.
///
/// No new quantization math lives here: every group is handed to
/// `Int8AffineEncoder.encodeGroup`, the bit-exact twin of the runtime's
/// `Quantization.quantizeInt8Affine` reference. `Int8AffineStreamingParityTests`
/// enforces that equality *through this streaming path*, across awkward shapes
/// and tile sizes down to a single group per tile, exactly as
/// `Int4AffineStreamingParityTests` does for INT4 — the W2.1a-style bootstrap
/// gate that has to pass before any of this is trusted to write an install.
///
/// Only the component transform is provided. The row-pool transforms
/// (`quantizeRowsInPlace`, `quantizeRowBlocksInPlace`) serve Flash-Next's PLE
/// n-gram table, which is INT4 by policy and has no INT8 counterpart; adding
/// unused INT8 variants would be untested surface.
public enum StreamingInt8Quantizer {

    public static let groupSize = Int8AffineEncoder.groupSize          // 64
    /// BF16 bytes consumed per quantization group.
    public static let groupSourceBytes = groupSize * 2                 // 128
    /// Packed bytes produced per group: one per weight.
    public static let groupPackedBytes = groupSize                     // 64
    /// One BF16 scale (or bias) per group.
    public static let companionBytesPerGroup = 2

    /// Destination bytes one `component` emits per source group. The component
    /// enum itself is shared with the INT4 quantizer — the three co-planned
    /// outputs mean the same thing at both widths — but the weight-side byte
    /// count is width-specific, which is why it is a function here rather than
    /// a property on the enum.
    public static func destinationBytesPerGroup(
        _ component: StreamingInt4Quantizer.Component
    ) -> Int {
        switch component {
        case .weights: groupPackedBytes
        case .scales, .biases: companionBytesPerGroup
        }
    }

    // MARK: - Geometry

    /// `true` when `rowDim` values can be group-64 quantized at all.
    public static func isQuantizableRowDim(_ rowDim: Int) -> Bool {
        rowDim > 0 && rowDim % groupSize == 0
    }

    // MARK: - Component transform (resident and per-expert tensors)

    /// Rewrite `sourceCount` BF16 bytes at the head of `buffer` into this
    /// component's output, returning the byte count written.
    ///
    /// Groups never straddle rows (a quantizable tensor's last dimension is a
    /// multiple of 64), so the source is a flat run of groups and any
    /// 128-byte-aligned split is safe. Output is half the input on the weight
    /// side and far smaller on the companion sides, so the forward in-place
    /// pass never clobbers unread bytes.
    public static func quantizeComponentInPlace(
        _ component: StreamingInt4Quantizer.Component,
        buffer: UnsafeMutableRawBufferPointer,
        sourceCount: Int
    ) throws -> Int {
        guard sourceCount > 0, sourceCount % groupSourceBytes == 0 else {
            throw RepackError.configurationInvalid(
                detail: "quantize-int8-g64 needs \(groupSourceBytes)-byte aligned "
                    + "source chunks, got \(sourceCount)")
        }
        let groups = sourceCount / groupSourceBytes
        let outputCount = groups * destinationBytesPerGroup(component)
        guard sourceCount <= buffer.count, outputCount <= buffer.count else {
            throw RepackError.scratchExceeded(
                requested: max(sourceCount, outputCount), limit: buffer.count)
        }
        let bytes = buffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
        var group = [Float](repeating: 0, count: groupSize)
        var discardedPacked = [UInt8](repeating: 0, count: groupPackedBytes)
        var scale: UInt16 = 0
        var bias: UInt16 = 0
        for g in 0..<groups {
            widenBF16(from: bytes + g * groupSourceBytes, into: &group, count: groupSize)
            group.withUnsafeBufferPointer { values in
                switch component {
                case .weights:
                    Int8AffineEncoder.encodeGroup(values.baseAddress!,
                                                  packed: bytes + g * groupPackedBytes,
                                                  scale: &scale,
                                                  bias: &bias)
                case .scales, .biases:
                    discardedPacked.withUnsafeMutableBufferPointer { sink in
                        Int8AffineEncoder.encodeGroup(values.baseAddress!,
                                                      packed: sink.baseAddress!,
                                                      scale: &scale,
                                                      bias: &bias)
                    }
                }
            }
            switch component {
            case .weights:
                break
            case .scales:
                writeLittleEndian16(scale, to: bytes + g * companionBytesPerGroup)
            case .biases:
                writeLittleEndian16(bias, to: bytes + g * companionBytesPerGroup)
            }
        }
        return outputCount
    }

    // MARK: - Internals

    /// Little-endian BF16 bit patterns -> Float. Source bytes may be unaligned,
    /// so the halves are assembled by hand.
    @inline(__always)
    private static func widenBF16(from source: UnsafePointer<UInt8>,
                                  into values: inout [Float],
                                  count: Int) {
        values.withUnsafeMutableBufferPointer { out in
            for i in 0..<count {
                let bits = UInt32(source[i * 2]) | UInt32(source[i * 2 + 1]) << 8
                out[i] = Float(bitPattern: bits << 16)
            }
        }
    }

    @inline(__always)
    private static func writeLittleEndian16(_ value: UInt16,
                                            to bytes: UnsafeMutablePointer<UInt8>) {
        bytes[0] = UInt8(truncatingIfNeeded: value)
        bytes[1] = UInt8(truncatingIfNeeded: value >> 8)
    }
}
