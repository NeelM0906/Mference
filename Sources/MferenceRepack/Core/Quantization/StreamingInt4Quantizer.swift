import Foundation

/// Quantize-in-flight: BF16 source bytes sitting in bounded scratch, rewritten
/// in place into the MLX INT4 affine group-64 layout (U32-packed nibbles plus
/// BF16 scales and BF16 biases).
///
/// This is Workstream 2's deliverable 1 (spec
/// `docs/superpowers/specs/2026-08-08-family-bringup-kit-design.md`). It exists
/// so the repacker can install a vendor's original BF16 checkpoint rather than
/// only re-laying-out somebody else's pre-quantized MLX conversion. No new
/// quantization math lives here: every group is handed to
/// `Int4AffineEncoder.encodeGroup`, the bit-exact twin of the runtime's
/// `Quantization.quantizeInt4Affine` reference.
///
/// The transforms run inside `HTTPRangeSourceByteProvider`'s write tile, so a
/// whole tensor is never resident: the caller streams `sourceCount` bytes at a
/// time (always a multiple of the transform's input unit) and each call
/// rewrites that scratch in place.
///
/// # Quality gates
///
/// Gate **W2.1a** (bit parity against the runtime's reference quantizer,
/// including through this streaming path) is enforced by
/// `Int4AffineStreamingParityTests` and `Int4AffineEncoderParityTests`. Its
/// INT8 sibling is `Int8AffineStreamingParityTests`.
///
/// Gate **W2.1b is CLOSED on both halves as of 2026-09-02**, measured against
/// mlx-community's independent conversion of `Qwen/Qwen3.6-35B-A3B`. Full
/// method and numbers: `docs/QUANTIZER_QUALITY.md`.
///
/// * **Weight level.** Over 124 sampled INT4 tensors this encoder's relative
///   Frobenius error against the BF16 source is 0.09612 mean, versus the
///   control's 0.09648 — better on 118 of 124. The packed bytes are *not*
///   bit-identical and cannot be: MLX snaps its affine grid so 0.0 is exactly
///   representable and anchors on the larger-magnitude endpoint, while this
///   encoder uses a plain min/max grid. The one place that costs us is tensors
///   carrying a large mass of near-exact zeros inside live groups (layer-0
///   routed `down_proj`, up to 1.54x worse). `Int4AffineEncoderConventionTests`
///   locks the convention so adopting the snap has to be a decision.
/// * **Model level.** Against a noise floor of *exactly zero* (temperature-0
///   decode is deterministic; the repeat dumps are byte-identical), the two
///   installs agree on 86.3 % of top-1 tokens and 81.3 % of top-5 sets over
///   882 teacher-forced positions, at a median per-position KL of 0.036 nats.
///   Every greedy divergence lands at a control-side top-2 margin below the
///   median inter-install logit difference.
///
/// Weights produced by this path are therefore validated at both levels for a
/// family whose install reproduces its control's *whole* configuration — not
/// the quantizer alone. Closing W2.1b surfaced two defects that had nothing to
/// do with quantization and that every other check passed: resident tensor
/// naming, and a missing RMSNorm `1 + w` fold. Both are documented in §7 of
/// `docs/QUANTIZER_QUALITY.md`, and the lesson generalises — a new family on
/// this path inherits the *quantizer's* validation, not a guarantee that its
/// own conventions match its runner's.
public enum StreamingInt4Quantizer {

    public static let groupSize = Int4AffineEncoder.groupSize          // 64
    /// BF16 bytes consumed per quantization group.
    public static let groupSourceBytes = groupSize * 2                 // 128
    /// Packed-nibble bytes produced per group.
    public static let groupPackedBytes = groupSize / 2                 // 32
    /// One BF16 scale (or bias) per group.
    public static let companionBytesPerGroup = 2

    /// Which of the three co-planned outputs a transform emits. All three read
    /// the *same* BF16 source range; the range planner emits one copy per
    /// component so the coalescer still downloads those bytes exactly once.
    public enum Component: String, Sendable, Equatable {
        case weights
        case scales
        case biases

        var destinationBytesPerGroup: Int {
            switch self {
            case .weights: groupPackedBytes
            case .scales, .biases: companionBytesPerGroup
            }
        }
    }

    // MARK: - Geometry

    /// Bytes one quantized row of `rowDim` values occupies as a self-contained
    /// record: packed nibbles, then that row's BF16 scales, then its biases.
    public static func rowRecordBytes(rowDim: Int) -> Int {
        rowDim / 2 + 2 * (rowDim / groupSize) * companionBytesPerGroup
    }

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
    /// 128-byte-aligned split is safe. Output is always smaller than input, so
    /// the forward in-place pass never clobbers unread bytes.
    public static func quantizeComponentInPlace(
        _ component: Component,
        buffer: UnsafeMutableRawBufferPointer,
        sourceCount: Int
    ) throws -> Int {
        guard sourceCount > 0, sourceCount % groupSourceBytes == 0 else {
            throw RepackError.configurationInvalid(
                detail: "quantize-int4-g64 needs \(groupSourceBytes)-byte aligned "
                    + "source chunks, got \(sourceCount)")
        }
        let groups = sourceCount / groupSourceBytes
        let outputCount = groups * component.destinationBytesPerGroup
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
                    Int4AffineEncoder.encodeGroup(values.baseAddress!,
                                                  packed: bytes + g * groupPackedBytes,
                                                  scale: &scale,
                                                  bias: &bias)
                case .scales, .biases:
                    discardedPacked.withUnsafeMutableBufferPointer { sink in
                        Int4AffineEncoder.encodeGroup(values.baseAddress!,
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

    // MARK: - Row transforms (the PLE n-gram row pool)

    /// Rewrite a dense run of BF16 rows into a dense run of self-contained
    /// quantized row records.
    ///
    /// A record is always smaller than its BF16 source row (0.5625x), so the
    /// forward pass is safe provided each row is staged before its record is
    /// written — which it is.
    public static func quantizeRowsInPlace(
        buffer: UnsafeMutableRawBufferPointer,
        sourceCount: Int,
        rowSourceBytes: Int
    ) throws -> Int {
        let rowDim = try validatedRowDim(rowSourceBytes)
        guard sourceCount > 0, sourceCount % rowSourceBytes == 0 else {
            throw RepackError.configurationInvalid(
                detail: "quantize-int4-g64-rows needs \(rowSourceBytes)-byte aligned "
                    + "source chunks, got \(sourceCount)")
        }
        let rows = sourceCount / rowSourceBytes
        let recordBytes = rowRecordBytes(rowDim: rowDim)
        let outputCount = rows * recordBytes
        guard sourceCount <= buffer.count, outputCount <= buffer.count else {
            throw RepackError.scratchExceeded(
                requested: max(sourceCount, outputCount), limit: buffer.count)
        }
        let bytes = buffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
        var staging = [Float](repeating: 0, count: rowDim)
        for row in 0..<rows {
            widenBF16(from: bytes + row * rowSourceBytes, into: &staging, count: rowDim)
            staging.withUnsafeBufferPointer { values in
                encodeRowRecord(values.baseAddress!,
                                rowDim: rowDim,
                                into: bytes + row * recordBytes)
            }
        }
        return outputCount
    }

    /// Rewrite whole page-aligned row *blocks*: `rowsPerBlock` BF16 rows in,
    /// `blockStride` bytes out — the records packed densely, then zero padding
    /// so the next block starts on a page boundary and no record ever straddles
    /// a page.
    ///
    /// A block's output can exceed its input when the block is mostly padding,
    /// so the pass walks blocks backwards in that case. Either way the whole
    /// block is staged before any of it is written.
    public static func quantizeRowBlocksInPlace(
        buffer: UnsafeMutableRawBufferPointer,
        sourceCount: Int,
        rowSourceBytes: Int,
        rowsPerBlock: Int,
        blockStride: UInt64
    ) throws -> Int {
        let rowDim = try validatedRowDim(rowSourceBytes)
        let recordBytes = rowRecordBytes(rowDim: rowDim)
        guard rowsPerBlock > 0,
              let stride = Int(exactly: blockStride),
              rowsPerBlock <= Int.max / max(rowSourceBytes, 1),
              stride >= rowsPerBlock * recordBytes else {
            throw RepackError.configurationInvalid(
                detail: "quantize-int4-g64-row-blocks geometry is invalid "
                    + "(rowsPerBlock \(rowsPerBlock), blockStride \(blockStride))")
        }
        let blockSourceBytes = rowsPerBlock * rowSourceBytes
        guard sourceCount > 0, sourceCount % blockSourceBytes == 0 else {
            throw RepackError.configurationInvalid(
                detail: "quantize-int4-g64-row-blocks needs \(blockSourceBytes)-byte "
                    + "aligned source chunks, got \(sourceCount)")
        }
        let blocks = sourceCount / blockSourceBytes
        let outputCount = blocks * stride
        guard sourceCount <= buffer.count, outputCount <= buffer.count else {
            throw RepackError.scratchExceeded(
                requested: max(sourceCount, outputCount), limit: buffer.count)
        }
        let bytes = buffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
        var staging = [Float](repeating: 0, count: rowsPerBlock * rowDim)
        let forward = stride <= blockSourceBytes
        for step in 0..<blocks {
            let block = forward ? step : blocks - 1 - step
            widenBF16(from: bytes + block * blockSourceBytes,
                      into: &staging,
                      count: rowsPerBlock * rowDim)
            let destination = bytes + block * stride
            staging.withUnsafeBufferPointer { values in
                for row in 0..<rowsPerBlock {
                    encodeRowRecord(values.baseAddress! + row * rowDim,
                                    rowDim: rowDim,
                                    into: destination + row * recordBytes)
                }
            }
            let used = rowsPerBlock * recordBytes
            if used < stride {
                memset(destination + used, 0, stride - used)
            }
        }
        return outputCount
    }

    /// Re-block a run of BF16 rows without quantizing them: the rows are copied
    /// verbatim and each block is zero-padded out to `blockStride`.
    ///
    /// This is the row-pool path for a table whose width the group size does
    /// not divide — Qwen3.8-Flash-Next's 160-wide n-gram table — where the only
    /// transformation the pool applies is the page-block padding that keeps a
    /// row from straddling a page.
    ///
    /// A padded block writes more than it reads, so the pass walks blocks
    /// backwards; `memmove` covers the within-block overlap.
    public static func padRowBlocksInPlace(
        buffer: UnsafeMutableRawBufferPointer,
        sourceCount: Int,
        rowSourceBytes: Int,
        rowsPerBlock: Int,
        blockStride: UInt64
    ) throws -> Int {
        guard rowSourceBytes > 0, rowsPerBlock > 0,
              rowsPerBlock <= Int.max / rowSourceBytes,
              let blockStrideBytes = Int(exactly: blockStride),
              blockStrideBytes >= rowsPerBlock * rowSourceBytes else {
            throw RepackError.configurationInvalid(
                detail: "bf16-row-blocks geometry is invalid (rowSourceBytes "
                    + "\(rowSourceBytes), rowsPerBlock \(rowsPerBlock), blockStride "
                    + "\(blockStride))")
        }
        let blockSourceBytes = rowsPerBlock * rowSourceBytes
        guard sourceCount > 0, sourceCount % blockSourceBytes == 0 else {
            throw RepackError.configurationInvalid(
                detail: "bf16-row-blocks needs \(blockSourceBytes)-byte aligned source "
                    + "chunks, got \(sourceCount)")
        }
        let blocks = sourceCount / blockSourceBytes
        let outputCount = blocks * blockStrideBytes
        guard sourceCount <= buffer.count, outputCount <= buffer.count else {
            throw RepackError.scratchExceeded(
                requested: max(sourceCount, outputCount), limit: buffer.count)
        }
        let bytes = buffer.baseAddress!.assumingMemoryBound(to: UInt8.self)
        // Backwards: block b writes into [b*stride, (b+1)*stride) while the
        // still-unread source is [0, b*blockSourceBytes), which cannot overlap
        // because stride >= blockSourceBytes. memmove covers the within-block
        // overlap.
        for step in 0..<blocks {
            let block = blocks - 1 - step
            let destination = bytes + block * blockStrideBytes
            memmove(destination, bytes + block * blockSourceBytes, blockSourceBytes)
            if blockSourceBytes < blockStrideBytes {
                memset(destination + blockSourceBytes, 0,
                       blockStrideBytes - blockSourceBytes)
            }
        }
        return outputCount
    }

    // MARK: - Internals

    private static func validatedRowDim(_ rowSourceBytes: Int) throws -> Int {
        guard rowSourceBytes > 0, rowSourceBytes % groupSourceBytes == 0 else {
            throw RepackError.configurationInvalid(
                detail: "quantize-int4-g64 row source \(rowSourceBytes) is not a "
                    + "positive multiple of \(groupSourceBytes)")
        }
        return rowSourceBytes / 2
    }

    /// One row -> `[packed nibbles | BF16 scales | BF16 biases]`.
    @inline(__always)
    private static func encodeRowRecord(_ values: UnsafePointer<Float>,
                                        rowDim: Int,
                                        into destination: UnsafeMutablePointer<UInt8>) {
        let groups = rowDim / groupSize
        let scaleBase = destination + rowDim / 2
        let biasBase = scaleBase + groups * companionBytesPerGroup
        var scale: UInt16 = 0
        var bias: UInt16 = 0
        for g in 0..<groups {
            Int4AffineEncoder.encodeGroup(values + g * groupSize,
                                          packed: destination + g * groupPackedBytes,
                                          scale: &scale,
                                          bias: &bias)
            writeLittleEndian16(scale, to: scaleBase + g * companionBytesPerGroup)
            writeLittleEndian16(bias, to: biasBase + g * companionBytesPerGroup)
        }
    }

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
