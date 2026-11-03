import Darwin
import Foundation

/// How a routed expert is stored on disk when the file omits bytes the
/// kernels still read.
///
/// `PackedExpertsLayout.expertStride` and the tensor offsets keep describing
/// the expanded expert the kernels address. The stored expert is a sequence
/// of `segments` copied to their memory offsets; every `impliedBiases` entry
/// is then rebuilt from its scales, so the slot holds exactly the bytes an
/// explicit install would have produced.
///
/// The only format is `impliedNeg8ScaleBiases`: Gemma 4 QAT's checkpoint is
/// Q4_0 re-expressed as MLX affine INT4, where every group's bias is exactly
/// `-8 * scale` in BF16. The installer and converter refuse a checkpoint in
/// which any group breaks that identity.
public struct ExpertStorage: Sendable, Equatable {
    public static let impliedNeg8ScaleBiasesFormat = "impliedNeg8ScaleBiases"

    public struct Segment: Sendable, Equatable {
        public let storedOffset: UInt64
        public let memoryOffset: UInt64
        public let size: UInt64

        public init(storedOffset: UInt64, memoryOffset: UInt64, size: UInt64) {
            self.storedOffset = storedOffset
            self.memoryOffset = memoryOffset
            self.size = size
        }
    }

    /// Memory offsets of a BF16 bias array and the BF16 scales it is rebuilt
    /// from; both hold `size` bytes.
    public struct ImpliedBias: Sendable, Equatable {
        public let biasesOffset: UInt64
        public let scalesOffset: UInt64
        public let size: UInt64

        public init(biasesOffset: UInt64, scalesOffset: UInt64, size: UInt64) {
            self.biasesOffset = biasesOffset
            self.scalesOffset = scalesOffset
            self.size = size
        }
    }

    public let storedExpertStride: UInt64
    public let segments: [Segment]
    public let impliedBiases: [ImpliedBias]

    public init(storedExpertStride: UInt64, segments: [Segment], impliedBiases: [ImpliedBias]) {
        self.storedExpertStride = storedExpertStride
        self.segments = segments.sorted { $0.storedOffset < $1.storedOffset }
        self.impliedBiases = impliedBiases
    }

    /// The BF16 bit pattern of `-8 * scale`, computed in FP32. Multiplying by
    /// a power of two keeps the mantissa, so the product is exact and its low
    /// 16 bits are zero for every finite scale: truncation is not rounding.
    @inline(__always)
    public static func neg8ScaleBits(_ scale: UInt16) -> UInt16 {
        let product = Float(bitPattern: UInt32(scale) << 16) * -8
        return UInt16(truncatingIfNeeded: product.bitPattern >> 16)
    }

    /// Writes `neg8ScaleBits` of `count` BF16 scales to `biases`.
    public static func fillNeg8ScaleBiases(scales: UnsafeRawPointer,
                                           biases: UnsafeMutableRawPointer,
                                           count: Int) {
        let factor = SIMD8<Float>(repeating: -8)
        var index = 0
        while index + 8 <= count {
            let scale = scales.loadUnaligned(fromByteOffset: index * 2, as: SIMD8<UInt16>.self)
            let widened = SIMD8<UInt32>(truncatingIfNeeded: scale) &<< 16
            let product = unsafeBitCast(widened, to: SIMD8<Float>.self) * factor
            let bits = unsafeBitCast(product, to: SIMD8<UInt32>.self) &>> 16
            biases.storeBytes(of: SIMD8<UInt16>(truncatingIfNeeded: bits),
                              toByteOffset: index * 2, as: SIMD8<UInt16>.self)
            index += 8
        }
        while index < count {
            let scale = scales.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
            biases.storeBytes(of: neg8ScaleBits(scale), toByteOffset: index * 2, as: UInt16.self)
            index += 1
        }
    }

    /// Rebuilds every implied bias array inside one expanded expert.
    public func fillImpliedBiases(expert: UnsafeMutableRawPointer) {
        for bias in impliedBiases {
            Self.fillNeg8ScaleBiases(scales: UnsafeRawPointer(expert.advanced(by: Int(bias.scalesOffset))),
                                     biases: expert.advanced(by: Int(bias.biasesOffset)),
                                     count: Int(bias.size) / 2)
        }
    }

    /// Reads `destinations.count` experts stored back to back from
    /// `fileOffset` into expanded experts, then rebuilds their biases. File
    /// bytes outside every segment (alignment padding) are read and dropped.
    func readExperts(fd: Int32,
                     fileOffset: UInt64,
                     into destinations: [UnsafeMutableRawPointer]) throws {
        guard !destinations.isEmpty else { return }
        // Bytes the file holds between or after segments; they are never
        // used, so one scratch region per call absorbs all of them.
        var gaps: [(offset: UInt64, size: UInt64)] = []
        var cursor: UInt64 = 0
        for segment in segments {
            if segment.storedOffset > cursor {
                gaps.append((cursor, segment.storedOffset - cursor))
            }
            cursor = segment.storedOffset + segment.size
        }
        if storedExpertStride > cursor {
            gaps.append((cursor, storedExpertStride - cursor))
        }
        let scratchSize = Int(gaps.map(\.size).max() ?? 0)
        let scratch = UnsafeMutableRawPointer.allocate(byteCount: max(1, scratchSize), alignment: 16)
        defer { scratch.deallocate() }

        var vectors: [iovec] = []
        vectors.reserveCapacity(destinations.count * (segments.count + gaps.count))
        for destination in destinations {
            var pieces: [(storedOffset: UInt64, base: UnsafeMutableRawPointer, size: UInt64)] =
                segments.map { ($0.storedOffset, destination.advanced(by: Int($0.memoryOffset)), $0.size) }
            pieces += gaps.map { ($0.offset, scratch, $0.size) }
            pieces.sort { $0.storedOffset < $1.storedOffset }
            for piece in pieces where piece.size > 0 {
                vectors.append(iovec(iov_base: piece.base, iov_len: Int(piece.size)))
            }
        }
        try Self.preadvAll(fd: fd, vectors: &vectors, fileOffset: fileOffset,
                           total: UInt64(destinations.count) * storedExpertStride)
        for destination in destinations {
            fillImpliedBiases(expert: destination)
        }
    }

    /// `preadv` until every vector is full, resuming inside a vector after a
    /// short read and splitting lists longer than `IOV_MAX`.
    static func preadvAll(fd: Int32, vectors: inout [iovec], fileOffset: UInt64, total: UInt64) throws {
        var first = 0
        var filled: UInt64 = 0
        while first < vectors.count {
            let count = min(vectors.count - first, Int(IOV_MAX))
            let readCount = vectors.withUnsafeBufferPointer { buffer in
                preadv(fd, buffer.baseAddress!.advanced(by: first), Int32(count),
                       off_t(fileOffset) + off_t(filled))
            }
            if readCount < 0 {
                if errno == EINTR { continue }
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: total, actual: filled)
            }
            filled += UInt64(readCount)
            var remaining = readCount
            while remaining > 0 {
                let length = vectors[first].iov_len
                if remaining >= length {
                    remaining -= length
                    first += 1
                } else {
                    vectors[first].iov_base = vectors[first].iov_base!.advanced(by: remaining)
                    vectors[first].iov_len = length - remaining
                    remaining = 0
                }
            }
        }
    }
}
