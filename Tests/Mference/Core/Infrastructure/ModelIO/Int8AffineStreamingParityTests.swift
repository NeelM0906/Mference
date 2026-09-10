import Foundation
import Testing
import MferenceRepackCore
@testable import Mference

/// The W2.1a-style bootstrap gate for the **INT8** half of the repacker's
/// quantize-in-flight path: `Int8AffineEncoder` and `StreamingInt8Quantizer`
/// must reproduce the runtime module's reference quantizer
/// (`Quantization.quantizeInt8Affine`) bit for bit — packed bytes, BF16 scale
/// bits, BF16 bias bits — no matter how the byte stream is chopped into write
/// tiles.
///
/// This suite is a prerequisite, not a formality. New production quantization
/// math cannot be trusted on assertion, and the INT8 path is what writes the
/// 80 router and shared-expert-gate tensors of a mixed-width install; if it
/// disagreed with the runtime's decode semantics by even one byte, the
/// `docs/QUANTIZER_QUALITY.md` model-level measurement would be scoring a bug
/// rather than a quantizer. It mirrors `Int4AffineStreamingParityTests`
/// exactly, minus the row-pool transforms, which have no INT8 counterpart
/// (Flash-Next's PLE table is INT4 by policy).
@Suite struct Int8AffineStreamingParityTests {

    // MARK: - Fixtures

    /// Deterministic BF16-exact test values: rounding to BF16 up front means
    /// the reference (which reads Floats) and the streaming path (which reads
    /// BF16 bytes) see identical numbers.
    private struct BF16Tensor {
        let rows: Int
        let cols: Int
        let values: [Float]     // row-major, every value exactly BF16
        let bytes: [UInt8]      // little-endian BF16 payload
    }

    private static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb = (bits >> 16) & 1
        return UInt16(truncatingIfNeeded: (bits &+ (0x7FFF &+ lsb)) >> 16)
    }

    private static func makeTensor(rows: Int, cols: Int, seed: UInt64) -> BF16Tensor {
        var state = seed | 1
        func nextFloat() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let unit = Float(state >> 40) / Float(1 << 24)
            return (unit - 0.5) * 6.0
        }
        var values = [Float]()
        var bytes = [UInt8]()
        values.reserveCapacity(rows * cols)
        bytes.reserveCapacity(rows * cols * 2)
        for index in 0..<(rows * cols) {
            // Sprinkle in constant, zero and extreme groups so the branchy
            // parts of the encoder are exercised by the streaming path too.
            let raw: Float
            switch (index / 64) % 5 {
            case 1: raw = 0
            case 2: raw = 2.5
            case 3: raw = index % 64 == 0 ? -8_192 : 8_192
            default: raw = nextFloat()
            }
            let bits = bf16Bits(raw)
            values.append(Float(bitPattern: UInt32(bits) << 16))
            bytes.append(UInt8(truncatingIfNeeded: bits))
            bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
        }
        return BF16Tensor(rows: rows, cols: cols, values: values, bytes: bytes)
    }

    /// The reference result, assembled row by row through the runtime module's
    /// own quantizer.
    private static func reference(_ tensor: BF16Tensor)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        var packed = [UInt8]()
        var scales = [UInt16]()
        var biases = [UInt16]()
        for row in 0..<tensor.rows {
            let slice = Array(tensor.values[row * tensor.cols ..< (row + 1) * tensor.cols])
            let encoded = Quantization.quantizeInt8Affine(slice)
            packed += encoded.packed
            scales += encoded.scales
            biases += encoded.biases
        }
        return (packed, scales, biases)
    }

    /// Drive one transform the way the installer does: repeated bounded-scratch
    /// calls, `tileSourceBytes` of source per call.
    private static func stream(
        _ source: [UInt8],
        tileSourceBytes: Int,
        scratchBytes: Int,
        _ body: (UnsafeMutableRawBufferPointer, Int) throws -> Int
    ) rethrows -> [UInt8] {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: scratchBytes, alignment: 16_384)
        defer { scratch.deallocate() }
        var output = [UInt8]()
        var offset = 0
        while offset < source.count {
            let count = min(tileSourceBytes, source.count - offset)
            source.withUnsafeBytes { raw in
                _ = memcpy(scratch.baseAddress!, raw.baseAddress! + offset, count)
            }
            let produced = try body(scratch, count)
            output.append(contentsOf: UnsafeRawBufferPointer(
                start: scratch.baseAddress!, count: produced))
            offset += count
        }
        return output
    }

    private static func bytes(of values: [UInt16]) -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(values.count * 2)
        for value in values {
            out.append(UInt8(truncatingIfNeeded: value))
            out.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        return out
    }

    private static func uint16s(_ bytes: [UInt8]) -> [UInt16] {
        stride(from: 0, to: bytes.count, by: 2).map {
            UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8
        }
    }

    // MARK: - Whole-tensor encoder

    /// The non-streaming encoder must match the reference too, so a failure in
    /// the streaming suite below can be attributed to tiling rather than to the
    /// nucleus.
    @Test("The whole-tensor INT8 encoder matches the runtime reference",
          arguments: [(rows: 1, cols: 64), (rows: 4, cols: 2_048),
                      (rows: 256, cols: 2_048), (rows: 1, cols: 2_048)])
    func encoderMatchesReference(shape: (rows: Int, cols: Int)) {
        let tensor = Self.makeTensor(rows: shape.rows, cols: shape.cols,
                                     seed: UInt64(shape.rows &* 7919 &+ shape.cols))
        let expected = Self.reference(tensor)
        let encoded = tensor.values.withUnsafeBufferPointer {
            Int8AffineEncoder.encodeTensor($0, rowLength: shape.cols)
        }
        #expect(encoded.packed == expected.packed)
        #expect(encoded.scales == expected.scales)
        #expect(encoded.biases == expected.biases)
    }

    // MARK: - Component transform

    /// Shapes chosen so tiles land mid-row: single row, row counts that share
    /// no factor with the tile size, and one wide row. The last two are the
    /// real router shapes — `mlp.gate` is [numExperts, hidden] = [256, 2048]
    /// and `mlp.shared_expert_gate` is the single row [1, 2048].
    static let awkwardShapes: [(rows: Int, cols: Int)] = [
        (1, 64),        // one row, exactly one group
        (1, 4_096),     // one very wide row
        (3, 192),       // three groups per row, prime row count
        (7, 320),       // five groups per row
        (33, 128),      // row count not a multiple of anything convenient
        (129, 64),      // one group per row, many rows
        (5, 1_024),
        (1, 2_048),     // shared_expert_gate
        (256, 2_048),   // mlp.gate (the router)
    ]

    @Test("Streaming INT8 components match the runtime reference on awkward shapes",
          arguments: awkwardShapes)
    func componentsMatchReference(shape: (rows: Int, cols: Int)) throws {
        let tensor = Self.makeTensor(rows: shape.rows, cols: shape.cols,
                                     seed: UInt64(shape.rows &* 1_000 &+ shape.cols))
        let expected = Self.reference(tensor)
        // 128 = one group per tile (worst case); 384 and 640 straddle row
        // boundaries for every shape above; the last two exceed the tensor.
        for tile in [128, 384, 640, 8_192, 1 << 20] {
            let scratch = max(tile, StreamingInt8Quantizer.groupSourceBytes)
            let packed = try Self.stream(tensor.bytes,
                                         tileSourceBytes: tile,
                                         scratchBytes: scratch) { buffer, count in
                try StreamingInt8Quantizer.quantizeComponentInPlace(
                    .weights, buffer: buffer, sourceCount: count)
            }
            let scales = try Self.stream(tensor.bytes,
                                         tileSourceBytes: tile,
                                         scratchBytes: scratch) { buffer, count in
                try StreamingInt8Quantizer.quantizeComponentInPlace(
                    .scales, buffer: buffer, sourceCount: count)
            }
            let biases = try Self.stream(tensor.bytes,
                                         tileSourceBytes: tile,
                                         scratchBytes: scratch) { buffer, count in
                try StreamingInt8Quantizer.quantizeComponentInPlace(
                    .biases, buffer: buffer, sourceCount: count)
            }
            #expect(packed == expected.packed,
                    "packed \(shape.rows)x\(shape.cols) tile \(tile)")
            #expect(scales == Self.bytes(of: expected.scales),
                    "scales \(shape.rows)x\(shape.cols) tile \(tile)")
            #expect(biases == Self.bytes(of: expected.biases),
                    "biases \(shape.rows)x\(shape.cols) tile \(tile)")
            #expect(packed.count == shape.rows * shape.cols,
                    "INT8 emits one byte per weight")
        }
    }

    @Test("Dequantizing the streamed INT8 result reproduces the reference dequant")
    func dequantRoundTripMatchesReference() throws {
        let tensor = Self.makeTensor(rows: 9, cols: 256, seed: 0xD3_9A_11)
        let expected = Self.reference(tensor)
        func streamed(_ component: StreamingInt4Quantizer.Component) throws -> [UInt8] {
            try Self.stream(tensor.bytes, tileSourceBytes: 384,
                            scratchBytes: 384) { buffer, count in
                try StreamingInt8Quantizer.quantizeComponentInPlace(
                    component, buffer: buffer, sourceCount: count)
            }
        }
        let packed = try streamed(.weights)
        // Read the companions back out of the *streamed* bytes so nothing in
        // this comparison is borrowed from the reference.
        let streamedScales = Self.uint16s(try streamed(.scales))
        let streamedBiases = Self.uint16s(try streamed(.biases))
        let groupsPerRow = tensor.cols / 64
        for row in 0..<tensor.rows {
            let rowPacked = Array(packed[row * tensor.cols ..< (row + 1) * tensor.cols])
            let streamedRow = Quantization.dequantizeInt8Affine(
                Quantization.Int8AffineRow(
                    packed: rowPacked,
                    scales: Array(streamedScales[row * groupsPerRow ..< (row + 1) * groupsPerRow]),
                    biases: Array(streamedBiases[row * groupsPerRow ..< (row + 1) * groupsPerRow])),
                n: tensor.cols)
            let referenceRow = Quantization.dequantizeInt8Affine(
                Quantization.quantizeInt8Affine(
                    Array(tensor.values[row * tensor.cols ..< (row + 1) * tensor.cols])),
                n: tensor.cols)
            #expect(streamedRow == referenceRow, "row \(row)")
        }
        #expect(streamedScales == expected.scales)
        #expect(streamedBiases == expected.biases)
    }

    // MARK: - Guard rails

    @Test("Misaligned INT8 source chunks are rejected rather than silently truncated")
    func misalignedChunksThrow() {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: 4_096, alignment: 16_384)
        defer { scratch.deallocate() }
        #expect(throws: RepackError.self) {
            _ = try StreamingInt8Quantizer.quantizeComponentInPlace(
                .weights, buffer: scratch, sourceCount: 130)
        }
        #expect(throws: RepackError.self) {
            _ = try StreamingInt8Quantizer.quantizeComponentInPlace(
                .weights, buffer: scratch, sourceCount: 0)
        }
        // A tile larger than the scratch would write past the buffer.
        #expect(throws: RepackError.self) {
            _ = try StreamingInt8Quantizer.quantizeComponentInPlace(
                .weights, buffer: scratch, sourceCount: 8_192)
        }
    }
}
