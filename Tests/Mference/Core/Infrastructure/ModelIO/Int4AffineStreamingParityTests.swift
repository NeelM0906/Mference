import Foundation
import Testing
import MferenceRepackCore
@testable import Mference

/// Phase A gate W2.1a, streaming half: the repacker's **quantize-in-flight**
/// path must reproduce the runtime module's reference quantizer
/// (`Quantization.quantizeInt4Affine`) bit for bit — packed nibbles, BF16 scale
/// bits, BF16 bias bits — no matter how the byte stream is chopped into write
/// tiles.
///
/// `Int4AffineEncoderParityTests` locks the whole-tensor encoder to the same
/// reference. This suite additionally proves that chunking the source through
/// bounded scratch (which is how `HTTPRangeSourceByteProvider` actually drives
/// the transform) changes nothing, including on awkward shapes where tile
/// boundaries land in the middle of a row.
///
/// Gate W2.1b (model-level KLD against a known-good conversion) is NOT covered
/// here and remains open; see the TODO at `StreamingInt4Quantizer`.
@Suite struct Int4AffineStreamingParityTests {

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
            let encoded = Quantization.quantizeInt4Affine(slice)
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

    // MARK: - Component transform

    /// Shapes chosen so tiles land mid-row: single row, row counts that share
    /// no factor with the tile size, and one wide row.
    static let awkwardShapes: [(rows: Int, cols: Int)] = [
        (1, 64),        // one row, exactly one group
        (1, 4_096),     // one very wide row
        (3, 192),       // three groups per row, prime row count
        (7, 320),       // five groups per row
        (33, 128),      // row count not a multiple of anything convenient
        (129, 64),      // one group per row, many rows
        (5, 1_024),
    ]

    @Test("Streaming components match the runtime reference on awkward shapes",
          arguments: awkwardShapes)
    func componentsMatchReference(shape: (rows: Int, cols: Int)) throws {
        let tensor = Self.makeTensor(rows: shape.rows, cols: shape.cols,
                                     seed: UInt64(shape.rows &* 1_000 &+ shape.cols))
        let expected = Self.reference(tensor)
        // 128 = one group per tile (worst case); 384 and 640 straddle row
        // boundaries for every shape above; the last two exceed the tensor.
        for tile in [128, 384, 640, 8_192, 1 << 20] {
            let scratch = max(tile, StreamingInt4Quantizer.groupSourceBytes)
            let packed = try Self.stream(tensor.bytes,
                                         tileSourceBytes: tile,
                                         scratchBytes: scratch) { buffer, count in
                try StreamingInt4Quantizer.quantizeComponentInPlace(
                    .weights, buffer: buffer, sourceCount: count)
            }
            let scales = try Self.stream(tensor.bytes,
                                         tileSourceBytes: tile,
                                         scratchBytes: scratch) { buffer, count in
                try StreamingInt4Quantizer.quantizeComponentInPlace(
                    .scales, buffer: buffer, sourceCount: count)
            }
            let biases = try Self.stream(tensor.bytes,
                                         tileSourceBytes: tile,
                                         scratchBytes: scratch) { buffer, count in
                try StreamingInt4Quantizer.quantizeComponentInPlace(
                    .biases, buffer: buffer, sourceCount: count)
            }
            #expect(packed == expected.packed,
                    "packed \(shape.rows)x\(shape.cols) tile \(tile)")
            #expect(scales == Self.bytes(of: expected.scales),
                    "scales \(shape.rows)x\(shape.cols) tile \(tile)")
            #expect(biases == Self.bytes(of: expected.biases),
                    "biases \(shape.rows)x\(shape.cols) tile \(tile)")
        }
    }

    @Test("Dequantizing the streamed result reproduces the reference dequant")
    func dequantRoundTripMatchesReference() throws {
        let tensor = Self.makeTensor(rows: 9, cols: 256, seed: 0xD3_9A_11)
        let expected = Self.reference(tensor)
        let packed = try Self.stream(tensor.bytes, tileSourceBytes: 384,
                                     scratchBytes: 384) { buffer, count in
            try StreamingInt4Quantizer.quantizeComponentInPlace(
                .weights, buffer: buffer, sourceCount: count)
        }
        let scales = try Self.stream(tensor.bytes, tileSourceBytes: 384,
                                     scratchBytes: 384) { buffer, count in
            try StreamingInt4Quantizer.quantizeComponentInPlace(
                .scales, buffer: buffer, sourceCount: count)
        }
        let biases = try Self.stream(tensor.bytes, tileSourceBytes: 384,
                                     scratchBytes: 384) { buffer, count in
            try StreamingInt4Quantizer.quantizeComponentInPlace(
                .biases, buffer: buffer, sourceCount: count)
        }
        // Read the companions back out of the *streamed* bytes so nothing in
        // this comparison is borrowed from the reference.
        let streamedScales = Self.uint16s(scales)
        let streamedBiases = Self.uint16s(biases)
        let groupsPerRow = tensor.cols / 64
        for row in 0..<tensor.rows {
            let rowPacked = Array(packed[row * tensor.cols / 2 ..< (row + 1) * tensor.cols / 2])
            let streamedRow = Quantization.dequantizeInt4Affine(
                Quantization.Int4AffineRow(
                    packed: rowPacked,
                    scales: Array(streamedScales[row * groupsPerRow ..< (row + 1) * groupsPerRow]),
                    biases: Array(streamedBiases[row * groupsPerRow ..< (row + 1) * groupsPerRow])),
                n: tensor.cols)
            let referenceRow = Quantization.dequantizeInt4Affine(
                Quantization.quantizeInt4Affine(
                    Array(tensor.values[row * tensor.cols ..< (row + 1) * tensor.cols])),
                n: tensor.cols)
            #expect(streamedRow == referenceRow, "row \(row)")
        }
        #expect(streamedScales == expected.scales)
        #expect(streamedBiases == expected.biases)
    }

    // MARK: - Row-record transform (PLE row pool)

    @Test("Streamed row records equal the reference row's packed|scales|biases",
          arguments: [64, 128, 320, 1_024])
    func rowRecordsMatchReference(rowDim: Int) throws {
        let rows = 11
        let tensor = Self.makeTensor(rows: rows, cols: rowDim,
                                     seed: UInt64(0xB10C_0000 + rowDim))
        let rowSourceBytes = rowDim * 2
        let recordBytes = StreamingInt4Quantizer.rowRecordBytes(rowDim: rowDim)
        for tile in [rowSourceBytes, 3 * rowSourceBytes, rows * rowSourceBytes] {
            let streamed = try Self.stream(tensor.bytes,
                                           tileSourceBytes: tile,
                                           scratchBytes: tile) { buffer, count in
                try StreamingInt4Quantizer.quantizeRowsInPlace(
                    buffer: buffer, sourceCount: count, rowSourceBytes: rowSourceBytes)
            }
            #expect(streamed.count == rows * recordBytes)
            for row in 0..<rows {
                let encoded = Quantization.quantizeInt4Affine(
                    Array(tensor.values[row * rowDim ..< (row + 1) * rowDim]))
                var expected = encoded.packed
                expected += Self.bytes(of: encoded.scales)
                expected += Self.bytes(of: encoded.biases)
                #expect(Array(streamed[row * recordBytes ..< (row + 1) * recordBytes])
                    == expected, "rowDim \(rowDim) tile \(tile) row \(row)")
            }
        }
    }

    @Test("Row blocks pad to the stride and never straddle a page")
    func rowBlocksPadToStride() throws {
        // rowDim 64 -> 36-byte record; a 16 KB page holds 455 of them with
        // 404 bytes of slack, which the transform must zero.
        let rowDim = 64
        let rowsPerBlock = 5
        let blockStride: UInt64 = 512
        let blocks = 3
        let rows = rowsPerBlock * blocks
        let tensor = Self.makeTensor(rows: rows, cols: rowDim, seed: 0x5B10_C0DE)
        let recordBytes = StreamingInt4Quantizer.rowRecordBytes(rowDim: rowDim)
        let sourceBytes = rows * rowDim * 2
        let streamed = try Self.stream(
            tensor.bytes,
            tileSourceBytes: sourceBytes,
            scratchBytes: max(sourceBytes, blocks * Int(blockStride))
        ) { buffer, count in
            try StreamingInt4Quantizer.quantizeRowBlocksInPlace(
                buffer: buffer,
                sourceCount: count,
                rowSourceBytes: rowDim * 2,
                rowsPerBlock: rowsPerBlock,
                blockStride: blockStride)
        }
        #expect(streamed.count == blocks * Int(blockStride))
        for block in 0..<blocks {
            let base = block * Int(blockStride)
            for slot in 0..<rowsPerBlock {
                let row = block * rowsPerBlock + slot
                let encoded = Quantization.quantizeInt4Affine(
                    Array(tensor.values[row * rowDim ..< (row + 1) * rowDim]))
                var expected = encoded.packed
                expected += Self.bytes(of: encoded.scales)
                expected += Self.bytes(of: encoded.biases)
                let start = base + slot * recordBytes
                #expect(Array(streamed[start ..< start + recordBytes]) == expected,
                        "block \(block) slot \(slot)")
            }
            let used = base + rowsPerBlock * recordBytes
            #expect(streamed[used ..< base + Int(blockStride)].allSatisfy { $0 == 0 },
                    "block \(block) padding is not zeroed")
        }
    }

    /// A block whose padding dominates writes more bytes than it reads, so the
    /// transform has to walk blocks backwards; prove it still lands correctly.
    @Test("Expanding row blocks (stride larger than the source block) are exact")
    func expandingRowBlocksAreExact() throws {
        let rowDim = 64
        let rowsPerBlock = 1
        let blockStride: UInt64 = 16_384
        let blocks = 4
        let tensor = Self.makeTensor(rows: blocks, cols: rowDim, seed: 0xEE_5A_11)
        let recordBytes = StreamingInt4Quantizer.rowRecordBytes(rowDim: rowDim)
        let sourceBytes = blocks * rowDim * 2
        let streamed = try Self.stream(
            tensor.bytes,
            tileSourceBytes: sourceBytes,
            scratchBytes: blocks * Int(blockStride)
        ) { buffer, count in
            try StreamingInt4Quantizer.quantizeRowBlocksInPlace(
                buffer: buffer,
                sourceCount: count,
                rowSourceBytes: rowDim * 2,
                rowsPerBlock: rowsPerBlock,
                blockStride: blockStride)
        }
        #expect(streamed.count == blocks * Int(blockStride))
        for block in 0..<blocks {
            let encoded = Quantization.quantizeInt4Affine(
                Array(tensor.values[block * rowDim ..< (block + 1) * rowDim]))
            var expected = encoded.packed
            expected += Self.bytes(of: encoded.scales)
            expected += Self.bytes(of: encoded.biases)
            let base = block * Int(blockStride)
            #expect(Array(streamed[base ..< base + recordBytes]) == expected,
                    "block \(block)")
        }
    }

    // MARK: - BF16 row blocks (the pool path for widths group-64 cannot divide)

    /// Flash-Next's n-gram rows are 160 wide, so they stay BF16 and the only
    /// thing the pool does is re-block them. The rows must survive byte for
    /// byte and the page slack must be zeroed.
    @Test("BF16 row blocks copy rows verbatim and zero the page slack",
          arguments: [(rowDim: 160, rowsPerBlock: 51, blockStride: UInt64(16_384)),
                      (rowDim: 96, rowsPerBlock: 85, blockStride: UInt64(16_384)),
                      // No slack at all: stride exactly fills the block.
                      (rowDim: 128, rowsPerBlock: 4, blockStride: UInt64(1_024))])
    func bf16RowBlocksArePreservedAndPadded(
        shape: (rowDim: Int, rowsPerBlock: Int, blockStride: UInt64)
    ) throws {
        let blocks = 3
        let rows = shape.rowsPerBlock * blocks
        let tensor = Self.makeTensor(rows: rows, cols: shape.rowDim,
                                     seed: UInt64(0xB16B_0000 + shape.rowDim))
        let rowBytes = shape.rowDim * 2
        let sourceBytes = rows * rowBytes
        let stride = Int(shape.blockStride)
        let streamed = try Self.stream(
            tensor.bytes,
            tileSourceBytes: sourceBytes,
            scratchBytes: max(sourceBytes, blocks * stride)
        ) { buffer, count in
            try StreamingInt4Quantizer.padRowBlocksInPlace(
                buffer: buffer,
                sourceCount: count,
                rowSourceBytes: rowBytes,
                rowsPerBlock: shape.rowsPerBlock,
                blockStride: shape.blockStride)
        }
        #expect(streamed.count == blocks * stride)
        for block in 0..<blocks {
            let base = block * stride
            for slot in 0..<shape.rowsPerBlock {
                let row = block * shape.rowsPerBlock + slot
                let start = base + slot * rowBytes
                #expect(Array(streamed[start ..< start + rowBytes])
                    == Array(tensor.bytes[row * rowBytes ..< (row + 1) * rowBytes]),
                    "block \(block) slot \(slot)")
                // One row, one page: never split across a page boundary.
                #expect(start / 16_384 == (start + rowBytes - 1) / 16_384,
                        "block \(block) slot \(slot) straddles a page")
            }
            let used = base + shape.rowsPerBlock * rowBytes
            #expect(streamed[used ..< base + stride].allSatisfy { $0 == 0 },
                    "block \(block) slack is not zeroed")
        }
    }

    /// A block that is mostly padding writes more than it reads, so the pass
    /// has to walk blocks backwards; prove no row is clobbered.
    @Test("Expanding BF16 row blocks do not clobber unread source")
    func expandingBF16RowBlocksAreExact() throws {
        let rowDim = 160
        let rowsPerBlock = 1
        let blockStride: UInt64 = 16_384
        let blocks = 4
        let tensor = Self.makeTensor(rows: blocks, cols: rowDim, seed: 0xE0_B1_6B)
        let rowBytes = rowDim * 2
        let streamed = try Self.stream(
            tensor.bytes,
            tileSourceBytes: blocks * rowBytes,
            scratchBytes: blocks * Int(blockStride)
        ) { buffer, count in
            try StreamingInt4Quantizer.padRowBlocksInPlace(
                buffer: buffer,
                sourceCount: count,
                rowSourceBytes: rowBytes,
                rowsPerBlock: rowsPerBlock,
                blockStride: blockStride)
        }
        for block in 0..<blocks {
            let base = block * Int(blockStride)
            #expect(Array(streamed[base ..< base + rowBytes])
                == Array(tensor.bytes[block * rowBytes ..< (block + 1) * rowBytes]),
                "block \(block)")
        }
    }

    // MARK: - Guard rails

    @Test("Misaligned source chunks are rejected rather than silently truncated")
    func misalignedChunksThrow() {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: 4_096, alignment: 16_384)
        defer { scratch.deallocate() }
        #expect(throws: RepackError.self) {
            _ = try StreamingInt4Quantizer.quantizeComponentInPlace(
                .weights, buffer: scratch, sourceCount: 130)
        }
        #expect(throws: RepackError.self) {
            _ = try StreamingInt4Quantizer.quantizeRowsInPlace(
                buffer: scratch, sourceCount: 128, rowSourceBytes: 96)
        }
        #expect(throws: RepackError.self) {
            _ = try StreamingInt4Quantizer.quantizeRowBlocksInPlace(
                buffer: scratch, sourceCount: 256, rowSourceBytes: 128,
                rowsPerBlock: 2, blockStride: 8)
        }
        // A stride that cannot hold its own block would silently drop rows.
        #expect(throws: RepackError.self) {
            _ = try StreamingInt4Quantizer.padRowBlocksInPlace(
                buffer: scratch, sourceCount: 640, rowSourceBytes: 320,
                rowsPerBlock: 2, blockStride: 512)
        }
        // Partial trailing block: the source must be a whole number of blocks.
        #expect(throws: RepackError.self) {
            _ = try StreamingInt4Quantizer.padRowBlocksInPlace(
                buffer: scratch, sourceCount: 960, rowSourceBytes: 320,
                rowsPerBlock: 2, blockStride: 1_024)
        }
    }
}
