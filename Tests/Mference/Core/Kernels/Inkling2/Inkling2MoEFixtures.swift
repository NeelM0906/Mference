import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// Synthetic Inkling 2 expert pools for the kernel tests and the
/// microbenchmark. Small shapes quantize random weights with the reference
/// encoders; production shapes fill blocks with random codes and bounded
/// scales directly (the encoders are too slow there in a debug build).
struct Inkling2SyntheticPool {
    let encoding: Inkling2MoEEncoding
    let shape: Inkling2MoEShape
    let blobs: [[UInt8]]
    let scales: [Float]   // gate-up scales [experts], then down scales [experts]
    let pool: Inkling2ExpertPool

    var referenceEncoding: Inkling2ExpertEncoding { encoding == .nvfp4 ? .nvfp4 : .q8_0 }

    func expert(_ e: Int) -> Inkling2ExpertWeights {
        let gateUpBytes = shape.gateUpBytes(encoding)
        return Inkling2ExpertWeights(
            encoding: referenceEncoding, d: shape.d, f: shape.f,
            gateUp: Array(blobs[e][0..<gateUpBytes]), down: Array(blobs[e][gateUpBytes...]),
            gateUpScale: scales[e], downScale: scales[shape.experts + e])
    }

    /// `quantized`: encode weights with the reference encoders (small
    /// shapes), each 16-value block at its own amplitude so block and
    /// sub-block scales vary, at magnitudes that keep SITU out of saturation;
    /// otherwise random blocks (`fullScaleRange`: NVFP4 scale bytes over all
    /// 256 values). `keepBlobs: false` skips the CPU copy (benchmarks).
    static func make(device: MTLDevice, encoding: Inkling2MoEEncoding, shape: Inkling2MoEShape,
                     seed: UInt64, quantized: Bool, fullScaleRange: Bool = false,
                     expertScales: ClosedRange<Float>? = nil, keepBlobs: Bool = true,
                     expertStride stride: Int? = nil) throws -> Inkling2SyntheticPool {
        var rng = SeedTree(seed).key("inkling2-pool-\(encoding)")
        let blobBytes = shape.expertBytes(encoding)
        let expertStride = stride ?? (blobBytes + 15) / 16 * 16
        let total = expertStride * shape.experts
        guard let buffer = device.makeBuffer(length: total, options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("synthetic pool")
        }
        let base = buffer.contents().bindMemory(to: UInt8.self, capacity: total)
        let reference: Inkling2ExpertEncoding = encoding == .nvfp4 ? .nvfp4 : .q8_0
        var blobs: [[UInt8]] = []
        for e in 0..<shape.experts {
            var blob: [UInt8]
            if quantized {
                // NVFP4 layers carry a per-expert scale (about 0.013 here);
                // Q8_0 layers do not, so their weights are that much smaller.
                let base: Float = encoding == .nvfp4 ? 6 : 0.08
                func row(_ k: Int) -> [Float] {
                    var out: [Float] = []
                    for _ in 0..<(k / 16) {
                        let amplitude = base * Float(pow(2.0, Double(rng.uniform(-2, 2))))
                        for _ in 0..<16 { out.append(rng.uniform(-1, 1) * amplitude) }
                    }
                    return out
                }
                blob = []
                blob.reserveCapacity(blobBytes)
                for _ in 0..<(2 * shape.f) { blob += reference.quantize(row(shape.d)) }
                for _ in 0..<shape.d { blob += reference.quantize(row(shape.f)) }
            } else {
                blob = randomBlocks(encoding: encoding, bytes: blobBytes, rng: &rng, fullScaleRange: fullScaleRange)
            }
            blob.withUnsafeBufferPointer { src in
                (base + e * expertStride).update(from: src.baseAddress!, count: blobBytes)
            }
            if keepBlobs { blobs.append(blob) }
        }
        let range = expertScales ?? 0.006...0.02
        let scales: [Float] = (0..<(2 * shape.experts)).map { _ in
            encoding == .nvfp4 ? rng.uniform(range.lowerBound, range.upperBound) : 1
        }
        guard let scaleBuffer = scales.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        }) else { throw PrefillGroupedRoutedMoEError.allocationFailed("synthetic scales") }
        let pool = try Inkling2ExpertPool(buffer: buffer, encoding: encoding, shape: shape,
                                          expertStride: expertStride,
                                          gateUpScales: scaleBuffer, gateUpScalesOffset: 0,
                                          downScales: scaleBuffer, downScalesOffset: shape.experts * 4)
        return Inkling2SyntheticPool(encoding: encoding, shape: shape, blobs: blobs, scales: scales, pool: pool)
    }

    /// Random blocks: NVFP4 scales in [0.25, 2) (or any byte with
    /// `fullScaleRange`: zero, 0x7F, sign bit, subnormal and exponent 15
    /// bytes included) and any codes; Q8_0 scales in [1/256, 1/64) and any
    /// int8. Both whole-block layouts, so a blob of `bytes` is a whole
    /// number of blocks.
    static func randomBlocks(encoding: Inkling2MoEEncoding, bytes: Int, rng: inout SplitMix64,
                             fullScaleRange: Bool = false) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: bytes)
        let block = encoding.blockBytes
        precondition(bytes % block == 0)
        out.withUnsafeMutableBufferPointer { p in
            var i = 0
            while i < bytes {
                if encoding == .nvfp4 {
                    for s in 0..<4 {
                        p[i + s] = fullScaleRange ? UInt8(rng.next() & 0xFF) : UInt8(0x28 + Int(rng.next() % 24))
                    }
                    var j = 4
                    while j < block {
                        var r = rng.next()
                        for _ in 0..<8 where j < block { p[i + j] = UInt8(r & 0xFF); r >>= 8; j += 1 }
                    }
                } else {
                    let d = Float16(rng.uniform(1.0 / 256, 1.0 / 64)).bitPattern
                    p[i] = UInt8(d & 0xFF); p[i + 1] = UInt8(d >> 8)
                    var j = 2
                    while j < block {
                        var r = rng.next()
                        for _ in 0..<8 where j < block { p[i + j] = UInt8(r & 0xFF); r >>= 8; j += 1 }
                    }
                }
                i += block
            }
        }
        return out
    }
}

enum Inkling2MoETestSupport {
    /// `topK` distinct experts per token, uniform, and positive weights.
    static func randomRoutes(tokens: Int, experts: Int, topK: Int, seed: UInt64) -> (ids: [UInt32], weights: [Float]) {
        var rng = SeedTree(seed).key("inkling2-routes")
        var ids: [UInt32] = [], weights: [Float] = []
        for _ in 0..<tokens {
            var chosen: [UInt32] = []
            while chosen.count < topK {
                let e = UInt32(rng.next() % UInt64(experts))
                if !chosen.contains(e) { chosen.append(e) }
            }
            ids += chosen
            weights += (0..<topK).map { _ in rng.uniform(0.2, 2) }
        }
        return (ids, weights)
    }

    static func write<T>(_ values: [T], to buffer: MTLBuffer, offset: Int = 0) {
        values.withUnsafeBytes { raw in
            (buffer.contents() + offset).copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
    }

    static func read<T>(_ buffer: MTLBuffer, as: T.Type, count: Int, offset: Int = 0) -> [T] {
        let p = (buffer.contents() + offset).bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: p, count: count))
    }

    static func halfBuffer(_ device: MTLDevice, rows: Int, stride: Int, d: Int, seed: UInt64,
                           range: Float = 2) -> (buffer: MTLBuffer, rows: [[Float]]) {
        var rng = SeedTree(seed).key("inkling2-hidden")
        let buffer = device.makeBuffer(length: max(16, rows * stride * 2), options: .storageModeShared)!
        let p = buffer.contents().bindMemory(to: Float16.self, capacity: rows * stride)
        var values: [[Float]] = []
        for r in 0..<rows {
            var row: [Float] = []
            for i in 0..<stride {
                let h = Float16(rng.uniform(-range, range))
                p[r * stride + i] = h
                if i < d { row.append(Float(h)) }
            }
            values.append(row)
        }
        return (buffer, values)
    }

    static func run(_ context: MetalContext, _ body: (MTLComputeCommandEncoder) -> Void) -> MTLCommandBuffer {
        let cb = context.queue.makeCommandBuffer()!
        let encoder = cb.makeComputeCommandEncoder()!
        body(encoder)
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return cb
    }
}
