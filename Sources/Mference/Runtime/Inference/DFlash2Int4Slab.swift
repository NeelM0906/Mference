import Darwin
import Foundation
import Metal

/// Runtime INT4 group-64 quantization of the DFlash2 drafter's projection
/// tensors into one shared slab, in exactly the layout
/// `dequant_int4_gemv_simd_multix` consumes: per row, `N/2` nibble-packed
/// bytes (element 2i low nibble, 2i+1 high), plus per-row-per-group BF16
/// scale/bias with `value = scale * q + bias`, `q in 0...15`, `bias = min`,
/// `scale = (max - min) / 15`.
///
/// Why quantize at load: the BF16 drafter is 3.85 GB — it thrashes a 24 GB
/// host against the 15 GB target and reads 4x the bytes per draft round.
/// INT4 puts the drafter at ~1.05 GB on the production GEMV kernels, the
/// same trade the MTP attach step makes. The result is cached next to the
/// checkpoint (`int4-slab.cache`, keyed on source size+mtime), so the
/// ~seconds of quantization happen once per checkpoint, not per process.
final class DFlash2Int4Slab {

    struct View {
        let weightsOffset: Int
        let scalesOffset: Int
        let biasesOffset: Int
    }

    let buffer: MTLBuffer
    private var views: [String: View] = [:]

    func view(_ name: String) -> View {
        guard let v = views[name] else {
            preconditionFailure("dflash2 int4 slab missing tensor \(name)")
        }
        return v
    }

    /// `tensors` are (name, bf16 payload pointer, [M, N]) with N % 64 == 0.
    init(tensors: [(name: String, base: UnsafePointer<UInt16>, shape: [Int])],
         cacheURL: URL?, cacheKey: String,
         device: MTLDevice) throws {
        var total = 0
        var layout: [(name: String, view: View, m: Int, n: Int)] = []
        for t in tensors {
            precondition(t.shape.count == 2 && t.shape[1] % 64 == 0,
                         "dflash2 quant tensor \(t.name) shape \(t.shape)")
            let m = t.shape[0], n = t.shape[1]
            let weightsOffset = total
            let scalesOffset = weightsOffset + m * n / 2
            let biasesOffset = scalesOffset + m * (n / 64) * 2
            total = biasesOffset + m * (n / 64) * 2
            total = (total + 63) / 64 * 64
            layout.append((t.name,
                           View(weightsOffset: weightsOffset,
                                scalesOffset: scalesOffset,
                                biasesOffset: biasesOffset),
                           m, n))
        }

        let pageSize = Int(getpagesize())
        let slabLen = (total + pageSize - 1) / pageSize * pageSize
        var raw: UnsafeMutableRawPointer?
        let allocResult = posix_memalign(&raw, pageSize, slabLen)
        guard allocResult == 0, let slab = raw else {
            throw ModelError.posixFailed(call: "posix_memalign(dflash2 slab)",
                                         errno: allocResult)
        }

        var loadedFromCache = false
        if let cacheURL,
           let cached = try? FileHandle(forReadingFrom: cacheURL) {
            defer { try? cached.close() }
            if let keyData = try? cached.read(upToCount: 64),
               String(data: keyData, encoding: .utf8) == cacheKey.padding(
                   toLength: 64, withPad: " ", startingAt: 0) {
                var filled = 0
                while filled < total {
                    guard let chunk = try? cached.read(upToCount: min(total - filled, 128 << 20)),
                          !chunk.isEmpty else { break }
                    chunk.withUnsafeBytes {
                        slab.advanced(by: filled)
                            .copyMemory(from: $0.baseAddress!, byteCount: $0.count)
                    }
                    filled += chunk.count
                }
                loadedFromCache = filled == total
            }
        }

        if !loadedFromCache {
            for entry in layout {
                let source = tensors.first { $0.name == entry.name }!.base
                Self.quantize(source: source,
                              m: entry.m, n: entry.n,
                              weights: slab.advanced(by: entry.view.weightsOffset)
                                  .assumingMemoryBound(to: UInt8.self),
                              scales: slab.advanced(by: entry.view.scalesOffset)
                                  .assumingMemoryBound(to: UInt16.self),
                              biases: slab.advanced(by: entry.view.biasesOffset)
                                  .assumingMemoryBound(to: UInt16.self))
            }
            if let cacheURL {
                let key = cacheKey.padding(toLength: 64, withPad: " ", startingAt: 0)
                var blob = Data(key.utf8)
                blob.append(Data(bytesNoCopy: slab, count: total, deallocator: .none))
                try? blob.write(to: cacheURL, options: .atomic)
            }
        }

        nonisolated(unsafe) let captureSlab = slab
        guard let buf = device.makeBuffer(
            bytesNoCopy: slab, length: slabLen,
            options: .storageModeShared,
            deallocator: { _, _ in free(captureSlab) }) else {
            free(slab)
            throw ModelError.residentBufferWrapFailed
        }
        buf.label = "dflash2.int4slab"
        self.buffer = buf
        for entry in layout { views[entry.name] = entry.view }
    }

    private static func bf16ToFloat(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }
    private static func floatToBF16(_ v: Float) -> UInt16 {
        // Round-to-nearest-even, matching the repack quantizer.
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    private static func quantize(source: UnsafePointer<UInt16>,
                                 m: Int, n: Int,
                                 weights: UnsafeMutablePointer<UInt8>,
                                 scales: UnsafeMutablePointer<UInt16>,
                                 biases: UnsafeMutablePointer<UInt16>) {
        let groups = n / 64
        DispatchQueue.concurrentPerform(iterations: m) { row in
            let src = source + row * n
            let wRow = weights + row * (n / 2)
            let sRow = scales + row * groups
            let bRow = biases + row * groups
            for g in 0..<groups {
                var lo = Float.greatestFiniteMagnitude
                var hi = -Float.greatestFiniteMagnitude
                let base = g * 64
                for i in 0..<64 {
                    let v = bf16ToFloat(src[base + i])
                    lo = min(lo, v)
                    hi = max(hi, v)
                }
                // Round-trip the BF16 storage of scale/bias so quantization
                // targets exactly what the kernel will dequantize with.
                let sBits = floatToBF16((hi - lo) / 15)
                let bBits = floatToBF16(lo)
                sRow[g] = sBits
                bRow[g] = bBits
                let s = bf16ToFloat(sBits)
                let b = bf16ToFloat(bBits)
                let inv = s > 0 ? 1 / s : 0
                for pair in 0..<32 {
                    let v0 = bf16ToFloat(src[base + pair * 2])
                    let v1 = bf16ToFloat(src[base + pair * 2 + 1])
                    let q0 = UInt8(max(0, min(15, ((v0 - b) * inv).rounded())))
                    let q1 = UInt8(max(0, min(15, ((v1 - b) * inv).rounded())))
                    wRow[g * 32 + pair] = q0 | (q1 << 4)
                }
            }
        }
    }
}
