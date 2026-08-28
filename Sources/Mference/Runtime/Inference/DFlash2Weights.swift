import Darwin
import Foundation
import Metal

/// mmap view of the DFlash2 drafter checkpoint (a single BF16 safetensors
/// file). In production the mapping is a *CPU-side source only*: the
/// drafter quantizes the projections into `DFlash2Int4Slab`, copies the
/// small norm/conv tensors into an anonymous GPU buffer, and reads the
/// selector codebooks straight from the mapping — no command buffer ever
/// references file-backed memory (the driver wires file-backed buffers on
/// the CB timeline; the DSV4 copy-free experiment measured that collapse,
/// and this loader reproduced it at ~20x before the split). The lazy
/// `gpuBuffer` wrap over the mapping exists for the BF16 parity path only.
final class DFlash2Weights {

    struct Tensor {
        let offset: Int      // byte offset into the mapping
        let shape: [Int]
        let byteCount: Int
    }

    private let base: UnsafeMutableRawPointer
    private let mappedLen: Int
    private let tensors: [String: Tensor]
    private var lazyGPUBuffer: MTLBuffer?
    private let device: MTLDevice

    init(safetensorsURL: URL, device: MTLDevice) throws {
        self.device = device
        let fd = open(safetensorsURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(safetensorsURL.path))",
                                         errno: errno)
        }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw ModelError.posixFailed(call: "fstat", errno: errno)
        }
        let fileSize = Int(st.st_size)
        guard fileSize > 8 else {
            throw ModelError.indexCorrupt(detail: "dflash2 safetensors too small")
        }
        let pageSize = Int(getpagesize())
        let mappedLen = (fileSize + pageSize - 1) / pageSize * pageSize
        let mapped = mmap(nil, mappedLen, PROT_READ, MAP_PRIVATE, fd, 0)
        guard mapped != MAP_FAILED, let mappedBase = mapped else {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }
        self.base = mappedBase
        self.mappedLen = mappedLen

        var headerLen: UInt64 = 0
        memcpy(&headerLen, mappedBase, 8)
        let dataBase = 8 + Int(headerLen)
        guard dataBase > 8, dataBase < fileSize else {
            munmap(mappedBase, mappedLen)
            throw ModelError.indexCorrupt(detail: "dflash2 safetensors header length")
        }
        let headerData = Data(bytes: mappedBase.advanced(by: 8), count: Int(headerLen))
        guard let header = try? JSONSerialization.jsonObject(with: headerData)
            as? [String: Any] else {
            munmap(mappedBase, mappedLen)
            throw ModelError.indexCorrupt(detail: "dflash2 safetensors header JSON")
        }

        var parsed: [String: Tensor] = [:]
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                munmap(mappedBase, mappedLen)
                throw ModelError.indexCorrupt(detail: "dflash2 tensor entry \(name)")
            }
            guard dtype == "BF16" else {
                munmap(mappedBase, mappedLen)
                throw ModelError.indexCorrupt(
                    detail: "dflash2 tensor \(name) has dtype \(dtype); expected BF16")
            }
            let start = dataBase + offsets[0]
            let byteCount = offsets[1] - offsets[0]
            guard start % 2 == 0, start + byteCount <= fileSize else {
                munmap(mappedBase, mappedLen)
                throw ModelError.indexCorrupt(detail: "dflash2 tensor \(name) offsets")
            }
            parsed[name] = Tensor(offset: start, shape: shape, byteCount: byteCount)
        }
        self.tensors = parsed
        _ = posix_madvise(mappedBase, mappedLen, POSIX_MADV_RANDOM)
    }

    deinit {
        // The lazy GPU wrap, when built, owns the unmap through its
        // deallocator; otherwise the mapping is ours to release.
        if lazyGPUBuffer == nil {
            munmap(base, mappedLen)
        }
    }

    func tensor(_ name: String, shape expected: [Int]) throws -> Tensor {
        guard let t = tensors[name] else {
            throw ModelError.indexCorrupt(detail: "dflash2 tensor missing: \(name)")
        }
        guard t.shape == expected else {
            throw ModelError.indexCorrupt(
                detail: "dflash2 tensor \(name) shape \(t.shape) != \(expected)")
        }
        return t
    }

    /// CPU pointer to a tensor's BF16 payload (quantization source and
    /// selector codebook reads).
    func cpuPointer(of t: Tensor) -> UnsafePointer<UInt16> {
        UnsafeRawPointer(base).advanced(by: t.offset)
            .assumingMemoryBound(to: UInt16.self)
    }

    /// File-backed GPU wrap of the whole mapping — BF16 parity path only;
    /// never referenced by production command buffers.
    func gpuBuffer() throws -> MTLBuffer {
        if let lazyGPUBuffer { return lazyGPUBuffer }
        nonisolated(unsafe) let captureBase = base
        let captureLen = mappedLen
        guard let buf = device.makeBuffer(
            bytesNoCopy: base, length: mappedLen,
            options: .storageModeShared,
            deallocator: { _, _ in munmap(captureBase, captureLen) }) else {
            throw ModelError.residentBufferWrapFailed
        }
        buf.label = "dflash2.weights.bf16"
        lazyGPUBuffer = buf
        return buf
    }
}
