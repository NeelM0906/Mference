import Darwin
import Foundation
import Metal
import Testing
@testable import Mference

/// Does binding a file-mapped `MTLBuffer` make Metal page in the whole file,
/// or only what a kernel reads? It decides how Inkling 2 may address staged
/// expert layers at decode on a 32 GB host. Writes a 1 GiB temporary file
/// past the page cache (`F_NOCACHE`), maps it, binds it to a kernel that
/// reads 4 bytes, and records page faults, resident size and time; then
/// does the same read from the CPU for comparison. Model-free, but it writes
/// 1 GiB to disk, so it runs only with `MFERENCE_INKLING2_BIND_PROBE` set.
@Suite("Inkling2 mapped-buffer binding probe", .serialized)
struct Inkling2MappedBindingProbe {
    static var enabled: Bool { ProcessInfo.processInfo.environment["MFERENCE_INKLING2_BIND_PROBE"] != nil }

    struct Sample {
        var minorFaults: Int
        var majorFaults: Int
        var residentBytes: UInt64
        var seconds: Double
    }

    private static func sample() -> (Int, Int, UInt64) {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return (usage.ru_minflt, usage.ru_majflt, info.resident_size)
    }

    private static func measure(_ body: () throws -> Void) rethrows -> Sample {
        let (min0, maj0, res0) = sample()
        let start = Date()
        try body()
        let seconds = Date().timeIntervalSince(start)
        let (min1, maj1, res1) = sample()
        return Sample(minorFaults: min1 - min0, majorFaults: maj1 - maj0,
                      residentBytes: res1 > res0 ? res1 - res0 : 0, seconds: seconds)
    }

    /// A file of `bytes` written past the page cache, its pages non-zero.
    private static func writeUncachedFile(bytes: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-bind-probe-\(UUID().uuidString).bin")
        let fd = open(url.path, O_CREAT | O_WRONLY | O_TRUNC, 0o600)
        guard fd >= 0 else { throw POSIXError(.EIO) }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        let chunk = 32 << 20
        let block = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16_384)
        defer { block.deallocate() }
        memset(block, 0x5A, chunk)
        var written = 0
        while written < bytes {
            let n = write(fd, block, min(chunk, bytes - written))
            guard n > 0 else { throw POSIXError(.EIO) }
            written += n
        }
        _ = fcntl(fd, F_FULLFSYNC)
        return url
    }

    private static func map(_ url: URL, bytes: Int, device: MTLDevice) throws -> (MTLBuffer, () -> Void) {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw POSIXError(.EIO) }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        guard let base = mmap(nil, bytes, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else {
            throw POSIXError(.ENOMEM)
        }
        nonisolated(unsafe) let start = base
        guard let buffer = device.makeBuffer(bytesNoCopy: start, length: bytes, options: .storageModeShared,
                                             deallocator: nil) else {
            munmap(base, bytes)
            throw POSIXError(.ENOMEM)
        }
        return (buffer, { munmap(start, bytes) })
    }

    @Test(.enabled(if: enabled, "needs MFERENCE_INKLING2_BIND_PROBE (writes a 1 GiB temporary file)"))
    func bindingAMappedFilePagesInOnlyWhatIsRead() throws {
        let context = try MetalContext()
        let device = context.device
        let bytes = 1 << 30
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void probe_read(device const uint *src [[buffer(0)]], device uint *dst [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
            if (id == 0) { dst[0] = src[0]; }
        }
        """
        let library = try device.makeLibrary(source: source, options: nil)
        let pipeline = try device.makeComputePipelineState(function: try #require(library.makeFunction(name: "probe_read")))
        let out = try #require(device.makeBuffer(length: 16, options: .storageModeShared))

        // GPU: bind the whole mapping, read 4 bytes.
        let gpuFile = try Self.writeUncachedFile(bytes: bytes)
        defer { try? FileManager.default.removeItem(at: gpuFile) }
        let (gpuBuffer, unmapGPU) = try Self.map(gpuFile, bytes: bytes, device: device)
        let gpu = try Self.measure {
            let cb = try #require(context.queue.makeCommandBuffer())
            let enc = try #require(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(gpuBuffer, offset: 0, index: 0)
            enc.setBuffer(out, offset: 0, index: 1)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.status == .completed)
        }
        #expect(out.contents().load(as: UInt32.self) == 0x5A5A5A5A)
        unmapGPU()

        // CPU: the same 4-byte read from a fresh mapping of a fresh file.
        let cpuFile = try Self.writeUncachedFile(bytes: bytes)
        defer { try? FileManager.default.removeItem(at: cpuFile) }
        let (cpuBuffer, unmapCPU) = try Self.map(cpuFile, bytes: bytes, device: device)
        let cpu = Self.measure {
            #expect(cpuBuffer.contents().load(as: UInt32.self) == 0x5A5A5A5A)
        }
        unmapCPU()

        let pages = bytes / Int(getpagesize())
        print("inkling2 bind probe: file \(bytes >> 20) MiB = \(pages) pages")
        print("inkling2 bind probe GPU bind+read: minor faults \(gpu.minorFaults), major \(gpu.majorFaults), "
              + "resident +\(gpu.residentBytes >> 20) MiB, \(String(format: "%.3f", gpu.seconds)) s")
        print("inkling2 bind probe CPU read: minor faults \(cpu.minorFaults), major \(cpu.majorFaults), "
              + "resident +\(cpu.residentBytes >> 20) MiB, \(String(format: "%.3f", cpu.seconds)) s")
    }
}
