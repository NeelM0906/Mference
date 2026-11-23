import Darwin
import Foundation
import Metal

/// Where one stored layer's routed experts live while Inkling 2 runs.
enum Inkling2ExpertPlacement: String, Equatable, Sendable {
    /// Read once at load into an anonymous buffer (`ResidentExpertStreamer`,
    /// `.copied`) and kept for the model's lifetime.
    case resident
    /// Read through the file cache into a staging slab one layer ahead of the
    /// visit that needs it (`Inkling2StagedExpertLayers`).
    case staged
}

/// The residency ladder of the integration plan (section 4.5), per stored
/// layer. The looped layers (6-17, three visits each, 75% of expert visits)
/// are held first; the rest are staged through a ring of slabs.
struct Inkling2ExpertResidencyPlan: Equatable, Sendable {
    let placements: [Inkling2ExpertPlacement]
    /// Staging slabs, each as large as the largest staged layer (0 when
    /// nothing is staged). Two give the one-layer-ahead overlap.
    let stagingSlabs: Int
    let residentBytes: Int
    let stagingBytes: Int

    /// Hosts of at least this much memory hold every layer.
    static let allResidentMemory: UInt64 = 48 << 30
    /// Hosts of at least this much memory (the 32 and 36 GB Macs) hold the
    /// looped layers and stage the rest.
    static let loopedResidentMemory: UInt64 = 30 << 30

    /// `layerBytes[i]`: bytes of stored layer `i`'s expert file (128 x stride).
    static func make(layerBytes: [Int], loopedLayers: Range<Int>, physicalMemory: UInt64,
                     stagingSlabs slabs: Int = 2) -> Inkling2ExpertResidencyPlan {
        precondition(slabs >= 2, "one layer ahead needs two slabs")
        let placements: [Inkling2ExpertPlacement] = layerBytes.indices.map { layer in
            if physicalMemory >= allResidentMemory { return .resident }
            if physicalMemory >= loopedResidentMemory && loopedLayers.contains(layer) { return .resident }
            return .staged
        }
        let resident = zip(placements, layerBytes).filter { $0.0 == .resident }.reduce(0) { $0 + $1.1 }
        let largestStaged = zip(placements, layerBytes).filter { $0.0 == .staged }.map(\.1).max() ?? 0
        let pageSize = Int(getpagesize())
        let slabBytes = (largestStaged + pageSize - 1) / pageSize * pageSize
        let staged = largestStaged == 0 ? 0 : slabs
        return Inkling2ExpertResidencyPlan(placements: placements, stagingSlabs: staged,
                                           residentBytes: resident, stagingBytes: staged * slabBytes)
    }
}

/// One stored layer's expert file region: `bytes` starting at `offset` in
/// the file at `path` (128 blobs at the layer's stride, as K1 writes
/// `packed_experts/layer_NN.bin`).
struct Inkling2ExpertLayerFile: Sendable, Equatable {
    let path: String
    let offset: UInt64
    let bytes: Int
}

/// The layers the plan stages, read through the file cache into a ring of
/// slabs one layer ahead of the GPU. A visit takes a ticket in visit order;
/// the reader fills ticket `n`'s slab once the visit that used the slab
/// before it (ticket `n - slabs`) has released it, then signals `filled`.
/// The command buffer waits for `filled >= n` before the visit's MoE and
/// signals `released = n` after it, so the GPU never waits on a route to
/// know what to read and the CPU never waits on the GPU to queue work.
///
/// Slot ownership: a slab belongs to its ticket's visit from the `filled`
/// signal until the `released` signal; the reader never writes it in between.
final class Inkling2StagedExpertLayers: @unchecked Sendable {
    struct Ticket {
        let sequence: UInt64
        let layer: Int
        let slab: MTLBuffer
    }

    enum StagingError: Error, Equatable {
        case openFailed(path: String, errno: Int32)
        case readFailed(path: String, errno: Int32)
        case layerTooLarge(layer: Int, bytes: Int, slab: Int)
        case allocationFailed
        case timedOut(sequence: UInt64)
    }

    static let chunkBytes = 32 << 20

    let slabs: [MTLBuffer]
    let slabBytes: Int
    let filled: MTLSharedEvent
    let released: MTLSharedEvent
    private let files: [Int: Inkling2ExpertLayerFile]
    private let queue = DispatchQueue(label: "inkling2.expert-staging", qos: .userInitiated)
    private let lock = NSLock()
    private var nextSequence: UInt64 = 0
    private var failure: StagingError?
    /// Bytes read so far and the time spent reading, for diagnostics.
    private(set) var bytesRead: Int = 0
    private(set) var secondsReading: Double = 0
    /// How long the reader may wait for a slab to be released (a GPU visit
    /// that never completes would otherwise hold the reader forever).
    var releaseTimeoutMS: UInt64 = 120_000

    init(device: MTLDevice, files: [Int: Inkling2ExpertLayerFile], slabs count: Int = 2) throws {
        precondition(count >= 2 && !files.isEmpty)
        let pageSize = Int(getpagesize())
        let largest = files.values.map(\.bytes).max() ?? 0
        slabBytes = (largest + pageSize - 1) / pageSize * pageSize
        var made: [MTLBuffer] = []
        for index in 0..<count {
            guard slabBytes <= device.maxBufferLength,
                  let slab = device.makeBuffer(length: slabBytes, options: .storageModeShared) else {
                throw StagingError.allocationFailed
            }
            slab.label = "inkling2.expert-staging.\(index)"
            made.append(slab)
        }
        guard let filled = device.makeSharedEvent(), let released = device.makeSharedEvent() else {
            throw StagingError.allocationFailed
        }
        self.slabs = made
        self.files = files
        self.filled = filled
        self.released = released
    }

    /// The first failure the reader met, if any. A failed fill still
    /// signals `filled` so the GPU does not hang; the caller checks this
    /// after the command buffer completes and discards the results.
    var error: StagingError? {
        lock.lock(); defer { lock.unlock() }
        return failure
    }

    /// Queues the next staged visit of `layer` and starts its read as soon
    /// as its slab is free. Call in visit order.
    func stage(layer: Int) throws -> Ticket {
        guard let file = files[layer] else { preconditionFailure("layer \(layer) is not staged") }
        guard file.bytes <= slabBytes else {
            throw StagingError.layerTooLarge(layer: layer, bytes: file.bytes, slab: slabBytes)
        }
        lock.lock()
        nextSequence += 1
        let sequence = nextSequence
        lock.unlock()
        let slab = slabs[Int((sequence - 1) % UInt64(slabs.count))]
        let ticket = Ticket(sequence: sequence, layer: layer, slab: slab)
        queue.async { [self] in fill(ticket, file: file) }
        return ticket
    }

    /// Before the visit's MoE encoder: wait until the slab holds the layer.
    func encodeWait(_ ticket: Ticket, commandBuffer: MTLCommandBuffer) {
        commandBuffer.encodeWaitForEvent(filled, value: ticket.sequence)
    }

    /// After the visit's last encoder that reads the slab: hand it back.
    func encodeRelease(_ ticket: Ticket, commandBuffer: MTLCommandBuffer) {
        commandBuffer.encodeSignalEvent(released, value: ticket.sequence)
    }

    private func record(_ error: StagingError) {
        lock.lock()
        if failure == nil { failure = error }
        lock.unlock()
    }

    /// Waits on the CPU until every queued fill has run (tests, teardown).
    func drain() { queue.sync {} }

    private func fill(_ ticket: Ticket, file: Inkling2ExpertLayerFile) {
        // Fills run in ticket order on one serial queue, so this is the only
        // writer of `filled` and its value only grows.
        defer { filled.signaledValue = ticket.sequence }
        let previous = ticket.sequence > UInt64(slabs.count) ? ticket.sequence - UInt64(slabs.count) : 0
        if previous > 0, released.signaledValue < previous,
           !released.wait(untilSignaledValue: previous, timeoutMS: releaseTimeoutMS) {
            record(.timedOut(sequence: ticket.sequence))
            return
        }
        let start = DispatchTime.now().uptimeNanoseconds
        do {
            try Inkling2ExpertStore.read(file, into: ticket.slab.contents(), bypassCache: false)
        } catch Inkling2ExpertStore.StoreError.openFailed(let path, let code) {
            record(.openFailed(path: path, errno: code))
            return
        } catch Inkling2ExpertStore.StoreError.readFailed(let path, let code) {
            record(.readFailed(path: path, errno: code))
        } catch {
            record(.readFailed(path: file.path, errno: EIO))
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        lock.lock()
        bytesRead += file.bytes
        secondsReading += seconds
        lock.unlock()
    }
}

/// One stored layer's experts as the install keeps them, with the
/// per-expert scales of an NVFP4 layer (nil for a Q8_0 layer: the store
/// binds ones).
struct Inkling2ExpertLayerSpec {
    let file: Inkling2ExpertLayerFile
    let encoding: Inkling2MoEEncoding
    let expertStride: Int
    let gateUpScales: (buffer: MTLBuffer, offset: Int)?
    let downScales: (buffer: MTLBuffer, offset: Int)?
}

/// All 24 layers' routed experts for one loaded model, placed by an
/// `Inkling2ExpertResidencyPlan`:
///
/// * resident layers: read once at load (direct reads, outside the file
///   cache) into one anonymous buffer each;
/// * staged layers: prefill visits go through the `Inkling2StagedExpertLayers`
///   ring (read through the file cache one layer ahead); decode visits read
///   the layer file mapped whole, which those prefill reads keep warm, so a
///   decode step needs neither a route readback nor a copy.
///
/// Measured on the M1 Max (synthetic layer, warm cache): a decode step on the
/// mapped layer costs the same as on an anonymous buffer when steps share a
/// command buffer (0.302 ms each) and about 0.1 ms more as a command buffer
/// of its own.
final class Inkling2ExpertStore: @unchecked Sendable {
    enum StoreError: Error, Equatable {
        case openFailed(path: String, errno: Int32)
        case readFailed(path: String, errno: Int32)
        case mapFailed(path: String, errno: Int32)
        case allocationFailed(String)
        case visitNotStarted(layer: Int)
    }

    /// Owns one `mmap` region; captured by the buffer's deallocator.
    private final class Mapping: @unchecked Sendable {
        let base: UnsafeMutableRawPointer
        let length: Int
        init(base: UnsafeMutableRawPointer, length: Int) { self.base = base; self.length = length }
        deinit { munmap(base, length) }
    }

    let plan: Inkling2ExpertResidencyPlan
    let shape: Inkling2MoEShape
    private let resident: [Int: Inkling2ExpertPool]
    private let mapped: [Int: Inkling2ExpertPool]
    private let templates: [Int: Inkling2ExpertPool]
    let staging: Inkling2StagedExpertLayers?
    private var open: [Int: [Inkling2StagedExpertLayers.Ticket]] = [:]
    private let lock = NSLock()

    init(device: MTLDevice, shape: Inkling2MoEShape, layers: [Inkling2ExpertLayerSpec],
         plan: Inkling2ExpertResidencyPlan) throws {
        precondition(layers.count == plan.placements.count)
        self.plan = plan
        self.shape = shape
        let pageSize = Int(getpagesize())
        guard let ones = device.makeBuffer(length: max(16, shape.experts * 4), options: .storageModeShared) else {
            throw StoreError.allocationFailed("scale ones")
        }
        let onesPointer = ones.contents().bindMemory(to: Float.self, capacity: shape.experts)
        for e in 0..<shape.experts { onesPointer[e] = 1 }

        func pool(_ spec: Inkling2ExpertLayerSpec, buffer: MTLBuffer, offset: Int) throws -> Inkling2ExpertPool {
            let gateUp = spec.gateUpScales ?? (ones, 0)
            let down = spec.downScales ?? (ones, 0)
            return try Inkling2ExpertPool(buffer: buffer, offset: offset, encoding: spec.encoding, shape: shape,
                                          expertStride: spec.expertStride,
                                          gateUpScales: gateUp.buffer, gateUpScalesOffset: gateUp.offset,
                                          downScales: down.buffer, downScalesOffset: down.offset)
        }

        var resident: [Int: Inkling2ExpertPool] = [:]
        var mapped: [Int: Inkling2ExpertPool] = [:]
        var templates: [Int: Inkling2ExpertPool] = [:]
        var stagedFiles: [Int: Inkling2ExpertLayerFile] = [:]
        for (layer, spec) in layers.enumerated() {
            switch plan.placements[layer] {
            case .resident:
                let length = (spec.file.bytes + pageSize - 1) / pageSize * pageSize
                guard length <= device.maxBufferLength,
                      let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                    throw StoreError.allocationFailed("resident layer \(layer)")
                }
                buffer.label = "inkling2.experts.layer\(layer)"
                try Self.read(spec.file, into: buffer.contents(), bypassCache: true)
                let p = try pool(spec, buffer: buffer, offset: 0)
                resident[layer] = p
                templates[layer] = p
            case .staged:
                let fd = Darwin.open(spec.file.path, O_RDONLY)
                guard fd >= 0 else { throw StoreError.openFailed(path: spec.file.path, errno: errno) }
                defer { close(fd) }
                let aligned = spec.file.offset / UInt64(pageSize) * UInt64(pageSize)
                let shift = Int(spec.file.offset - aligned)
                let length = (shift + spec.file.bytes + pageSize - 1) / pageSize * pageSize
                guard let base = mmap(nil, length, PROT_READ, MAP_SHARED, fd, off_t(aligned)), base != MAP_FAILED else {
                    throw StoreError.mapFailed(path: spec.file.path, errno: errno)
                }
                let mapping = Mapping(base: base, length: length)
                nonisolated(unsafe) let start = base
                guard let buffer = device.makeBuffer(bytesNoCopy: start, length: length, options: .storageModeShared,
                                                     deallocator: { _, _ in _ = mapping }) else {
                    throw StoreError.allocationFailed("mapped layer \(layer)")
                }
                buffer.label = "inkling2.experts.layer\(layer).mapped"
                let p = try pool(spec, buffer: buffer, offset: shift)
                mapped[layer] = p
                templates[layer] = p
                stagedFiles[layer] = spec.file
            }
        }
        self.resident = resident
        self.mapped = mapped
        self.templates = templates
        self.staging = stagedFiles.isEmpty ? nil
            : try Inkling2StagedExpertLayers(device: device, files: stagedFiles, slabs: max(2, plan.stagingSlabs))
    }

    /// The pool a decode visit of `layer` reads: resident, or the mapped file.
    func decodePool(layer: Int) -> Inkling2ExpertPool {
        resident[layer] ?? mapped[layer]!
    }

    /// Starts a prefill visit of `layer` in `commandBuffer`, before its MoE
    /// encoder: a staged layer takes the next slab ticket (its read starts
    /// now, one layer ahead of the GPU) and the command buffer waits for it.
    /// Call in visit order, and `endPrefillVisit` after the visit's encoders.
    func beginPrefillVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws -> Inkling2ExpertPool {
        if let pool = resident[layer] { return pool }
        guard let staging, let template = templates[layer] else { preconditionFailure("layer \(layer)") }
        let ticket = try staging.stage(layer: layer)
        staging.encodeWait(ticket, commandBuffer: commandBuffer)
        lock.lock(); open[layer, default: []].append(ticket); lock.unlock()
        return try template.rebound(to: ticket.slab)
    }

    /// Ends a prefill visit: a staged layer's slab is released when the
    /// command buffer reaches this point.
    func endPrefillVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws {
        guard resident[layer] == nil, let staging else { return }
        lock.lock()
        guard var tickets = open[layer], !tickets.isEmpty else {
            lock.unlock()
            throw StoreError.visitNotStarted(layer: layer)
        }
        let ticket = tickets.removeFirst()
        open[layer] = tickets
        lock.unlock()
        staging.encodeRelease(ticket, commandBuffer: commandBuffer)
    }

    /// Reads `file` into `destination` (32 MiB chunks, four readers);
    /// `bypassCache` sets `F_NOCACHE` so a resident copy does not also fill
    /// the file cache.
    static func read(_ file: Inkling2ExpertLayerFile, into destination: UnsafeMutableRawPointer,
                     bypassCache: Bool) throws {
        let fd = Darwin.open(file.path, O_RDONLY)
        guard fd >= 0 else { throw StoreError.openFailed(path: file.path, errno: errno) }
        defer { close(fd) }
        if bypassCache { _ = fcntl(fd, F_NOCACHE, 1) }
        let chunk = Inkling2StagedExpertLayers.chunkBytes
        let chunks = (file.bytes + chunk - 1) / chunk
        nonisolated(unsafe) let target = destination
        nonisolated(unsafe) var code: Int32 = 0
        let codeLock = NSLock()
        DispatchQueue.concurrentPerform(iterations: max(1, min(4, chunks))) { worker in
            var c = worker
            while c < chunks {
                let begin = c * chunk, length = min(chunk, file.bytes - begin)
                var done = 0
                while done < length {
                    let n = pread(fd, target.advanced(by: begin + done), length - done,
                                  off_t(file.offset) + off_t(begin + done))
                    if n <= 0 {
                        let failure = n == 0 ? EIO : errno
                        if failure == EINTR { continue }
                        codeLock.lock(); code = failure; codeLock.unlock()
                        return
                    }
                    done += n
                }
                c += 4
            }
        }
        if code != 0 { throw StoreError.readFailed(path: file.path, errno: code) }
    }
}
