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
    /// A Metal buffer may be handed between threads; the reader writes the
    /// slab only between the release of its previous ticket and `filled`.
    struct Ticket: @unchecked Sendable {
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
        // The sequence and the queued fill are taken together, so fills run in
        // sequence order even when callers race.
        lock.lock()
        defer { lock.unlock() }
        nextSequence += 1
        let sequence = nextSequence
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

    /// Releases tickets whose command buffer will never run (an encode
    /// error, a cancelled chunk), so later fills can reuse their slabs: each
    /// is marked released once every earlier ticket is, which may wait on the
    /// CPU for visits already committed. They must be the latest tickets
    /// taken (nothing staged after them was committed). After any staging
    /// error, discard this object instead.
    func abandon(_ tickets: [Ticket]) throws {
        for ticket in tickets.sorted(by: { $0.sequence < $1.sequence }) {
            let earlier = ticket.sequence - 1
            if earlier > 0, released.signaledValue < earlier,
               !released.wait(untilSignaledValue: earlier, timeoutMS: releaseTimeoutMS) {
                throw StagingError.timedOut(sequence: ticket.sequence)
            }
            if released.signaledValue < ticket.sequence { released.signaledValue = ticket.sequence }
        }
    }

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
            return
        } catch {
            record(.readFailed(path: file.path, errno: EIO))
            return
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
/// binds ones). Scale buffers must be CPU-readable (shared storage): the
/// decode cache copies the scales of the experts it holds.
struct Inkling2ExpertLayerSpec {
    let file: Inkling2ExpertLayerFile
    let encoding: Inkling2MoEEncoding
    let expertStride: Int
    let gateUpScales: (buffer: MTLBuffer, offset: Int)?
    let downScales: (buffer: MTLBuffer, offset: Int)?
}

/// What a staged layer's decode visit binds: `pool` holds the visit's
/// experts in cache slots, `ids` (`topK` uint32, written by the CPU) names
/// each rank's slot, and the pool's scale tables follow the slots. Pass `ids`
/// to `encodeDecode` in place of the router's ids; the route weights are
/// unchanged.
struct Inkling2DecodeBinding {
    let pool: Inkling2ExpertPool
    let ids: MTLBuffer
    let hits: Int
    let misses: Int
}

/// All 24 layers' routed experts for one loaded model, placed by an
/// `Inkling2ExpertResidencyPlan`:
///
/// * resident layers: read once at load (direct reads, outside the file
///   cache) into one anonymous buffer each; prefill and decode bind them
///   with no readback;
/// * staged layers, prefill: the `Inkling2StagedExpertLayers` ring (whole
///   layers read through the file cache one layer ahead);
/// * staged layers, decode: after the runner reads back a visit's routed
///   ids, `prepareDecodeVisit` makes sure those experts are in the layer's
///   cache slots (LRU, misses read through the file cache) and binds them.
///   The slots live in the staging slabs, which decode leaves idle; a
///   prefill visit of a staged layer empties every cache.
///
/// Staged layer files are never mapped and bound: Metal makes a bound
/// mapping resident in full, which would page in and pin every staged layer
/// (8.9 GB on the Mini) at a decode step (K4's probe, 2026-11-23).
final class Inkling2ExpertStore: @unchecked Sendable {
    enum StoreError: Error, Equatable {
        case openFailed(path: String, errno: Int32)
        case readFailed(path: String, errno: Int32)
        case allocationFailed(String)
        case visitNotStarted(layer: Int)
        case cacheTooSmall(layer: Int, slots: Int)
        case notStaged(layer: Int)
    }

    /// One staged layer's decode cache: `slots` experts at the layer's stride
    /// from `offset` in `slab`.
    private final class DecodeCache {
        let layer: Int
        let file: Inkling2ExpertLayerFile
        let stride: Int
        let blobBytes: Int
        let slab: MTLBuffer
        let offset: Int
        let slots: Int
        let pool: Inkling2ExpertPool
        let ids: MTLBuffer
        let gateUpScales: MTLBuffer
        let downScales: MTLBuffer
        let sourceGateUp: (buffer: MTLBuffer, offset: Int)?
        let sourceDown: (buffer: MTLBuffer, offset: Int)?
        var slotOf: [Int: Int] = [:]
        var expertIn: [Int]
        var lastUse: [UInt64]
        var clock: UInt64 = 0

        init(layer: Int, spec: Inkling2ExpertLayerSpec, shape: Inkling2MoEShape, slab: MTLBuffer, offset: Int,
             slots: Int, device: MTLDevice) throws {
            self.layer = layer
            self.file = spec.file
            self.stride = spec.expertStride
            self.blobBytes = shape.expertBytes(spec.encoding)
            self.slab = slab
            self.offset = offset
            self.slots = slots
            self.sourceGateUp = spec.gateUpScales
            self.sourceDown = spec.downScales
            guard let ids = device.makeBuffer(length: max(16, shape.topK * 4), options: .storageModeShared),
                  let gateUp = device.makeBuffer(length: slots * 4, options: .storageModeShared),
                  let down = device.makeBuffer(length: slots * 4, options: .storageModeShared) else {
                throw StoreError.allocationFailed("decode cache of layer \(layer)")
            }
            self.ids = ids
            self.gateUpScales = gateUp
            self.downScales = down
            var slotShape = shape
            slotShape.experts = slots
            self.pool = try Inkling2ExpertPool(buffer: slab, offset: offset, encoding: spec.encoding, shape: slotShape,
                                               expertStride: spec.expertStride, gateUpScales: gateUp, downScales: down)
            self.expertIn = [Int](repeating: -1, count: slots)
            self.lastUse = [UInt64](repeating: 0, count: slots)
        }

        func clear() {
            slotOf.removeAll()
            for i in 0..<slots { expertIn[i] = -1; lastUse[i] = 0 }
        }

        func scale(_ source: (buffer: MTLBuffer, offset: Int)?, expert: Int) -> Float {
            guard let source else { return 1 }
            return (source.buffer.contents() + source.offset).load(fromByteOffset: expert * 4, as: Float.self)
        }
    }

    let plan: Inkling2ExpertResidencyPlan
    let shape: Inkling2MoEShape
    private let resident: [Int: Inkling2ExpertPool]
    private let templates: [Int: Inkling2ExpertPool]
    private let caches: [Int: DecodeCache]
    let staging: Inkling2StagedExpertLayers?
    private var open: [Int: [Inkling2StagedExpertLayers.Ticket]] = [:]
    /// Set by a staged prefill visit (it overwrites the slabs the caches use).
    private var cachesStale = false
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
        var templates: [Int: Inkling2ExpertPool] = [:]
        var stagedFiles: [Int: Inkling2ExpertLayerFile] = [:]
        var stagedLayers: [Int] = []
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
                stagedFiles[layer] = spec.file
                stagedLayers.append(layer)
            }
        }
        let staging = stagedFiles.isEmpty ? nil
            : try Inkling2StagedExpertLayers(device: device, files: stagedFiles, slabs: max(2, plan.stagingSlabs))

        // Staged layers: a template over slab 0 (every staged layout must fit
        // a slab, so a visit's rebinding cannot fail after its ticket is
        // taken), and a decode cache carved out of the slabs: the slabs'
        // bytes shared evenly, each layer's slots inside one slab.
        var caches: [Int: DecodeCache] = [:]
        if let staging {
            let perSlab = (stagedLayers.count + staging.slabs.count - 1) / staging.slabs.count
            let region = staging.slabBytes / perSlab / pageSize * pageSize
            for (index, layer) in stagedLayers.enumerated() {
                let spec = layers[layer]
                templates[layer] = try pool(spec, buffer: staging.slabs[0], offset: 0)
                let slots = min(shape.experts, region / spec.expertStride)
                guard slots >= shape.topK else { throw StoreError.cacheTooSmall(layer: layer, slots: slots) }
                caches[layer] = try DecodeCache(layer: layer, spec: spec, shape: shape,
                                                slab: staging.slabs[index / perSlab],
                                                offset: (index % perSlab) * region, slots: slots, device: device)
            }
        }
        self.resident = resident
        self.templates = templates
        self.caches = caches
        self.staging = staging
    }

    func isStaged(layer: Int) -> Bool { resident[layer] == nil }

    /// Decode slots per staged layer (diagnostics, tests).
    func decodeSlots(layer: Int) -> Int? { caches[layer]?.slots }

    /// The pool a decode visit of a resident layer binds (router ids used on
    /// the GPU, no readback). Staged layers go through `prepareDecodeVisit`.
    func residentPool(layer: Int) -> Inkling2ExpertPool? { resident[layer] }

    /// Starts a prefill visit of `layer` in `commandBuffer`, before its MoE
    /// encoder: a staged layer takes the next slab ticket (its read starts
    /// now, one layer ahead of the GPU) and the command buffer waits for it.
    /// Call in visit order, and `endPrefillVisit` after the visit's encoders.
    func beginPrefillVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws -> Inkling2ExpertPool {
        if let pool = resident[layer] { return pool }
        guard let staging, let template = templates[layer] else { preconditionFailure("layer \(layer)") }
        let ticket = try staging.stage(layer: layer)
        staging.encodeWait(ticket, commandBuffer: commandBuffer)
        lock.lock(); open[layer, default: []].append(ticket); cachesStale = true; lock.unlock()
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

    /// For a command buffer that will never be committed: releases every
    /// prefill visit begun and not yet ended, in order (see
    /// `Inkling2StagedExpertLayers.abandon`).
    func abandonOpenVisits() throws {
        lock.lock()
        let tickets = open.values.flatMap { $0 }
        open.removeAll()
        lock.unlock()
        try staging?.abandon(tickets)
    }

    /// A decode visit of a staged layer, on the CPU after the runner has read
    /// back the visit's routed `expertIDs` (`topK` of them): loads the misses
    /// into the layer's cache slots (least recently used first, never a slot
    /// the visit itself needs), refreshes the slot scales and writes the
    /// slot ids. Every command buffer that read this layer's slots before
    /// must have completed (the readback that produced `expertIDs` waits for
    /// all earlier work), and no prefill may be in flight.
    func prepareDecodeVisit(layer: Int, expertIDs: [Int]) throws -> Inkling2DecodeBinding {
        guard let cache = caches[layer] else { throw StoreError.notStaged(layer: layer) }
        precondition(expertIDs.count == shape.topK && expertIDs.allSatisfy { $0 >= 0 && $0 < shape.experts })
        lock.lock()
        if cachesStale {
            // A prefill wrote the slabs: wait for its last fill, then forget.
            lock.unlock()
            staging?.drain()
            lock.lock()
            for c in caches.values { c.clear() }
            cachesStale = false
        }
        lock.unlock()

        cache.clock += 1
        var slotFor = [Int](repeating: -1, count: expertIDs.count)
        var hits = 0
        for (rank, expert) in expertIDs.enumerated() {
            if let slot = cache.slotOf[expert] {
                slotFor[rank] = slot
                cache.lastUse[slot] = cache.clock
                hits += 1
            }
        }
        let needed = Set(slotFor.filter { $0 >= 0 })
        var loads: [(slot: Int, expert: Int)] = []
        for (rank, expert) in expertIDs.enumerated() where slotFor[rank] < 0 {
            if let slot = cache.slotOf[expert] {           // a repeated id in this visit
                slotFor[rank] = slot
                continue
            }
            var victim = -1
            for slot in 0..<cache.slots where !needed.contains(slot) && !loads.contains(where: { $0.slot == slot }) {
                if victim < 0 || cache.lastUse[slot] < cache.lastUse[victim] { victim = slot }
            }
            precondition(victim >= 0, "fewer slots than ranks")
            if cache.expertIn[victim] >= 0 { cache.slotOf[cache.expertIn[victim]] = nil }
            cache.expertIn[victim] = expert
            cache.slotOf[expert] = victim
            cache.lastUse[victim] = cache.clock
            slotFor[rank] = victim
            loads.append((victim, expert))
        }
        if !loads.isEmpty {
            try Self.readExperts(cache: cache, loads: loads)
            let gateUp = cache.gateUpScales.contents().bindMemory(to: Float.self, capacity: cache.slots)
            let down = cache.downScales.contents().bindMemory(to: Float.self, capacity: cache.slots)
            for load in loads {
                gateUp[load.slot] = cache.scale(cache.sourceGateUp, expert: load.expert)
                down[load.slot] = cache.scale(cache.sourceDown, expert: load.expert)
            }
        }
        let ids = cache.ids.contents().bindMemory(to: UInt32.self, capacity: shape.topK)
        for (rank, slot) in slotFor.enumerated() { ids[rank] = UInt32(slot) }
        return Inkling2DecodeBinding(pool: cache.pool, ids: cache.ids, hits: hits, misses: loads.count)
    }

    /// Reads each loaded expert's blob from the layer file into its slot, in
    /// parallel, through the file cache.
    private static func readExperts(cache: DecodeCache, loads: [(slot: Int, expert: Int)]) throws {
        let fd = Darwin.open(cache.file.path, O_RDONLY)
        guard fd >= 0 else { throw StoreError.openFailed(path: cache.file.path, errno: errno) }
        defer { close(fd) }
        nonisolated(unsafe) let base = cache.slab.contents() + cache.offset
        nonisolated(unsafe) var code: Int32 = 0
        let codeLock = NSLock()
        let stride = cache.stride, bytes = cache.blobBytes, fileOffset = cache.file.offset
        let work = loads
        DispatchQueue.concurrentPerform(iterations: work.count) { i in
            let target = base + work[i].slot * stride
            let source = off_t(fileOffset) + off_t(work[i].expert * stride)
            var done = 0
            while done < bytes {
                let n = pread(fd, target + done, bytes - done, source + off_t(done))
                if n <= 0 {
                    let failure = n == 0 ? EIO : errno
                    if failure == EINTR { continue }
                    codeLock.lock(); code = failure; codeLock.unlock()
                    return
                }
                done += n
            }
        }
        if code != 0 { throw StoreError.readFailed(path: cache.file.path, errno: code) }
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
