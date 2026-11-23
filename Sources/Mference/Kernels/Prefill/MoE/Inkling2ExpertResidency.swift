import Darwin
import Foundation
import Metal

/// Where one stored layer's routed experts live while Inkling 2 runs.
enum Inkling2ExpertPlacement: String, Equatable, Sendable {
    /// Read once at load into an anonymous buffer (`ResidentExpertStreamer`,
    /// `.copied`) and kept for the model's lifetime.
    case resident
    /// Read into a staging slab one layer ahead of the visit that needs it
    /// (`Inkling2StagedExpertLayers`), uncached by default below 48 GB
    /// (`Inkling2StagedReadMode`).
    case staged
}

/// How the staged layers' bytes are read from their files: the prefill slab
/// fills and the decode-cache misses. (Resident layers are always read
/// uncached, once, at load.)
enum Inkling2StagedReadMode: String, Equatable, Sendable {
    /// `F_NOCACHE`: the reads do not fill the file cache, so a staged layer's
    /// bytes are held only in the slab. Without it a run would also keep
    /// every staged layer in clean file-cache pages (8.9 GB on the Mini),
    /// which a 32 GB host cannot spare (coordinator ruling, 2026-11-23).
    case uncached
    /// Through the file cache: repeat reads of cached pages are faster, at
    /// the cost of the cache's memory. An option, to be measured as a speed
    /// setting on hosts with room for it.
    case cached

    /// Uncached on hosts below `Inkling2ExpertResidencyPlan.allResidentMemory`
    /// (48 GB, the hosts that stage anything), cached at or above it.
    static func forHost(physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Self {
        physicalMemory < Inkling2ExpertResidencyPlan.allResidentMemory ? .uncached : .cached
    }

    /// The `MFERENCE_INKLING2_DECODE_MISS_READS` experiment toggle (`cached`
    /// or `uncached`), if set.
    static var decodeMissOverride: Self? {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_DECODE_MISS_READS"].flatMap(Self.init(rawValue:))
    }
}

/// How the staged layers' decode slots are laid out in the staging slabs.
enum Inkling2DecodeSlots: Equatable, Sendable {
    /// The slabs as prefill needs them (each the size of the largest staged
    /// layer), their bytes shared evenly between the staged layers: at
    /// 32 GB, 40 NVFP4 or 21 Q8_0 slots a layer.
    case evenBytes
    /// The same number of slots for every staged layer (capped at the
    /// expert count), the slabs enlarged where that needs more bytes. A Q8_0
    /// miss costs more than an NVFP4 one, so equal counts beat even bytes at
    /// equal memory (K2's replay of real routes, 2026-11-23).
    case perLayer(Int)

    /// `.perLayer(n)` from the `MFERENCE_INKLING2_DECODE_SLOTS` experiment
    /// toggle, otherwise `.evenBytes`.
    static var fromEnvironment: Self {
        ProcessInfo.processInfo.environment["MFERENCE_INKLING2_DECODE_SLOTS"].flatMap(Int.init).map { .perLayer($0) }
            ?? .evenBytes
    }
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

/// The layers the plan stages, read (by `readMode`) into a ring of slabs one
/// layer ahead of the GPU. A visit takes a ticket in visit order;
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
    /// Whether the fills bypass the file cache. Each fill reads it when it
    /// starts; change it between runs (after `drain()`) to compare the modes
    /// on one loaded model.
    var readMode: Inkling2StagedReadMode {
        get { lock.withLock { mode } }
        set { lock.withLock { mode = newValue } }
    }
    private var mode: Inkling2StagedReadMode
    private let files: [Int: Inkling2ExpertLayerFile]
    private let queue = DispatchQueue(label: "inkling2.expert-staging", qos: .userInitiated)
    private let lock = NSLock()
    private var nextSequence: UInt64 = 0
    private var failure: StagingError?

    /// One staged visit's timings (uptime nanoseconds; 0 = not reached), for
    /// diagnostics: queued by `stage`, read started (slab free and fence
    /// passed) and finished, and when the GPU reached the visit's wait
    /// (a shared-event notification, so a few microseconds late). The GPU
    /// waited on the read for `finished - gpuArrived` when that is positive.
    struct TimelineEntry: Equatable, Sendable {
        let sequence: UInt64
        let layer: Int
        let bytes: Int
        var queued: UInt64 = 0
        var started: UInt64 = 0
        var finished: UInt64 = 0
        var gpuArrived: UInt64 = 0
        /// How long the GPU waited for this visit's read, in seconds.
        var gpuStallSeconds: Double {
            guard finished > 0, gpuArrived > 0, finished > gpuArrived else { return 0 }
            return Double(finished - gpuArrived) / 1e9
        }
    }
    /// Records a `TimelineEntry` per staged visit while set (off by default;
    /// set it before staging). Read the entries after `drain()` and after the
    /// command buffers complete.
    var recordsTimeline: Bool {
        get { lock.withLock { recording } }
        set { lock.withLock { recording = newValue } }
    }
    private var recording = false
    private var entries: [UInt64: TimelineEntry] = [:]
    private var arrived: MTLSharedEvent?
    private var listener: MTLSharedEventListener?
    private var listenerQueue: DispatchQueue?
    /// Bytes read so far and the time spent reading, for diagnostics.
    private(set) var bytesRead: Int = 0
    private(set) var secondsReading: Double = 0
    /// How long the reader may wait for a slab to be released (a GPU visit
    /// that never completes would otherwise hold the reader forever).
    var releaseTimeoutMS: UInt64 = 120_000

    /// `minimumSlabBytes` enlarges the slabs past the largest layer (room for
    /// decode slots).
    init(device: MTLDevice, files: [Int: Inkling2ExpertLayerFile], slabs count: Int = 2,
         readMode: Inkling2StagedReadMode = .forHost(), minimumSlabBytes: Int = 0) throws {
        precondition(count >= 2 && !files.isEmpty)
        let pageSize = Int(getpagesize())
        let largest = max(files.values.map(\.bytes).max() ?? 0, minimumSlabBytes)
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
        self.mode = readMode
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

    /// The last sequence handed out (0 before any).
    var lastSequence: UInt64 {
        lock.lock(); defer { lock.unlock() }
        return nextSequence
    }

    /// Queues the next staged visit of `layer` and starts its read as soon
    /// as its slab is free and, with `fence`, once `fence.event` reaches
    /// `fence.value` (other GPU readers of the slabs, such as decode caches,
    /// must be done). Call in visit order.
    func stage(layer: Int, after fence: (event: MTLSharedEvent, value: UInt64)? = nil) throws -> Ticket {
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
        if recording {
            entries[sequence] = TimelineEntry(sequence: sequence, layer: layer, bytes: file.bytes,
                                              queued: DispatchTime.now().uptimeNanoseconds)
        }
        queue.async { [self] in fill(ticket, file: file, after: fence) }
        return ticket
    }

    /// Before the visit's MoE encoder: wait until the slab holds the layer.
    func encodeWait(_ ticket: Ticket, commandBuffer: MTLCommandBuffer) {
        if let arrived = timelineArrivalEvent(commandBuffer.device), let listener {
            // Diagnostics only: a signal just before the wait tells when the
            // GPU reached it.
            commandBuffer.encodeSignalEvent(arrived, value: ticket.sequence)
            let sequence = ticket.sequence
            arrived.notify(listener, atValue: sequence) { [weak self] _, _ in
                let now = DispatchTime.now().uptimeNanoseconds
                self?.lock.withLock { self?.entries[sequence]?.gpuArrived = now }
            }
        }
        commandBuffer.encodeWaitForEvent(filled, value: ticket.sequence)
    }

    /// The arrival event while the timeline records (made on first use).
    private func timelineArrivalEvent(_ device: MTLDevice) -> MTLSharedEvent? {
        lock.withLock {
            guard recording else { return nil }
            if arrived == nil {
                let queue = DispatchQueue(label: "inkling2.expert-staging.timeline")
                arrived = device.makeSharedEvent()
                listener = MTLSharedEventListener(dispatchQueue: queue)
                listenerQueue = queue
            }
            return arrived
        }
    }

    /// The recorded visits in sequence order. Arrival notifications reach
    /// the listener's queue a little after the GPU's signal, so this waits
    /// (up to a second) for those of visits the GPU has already reached.
    func timeline() -> [TimelineEntry] {
        let (event, queue) = lock.withLock { (arrived, listenerQueue) }
        if let event, let queue {
            let deadline = DispatchTime.now().uptimeNanoseconds + 1_000_000_000
            while DispatchTime.now().uptimeNanoseconds < deadline {
                queue.sync {}
                let reached = event.signaledValue
                let pending = lock.withLock { entries.values.contains { $0.sequence <= reached && $0.gpuArrived == 0 } }
                if !pending { break }
                usleep(100)
            }
        }
        return lock.withLock { entries.values.sorted { $0.sequence < $1.sequence } }
    }

    /// Forgets the recorded visits (recording continues while set).
    func resetTimeline() {
        lock.withLock { entries.removeAll() }
    }

    /// After the visit's last encoder that reads the slab: hand it back.
    func encodeRelease(_ ticket: Ticket, commandBuffer: MTLCommandBuffer) {
        commandBuffer.encodeSignalEvent(released, value: ticket.sequence)
    }

    /// Marks every ticket taken so far released, for when a failed command
    /// buffer skipped its release signals. Only once no command buffer that
    /// could still read a slab is in flight.
    func releaseIssued() {
        lock.withLock { if released.signaledValue < nextSequence { released.signaledValue = nextSequence } }
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

    private func fill(_ ticket: Ticket, file: Inkling2ExpertLayerFile,
                      after fence: (event: MTLSharedEvent, value: UInt64)?) {
        // Fills run in ticket order on one serial queue, so this is the only
        // writer of `filled` and its value only grows.
        defer { filled.signaledValue = ticket.sequence }
        if let fence, fence.event.signaledValue < fence.value,
           !fence.event.wait(untilSignaledValue: fence.value, timeoutMS: releaseTimeoutMS) {
            record(.timedOut(sequence: ticket.sequence))
            return
        }
        let previous = ticket.sequence > UInt64(slabs.count) ? ticket.sequence - UInt64(slabs.count) : 0
        if previous > 0, released.signaledValue < previous,
           !released.wait(untilSignaledValue: previous, timeoutMS: releaseTimeoutMS) {
            record(.timedOut(sequence: ticket.sequence))
            return
        }
        let start = DispatchTime.now().uptimeNanoseconds
        let uncached = lock.withLock {
            entries[ticket.sequence]?.started = start
            return mode == .uncached
        }
        do {
            try Inkling2ExpertStore.read(file, into: ticket.slab.contents(), bypassCache: uncached)
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
        let end = DispatchTime.now().uptimeNanoseconds
        lock.lock()
        bytesRead += file.bytes
        secondsReading += Double(end - start) / 1e9
        entries[ticket.sequence]?.finished = end
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
    let layer: Int
    let pool: Inkling2ExpertPool
    let ids: MTLBuffer
    /// Ranks served from slots already holding their expert (a rank repeating
    /// an expert loaded for an earlier rank of the same visit counts as a hit).
    let hits: Int
    /// Experts read from the layer file for this visit.
    let misses: Int
}

/// All 24 layers' routed experts for one loaded model, placed by an
/// `Inkling2ExpertResidencyPlan`:
///
/// * resident layers: read once at load (direct reads, outside the file
///   cache) into one anonymous buffer each; prefill and decode bind them
///   with no readback;
/// * staged layers, prefill: the `Inkling2StagedExpertLayers` ring (whole
///   layers read one layer ahead);
/// * staged layers, decode: after the runner reads back a visit's routed
///   ids, `prepareDecodeVisit` makes sure those experts are in the layer's
///   cache slots (LRU, misses read from the layer file) and binds them.
///   The slots live in the staging slabs, which decode leaves idle; a
///   prefill visit of a staged layer empties every cache.
///
/// Both staged paths read by `readMode`: uncached (`F_NOCACHE`) by default
/// on hosts below 48 GB, so the staged layers are not held a second time in
/// the file cache.
///
/// Slot ownership: a decode visit's slots belong to its command buffer from
/// `prepareDecodeVisit` until `endDecodeVisit`'s signal; the store waits for
/// that signal before refilling a slot of the layer, and a prefill fill of
/// the slabs waits for every decode visit ended so far. A decode visit waits
/// for every prefill visit's GPU reads of the slabs before it refills them.
/// Use the store from one thread (the runner's).
///
/// Staged layer files are mapped read-only for CPU readers (references,
/// checks) but never bound on the GPU: Metal makes a bound mapping resident
/// in full, which would page in and pin every staged layer (8.9 GB on the
/// Mini) at a decode step (K4's probe, 2026-11-23).
final class Inkling2ExpertStore: @unchecked Sendable {
    enum StoreError: Error, Equatable {
        case openFailed(path: String, errno: Int32)
        case readFailed(path: String, errno: Int32)
        case allocationFailed(String)
        case visitNotStarted(layer: Int)
        case cacheTooSmall(layer: Int, slots: Int)
        case notStaged(layer: Int)
        case timedOut(String)
        case mapFailed(path: String, errno: Int32)
    }

    /// Owns one read-only `mmap` region; captured by the buffer's deallocator.
    private final class Mapping: @unchecked Sendable {
        let base: UnsafeMutableRawPointer
        let length: Int
        init(base: UnsafeMutableRawPointer, length: Int) { self.base = base; self.length = length }
        deinit { munmap(base, length) }
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
        /// `decodeDone` value of the last ended decode visit of this layer.
        var lastRead: UInt64 = 0

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
                                               expertStride: spec.expertStride, gateUpScales: gateUp, downScales: down,
                                               isSlotPool: true)
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
    /// How staged layers are read (slab fills and decode-cache misses).
    /// Change it only between runs (no prefill in flight), for example to
    /// compare the two modes on one loaded model.
    var readMode: Inkling2StagedReadMode {
        didSet { staging?.readMode = readMode }
    }
    /// The read mode of decode-cache misses when it differs from
    /// `readMode` (nil: the same). Cached decode-miss reads keep single
    /// experts (4.5 to 8.5 MB) in the file cache, not whole layers.
    var decodeMissReadMode: Inkling2StagedReadMode?
    /// How the decode slots were laid out.
    let decodeSlotLayout: Inkling2DecodeSlots
    /// Every expert byte the store's buffers can make resident: the resident
    /// layers and the staging slabs (decode slots live in the slabs).
    var boundBytes: Int { plan.residentBytes + (staging.map { $0.slabs.count * $0.slabBytes } ?? 0) }
    private let resident: [Int: Inkling2ExpertPool]
    private let mapped: [Int: Inkling2ExpertPool]
    private let templates: [Int: Inkling2ExpertPool]
    private let caches: [Int: DecodeCache]
    /// The staged ring, for reading its state (`error`, `bytesRead`). Stage
    /// only through `beginPrefillVisit`: a direct `stage` would bypass the
    /// decode-cache bookkeeping and the slot-ownership fence.
    let staging: Inkling2StagedExpertLayers?
    private var open: [Int: [Inkling2StagedExpertLayers.Ticket]] = [:]
    /// Set by a staged prefill visit (it overwrites the slabs the caches use).
    private var cachesStale = false
    /// Signalled by each ended decode visit's command buffer with its count.
    private let decodeDone: MTLSharedEvent
    private var decodeIssued: UInt64 = 0
    private let lock = NSLock()
    /// How long the CPU waits for the GPU before reporting `timedOut`.
    var waitTimeoutMS: UInt64 = 120_000

    /// `decodeSlots` lays out the decode slots (default: the experiment
    /// toggle, else even bytes in the slabs as prefill sizes them);
    /// `decodeMissReadMode` reads decode misses in their own mode (default:
    /// the toggle, else `readMode`).
    init(device: MTLDevice, shape: Inkling2MoEShape, layers: [Inkling2ExpertLayerSpec],
         plan: Inkling2ExpertResidencyPlan, readMode: Inkling2StagedReadMode = .forHost(),
         decodeSlots: Inkling2DecodeSlots = .fromEnvironment,
         decodeMissReadMode: Inkling2StagedReadMode? = Inkling2StagedReadMode.decodeMissOverride) throws {
        precondition(layers.count == plan.placements.count)
        self.plan = plan
        self.shape = shape
        self.readMode = readMode
        self.decodeMissReadMode = decodeMissReadMode
        self.decodeSlotLayout = decodeSlots
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
                // A read-only mapping for CPU readers (address space only
                // until something touches it).
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
                buffer.label = "inkling2.experts.layer\(layer).mapped.cpu-only"
                mapped[layer] = try pool(spec, buffer: buffer, offset: shift)
            }
        }
        // Equal slot counts: each staged layer's slots, page-rounded, packed
        // largest first onto the slab with the fewest bytes so far; the slabs
        // grow to the fullest slab.
        let slabCount = max(2, plan.stagingSlabs)
        var placed: [Int: (slab: Int, offset: Int, slots: Int)] = [:]
        var minimumSlabBytes = 0
        if case .perLayer(let requested) = decodeSlots, !stagedLayers.isEmpty {
            let count = min(max(requested, shape.topK), shape.experts)
            var used = [Int](repeating: 0, count: slabCount)
            let order = stagedLayers.sorted { layers[$0].expertStride > layers[$1].expertStride ||
                (layers[$0].expertStride == layers[$1].expertStride && $0 < $1) }
            for layer in order {
                let bytes = (count * layers[layer].expertStride + pageSize - 1) / pageSize * pageSize
                let slab = used.indices.min { used[$0] < used[$1] || (used[$0] == used[$1] && $0 < $1) }!
                placed[layer] = (slab, used[slab], count)
                used[slab] += bytes
            }
            minimumSlabBytes = used.max() ?? 0
        }
        let staging = stagedFiles.isEmpty ? nil
            : try Inkling2StagedExpertLayers(device: device, files: stagedFiles, slabs: slabCount,
                                             readMode: readMode, minimumSlabBytes: minimumSlabBytes)

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
                let (slab, offset, slots) = placed[layer]
                    ?? (index / perSlab, (index % perSlab) * region, min(shape.experts, region / spec.expertStride))
                guard slots >= shape.topK else { throw StoreError.cacheTooSmall(layer: layer, slots: slots) }
                caches[layer] = try DecodeCache(layer: layer, spec: spec, shape: shape, slab: staging.slabs[slab],
                                                offset: offset, slots: slots, device: device)
            }
        }
        guard let decodeDone = device.makeSharedEvent() else { throw StoreError.allocationFailed("decode event") }
        self.decodeDone = decodeDone
        self.resident = resident
        self.mapped = mapped
        self.templates = templates
        self.caches = caches
        self.staging = staging
    }

    func isStaged(layer: Int) -> Bool {
        precondition(plan.placements.indices.contains(layer), "layer \(layer) out of range")
        return plan.placements[layer] == .staged
    }

    /// The staging reader's first error; after one, discard the store.
    var stagingError: Inkling2StagedExpertLayers.StagingError? { staging?.error }
    /// Bytes the staging reader has read (diagnostics).
    var stagingBytesRead: Int { staging?.bytesRead ?? 0 }
    /// Waits on the CPU until every queued staging read has run.
    func drainStaging() { staging?.drain() }

    /// Decode slots per staged layer (diagnostics, tests).
    func decodeSlots(layer: Int) -> Int? { caches[layer]?.slots }

    /// The pool a decode visit of a resident layer binds (router ids used on
    /// the GPU, no readback). Staged layers go through `prepareDecodeVisit`.
    func residentPool(layer: Int) -> Inkling2ExpertPool? { resident[layer] }

    /// Every layer's experts for CPU readers: a resident layer's buffer, or
    /// a staged layer's file mapped read-only. Never bind a staged layer's
    /// pool on the GPU (Metal would page in and pin the whole file); decode
    /// staged layers through `prepareDecodeVisit`.
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
        lock.lock()
        defer { lock.unlock() }
        // The fill must not overwrite decode slots a queued command still reads.
        let ticket = try staging.stage(layer: layer, after: (decodeDone, decodeIssued))
        staging.encodeWait(ticket, commandBuffer: commandBuffer)
        open[layer, default: []].append(ticket)
        cachesStale = true
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
    /// `Inkling2StagedExpertLayers.abandon`), and marks every decode visit
    /// ended so far as done. Call it only after every committed command
    /// buffer has completed.
    func abandonOpenVisits() throws {
        lock.lock()
        let tickets = open.values.flatMap { $0 }
        open.removeAll()
        if decodeDone.signaledValue < decodeIssued { decodeDone.signaledValue = decodeIssued }
        lock.unlock()
        try staging?.abandon(tickets)
    }

    /// After a command buffer failed, once every committed one has completed:
    /// its signals never ran, so every staged visit and decode visit issued
    /// so far is marked released and done, and the decode caches start empty
    /// (otherwise the next fill or slot refill would wait out the timeout).
    func recoverAfterFailedCommands() {
        lock.lock()
        open.removeAll()
        if decodeDone.signaledValue < decodeIssued { decodeDone.signaledValue = decodeIssued }
        cachesStale = true
        lock.unlock()
        staging?.releaseIssued()
    }

    /// Ends a decode visit: after the encoder that reads `binding.pool`, in
    /// the same command buffer. Its slots are released when the command
    /// buffer reaches this point.
    func endDecodeVisit(_ binding: Inkling2DecodeBinding, commandBuffer: MTLCommandBuffer) {
        lock.lock()
        decodeIssued += 1
        caches[binding.layer]?.lastRead = decodeIssued
        let value = decodeIssued
        lock.unlock()
        commandBuffer.encodeSignalEvent(decodeDone, value: value)
    }

    /// A decode visit of a staged layer, on the CPU after the runner has read
    /// back the visit's routed `expertIDs` (`topK` of them): loads the misses
    /// into the layer's cache slots (least recently used first, never a slot
    /// the visit itself needs), refreshes the slot scales and writes the
    /// slot ids. Before refilling, it waits for the GPU to finish the prefill
    /// visits that wrote the slabs and the last decode visit of this layer
    /// (`endDecodeVisit`). Encode the visit with the binding, then call
    /// `endDecodeVisit` in the same command buffer.
    func prepareDecodeVisit(layer: Int, expertIDs: [Int]) throws -> Inkling2DecodeBinding {
        guard let cache = caches[layer] else { throw StoreError.notStaged(layer: layer) }
        precondition(expertIDs.count == shape.topK && expertIDs.allSatisfy { $0 >= 0 && $0 < shape.experts })
        lock.lock()
        defer { lock.unlock() }
        if cachesStale, let staging {
            // Prefill wrote the slabs: wait for its last read into them and
            // for the GPU to finish reading them, then forget every cache.
            staging.drain()
            let last = staging.lastSequence
            if last > 0, staging.released.signaledValue < last,
               !staging.released.wait(untilSignaledValue: last, timeoutMS: waitTimeoutMS) {
                throw StoreError.timedOut("prefill reads of the staging slabs")
            }
            for c in caches.values { c.clear() }
            cachesStale = false
        }

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
        var loads: [(slot: Int, expert: Int, previous: Int, previousUse: UInt64)] = []
        for (rank, expert) in expertIDs.enumerated() where slotFor[rank] < 0 {
            if let slot = cache.slotOf[expert] {           // loaded for an earlier rank of this visit
                slotFor[rank] = slot
                hits += 1
                continue
            }
            var victim = -1
            for slot in 0..<cache.slots where !needed.contains(slot) && !loads.contains(where: { $0.slot == slot }) {
                if victim < 0 || cache.lastUse[slot] < cache.lastUse[victim] { victim = slot }
            }
            precondition(victim >= 0, "fewer slots than ranks")
            loads.append((victim, expert, cache.expertIn[victim], cache.lastUse[victim]))
            if cache.expertIn[victim] >= 0 { cache.slotOf[cache.expertIn[victim]] = nil }
            cache.expertIn[victim] = expert
            cache.slotOf[expert] = victim
            cache.lastUse[victim] = cache.clock
            slotFor[rank] = victim
        }
        if !loads.isEmpty {
            // The previous decode visit of this layer must be done reading.
            if decodeDone.signaledValue < cache.lastRead,
               !decodeDone.wait(untilSignaledValue: cache.lastRead, timeoutMS: waitTimeoutMS) {
                rollBack(cache, loads)
                throw StoreError.timedOut("decode reads of layer \(layer)'s slots")
            }
            do {
                try Self.readExperts(cache: cache, loads: loads.map { ($0.slot, $0.expert) },
                                     bypassCache: (decodeMissReadMode ?? readMode) == .uncached)
            } catch {
                // The slots may hold partial bytes: mark them empty.
                rollBack(cache, loads, keepPrevious: false)
                throw error
            }
            let gateUp = cache.gateUpScales.contents().bindMemory(to: Float.self, capacity: cache.slots)
            let down = cache.downScales.contents().bindMemory(to: Float.self, capacity: cache.slots)
            for load in loads {
                gateUp[load.slot] = cache.scale(cache.sourceGateUp, expert: load.expert)
                down[load.slot] = cache.scale(cache.sourceDown, expert: load.expert)
            }
        }
        let ids = cache.ids.contents().bindMemory(to: UInt32.self, capacity: shape.topK)
        for (rank, slot) in slotFor.enumerated() { ids[rank] = UInt32(slot) }
        return Inkling2DecodeBinding(layer: layer, pool: cache.pool, ids: cache.ids, hits: hits, misses: loads.count)
    }

    /// Undoes a visit's slot assignments: back to the evicted experts while
    /// the slots are untouched, or empty once a read may have written them.
    private func rollBack(_ cache: DecodeCache,
                          _ loads: [(slot: Int, expert: Int, previous: Int, previousUse: UInt64)],
                          keepPrevious: Bool = true) {
        for load in loads.reversed() {
            cache.slotOf[load.expert] = nil
            if keepPrevious && load.previous >= 0 {
                cache.expertIn[load.slot] = load.previous
                cache.slotOf[load.previous] = load.slot
                cache.lastUse[load.slot] = load.previousUse
            } else {
                cache.expertIn[load.slot] = -1
                cache.lastUse[load.slot] = 0
            }
        }
    }

    /// Reads each loaded expert's blob from the layer file into its slot, in
    /// parallel; `bypassCache` sets `F_NOCACHE`.
    private static func readExperts(cache: DecodeCache, loads: [(Int, Int)], bypassCache: Bool) throws {
        let fd = Darwin.open(cache.file.path, O_RDONLY)
        guard fd >= 0 else { throw StoreError.openFailed(path: cache.file.path, errno: errno) }
        defer { close(fd) }
        if bypassCache, fcntl(fd, F_NOCACHE, 1) == -1 {
            throw StoreError.readFailed(path: cache.file.path, errno: errno)
        }
        nonisolated(unsafe) let base = cache.slab.contents() + cache.offset
        nonisolated(unsafe) var code: Int32 = 0
        let codeLock = NSLock()
        let stride = cache.stride, bytes = cache.blobBytes, fileOffset = cache.file.offset
        let work = loads
        DispatchQueue.concurrentPerform(iterations: work.count) { i in
            let target = base + work[i].0 * stride
            let source = off_t(fileOffset) + off_t(work[i].1 * stride)
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
    /// `bypassCache` sets `F_NOCACHE` so the copy does not also fill the file
    /// cache (resident layers always; staged slabs by `readMode`).
    static func read(_ file: Inkling2ExpertLayerFile, into destination: UnsafeMutableRawPointer,
                     bypassCache: Bool) throws {
        let fd = Darwin.open(file.path, O_RDONLY)
        guard fd >= 0 else { throw StoreError.openFailed(path: file.path, errno: errno) }
        defer { close(fd) }
        // A descriptor that cannot bypass the cache fails the read rather
        // than fill the cache the caller asked to keep out of.
        if bypassCache, fcntl(fd, F_NOCACHE, 1) == -1 { throw StoreError.readFailed(path: file.path, errno: errno) }
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
