import Foundation
import Darwin
import Metal

/// System memory pressure as the kernel reports it to a dispatch source.
public enum MemoryPressureLevel: Sendable, Equatable {
    case normal
    case warning
    case critical
}

/// One reading of the host's live memory, in bytes.
///
/// `free` excludes speculative pages: `host_statistics64` folds them into
/// `free_count`, which is why `vm_stat` prints `free_count - speculative_count`
/// as "Pages free". So `free + speculative` is the kernel's own free figure and
/// nothing is counted twice.
public struct LiveMemorySnapshot: Sendable, Equatable {
    public let physical: UInt64
    public let free: UInt64
    public let inactive: UInt64
    public let purgeable: UInt64
    public let speculative: UInt64
    /// Diagnostics only; never part of an available view.
    public let wired: UInt64
    public let compressed: UInt64
    public let swapUsed: UInt64
    /// `MTLDevice.recommendedMaxWorkingSetSize`; nil without a device.
    public let gpuRecommendedMaxWorkingSet: UInt64?
    /// `MTLDevice.currentAllocatedSize`; nil without a device.
    public let gpuCurrentAllocated: UInt64?
    public let pressure: MemoryPressureLevel

    public init(physical: UInt64,
                free: UInt64,
                inactive: UInt64,
                purgeable: UInt64,
                speculative: UInt64,
                wired: UInt64 = 0,
                compressed: UInt64 = 0,
                swapUsed: UInt64 = 0,
                gpuRecommendedMaxWorkingSet: UInt64? = nil,
                gpuCurrentAllocated: UInt64? = nil,
                pressure: MemoryPressureLevel = .normal) {
        self.physical = physical
        self.free = free
        self.inactive = inactive
        self.purgeable = purgeable
        self.speculative = speculative
        self.wired = wired
        self.compressed = compressed
        self.swapUsed = swapUsed
        self.gpuRecommendedMaxWorkingSet = gpuRecommendedMaxWorkingSet
        self.gpuCurrentAllocated = gpuCurrentAllocated
        self.pressure = pressure
    }

    /// Memory an owner could take without forcing anything to swap, counting
    /// inactive pages (mostly the file cache) as reclaimable. Volatile
    /// purgeable pages also sit on the active and inactive queues, so adding
    /// `purgeable` to `inactive` could count the same pages twice; the larger
    /// of the two is a safe lower bound on their union.
    public var availableIncludingFileCache: UInt64 {
        Self.sum(free, speculative, max(inactive, purgeable))
    }

    /// The conservative view: inactive and speculative pages are left alone.
    /// While routed experts stream through the page cache, inactive pages are
    /// the expert pool and speculative pages are read-ahead of the same files,
    /// so spending either slows decode. This undercounts rather than overcounts.
    public var availableExcludingFileCache: UInt64 {
        Self.sum(free, purgeable)
    }

    func with(pressure level: MemoryPressureLevel) -> LiveMemorySnapshot {
        LiveMemorySnapshot(physical: physical, free: free, inactive: inactive,
                           purgeable: purgeable, speculative: speculative,
                           wired: wired, compressed: compressed, swapUsed: swapUsed,
                           gpuRecommendedMaxWorkingSet: gpuRecommendedMaxWorkingSet,
                           gpuCurrentAllocated: gpuCurrentAllocated,
                           pressure: level)
    }

    private static func sum(_ values: UInt64...) -> UInt64 {
        values.reduce(UInt64(0)) { acc, v in
            let (s, o) = acc.addingReportingOverflow(v)
            return o ? .max : s
        }
    }

    /// One send right for the life of the process: `mach_host_self()` adds a
    /// reference on every call.
    private static let hostPort = mach_host_self()

    /// Reads `host_statistics64(HOST_VM_INFO64)`, `vm.swapusage` and, when a
    /// device is given, its working-set figures. A failed host query reports
    /// zero available memory, so every reservation is refused (fail closed).
    public static func current(device: MTLDevice?,
                               pressure: MemoryPressureLevel = .normal) -> LiveMemorySnapshot {
        let physical = ProcessInfo.processInfo.physicalMemory
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(hostPort, HOST_VM_INFO64, $0, &count)
            }
        }
        // host_statistics64 counts in host pages (16 KiB on Apple silicon).
        var hostPage: vm_size_t = 0
        let page = host_page_size(hostPort, &hostPage) == KERN_SUCCESS
            ? UInt64(hostPage) : UInt64(getpagesize())
        var swap = xsw_usage()
        var swapSize = MemoryLayout<xsw_usage>.size
        let swapUsed = sysctlbyname("vm.swapusage", &swap, &swapSize, nil, 0) == 0
            ? swap.xsu_used : 0
        let gpuMax = device.map { $0.recommendedMaxWorkingSetSize }
        let gpuNow = device.map { UInt64($0.currentAllocatedSize) }
        guard result == KERN_SUCCESS else {
            return LiveMemorySnapshot(physical: physical, free: 0, inactive: 0,
                                      purgeable: 0, speculative: 0, swapUsed: swapUsed,
                                      gpuRecommendedMaxWorkingSet: gpuMax,
                                      gpuCurrentAllocated: gpuNow, pressure: pressure)
        }
        let freeCount = UInt64(stats.free_count)
        let speculativeCount = UInt64(stats.speculative_count)
        return LiveMemorySnapshot(
            physical: physical,
            free: (freeCount > speculativeCount ? freeCount - speculativeCount : 0) * page,
            inactive: UInt64(stats.inactive_count) * page,
            purgeable: UInt64(stats.purgeable_count) * page,
            speculative: speculativeCount * page,
            wired: UInt64(stats.wire_count) * page,
            compressed: UInt64(stats.compressor_page_count) * page,
            swapUsed: swapUsed,
            gpuRecommendedMaxWorkingSet: gpuMax,
            gpuCurrentAllocated: gpuNow,
            pressure: pressure)
    }
}

/// The one process-wide source of memory budget for runtime owners that size
/// themselves from live memory: resident routed experts (WS2 D1) and the
/// server's RAM prefix cache (WS3 K2). Both draw down from here so free RAM is
/// counted once.
///
/// grantable(basis) = available(basis) − headroom − Σ pending reservations.
///
/// A reservation is *pending* from `reserve` until its owner calls `commit()`
/// after allocating: from then on the allocation is visible in the live
/// statistics, so subtracting it again would count it twice. A `gpuResident`
/// reservation must also fit Metal's working set: current Metal allocation
/// (which already includes committed GPU memory) plus pending GPU-resident
/// reservations must stay under `recommendedMaxWorkingSetSize` minus a margin.
///
/// Pressure: one system source, created lazily on the first `onPressure`.
/// Handlers run on the helper's queue in a fixed owner order — the prefix
/// cache before resident experts — and must only set flags; owners act at
/// their own safe points.
///
/// No caller exists yet; adding this type changes no runtime behaviour.
public final class LiveMemoryBudget: @unchecked Sendable {
    public enum Owner: Sendable, Hashable, CaseIterable {
        case residentExperts
        case prefixCache
    }

    public enum Basis: Sendable {
        case includingFileCache
        case excludingFileCache
    }

    /// Owners are notified in this order on every pressure event.
    static let pressureOrder: [Owner] = [.prefixCache, .residentExperts]

    /// Memory no owner may take: max(4 GiB, physical / 8).
    public static func defaultHeadroom(physical: UInt64) -> UInt64 {
        max(UInt64(4) << 30, physical / 8)
    }

    /// Kept free inside the GPU working set for scratch and transient buffers.
    public static let gpuWorkingSetMargin = UInt64(512) << 20

    public static let shared = LiveMemoryBudget(
        snapshot: { LiveMemorySnapshot.current(device: LiveMemoryBudget.systemDevice) },
        installsSystemPressureSource: true)

    /// Through `MetalContext`'s factory, so the driver's one-time environment
    /// read sees the watchdog setting even when the budget is touched first.
    nonisolated(unsafe) private static let systemDevice: MTLDevice? = MetalContext.makeSystemDefaultDevice()

    public final class Reservation: @unchecked Sendable {
        public let owner: Owner
        public let bytes: UInt64
        public let gpuResident: Bool
        fileprivate enum State { case pending, committed, released }
        fileprivate var state: State = .pending
        private weak var budget: LiveMemoryBudget?

        fileprivate init(owner: Owner, bytes: UInt64, gpuResident: Bool,
                         budget: LiveMemoryBudget) {
            self.owner = owner
            self.bytes = bytes
            self.gpuResident = gpuResident
            self.budget = budget
        }

        /// The owner has allocated the memory **and made it resident** (pages
        /// touched, wired or filled): lazily faulted memory does not reduce the
        /// live figures until it is touched, so committing earlier would let
        /// another owner be granted the same bytes. Committed anonymous pages
        /// that later age onto the inactive queue can reappear in the
        /// including-file-cache view; owners whose memory is not kept resident
        /// by the GPU should draw with `.excludingFileCache`. Idempotent; no
        /// effect after release.
        public func commit() { budget?.transition(self, to: .committed) }

        /// Returns the reservation. Idempotent; also runs on deinit.
        public func release() { budget?.transition(self, to: .released) }

        deinit { release() }
    }

    private let lock = NSLock()
    private let snapshotProvider: @Sendable () -> LiveMemorySnapshot
    private let headroom: @Sendable (UInt64) -> UInt64
    private let installsSystemPressureSource: Bool

    // Guarded by `lock`.
    private var pendingBytes: UInt64 = 0
    private var pendingGPUBytes: UInt64 = 0
    private var ownerBytes: [Owner: UInt64] = [:]
    private var level: MemoryPressureLevel = .normal
    private var handlers: [Owner: [@Sendable (MemoryPressureLevel) -> Void]] = [:]
    private var pressureSource: DispatchSourceMemoryPressure?
    private let pressureQueue = DispatchQueue(label: "mference.live-memory-budget.pressure")

    init(snapshot: @escaping @Sendable () -> LiveMemorySnapshot,
         headroom: @escaping @Sendable (UInt64) -> UInt64 = { LiveMemoryBudget.defaultHeadroom(physical: $0) },
         installsSystemPressureSource: Bool = false) {
        self.snapshotProvider = snapshot
        self.headroom = headroom
        self.installsSystemPressureSource = installsSystemPressureSource
    }

    deinit { pressureSource?.cancel() }

    /// The live reading, with the pressure level last delivered to the budget.
    public func snapshot() -> LiveMemorySnapshot {
        lock.lock()
        let current = level
        lock.unlock()
        return snapshotProvider().with(pressure: current)
    }

    public var pressure: MemoryPressureLevel {
        lock.lock()
        defer { lock.unlock() }
        return level
    }

    /// Atomic draw-down. nil when `bytes` exceeds what is grantable under
    /// `basis`, or, for a GPU-resident reservation, the working-set room.
    public func reserve(_ bytes: UInt64, for owner: Owner, basis: Basis,
                        gpuResident: Bool = false) -> Reservation? {
        guard bytes > 0 else { return nil }
        lock.lock()
        defer { lock.unlock() }
        // Read the live figures under the lock: a snapshot taken before another
        // owner's allocation, combined with that owner's commit landing before
        // we lock, would count its bytes nowhere. The provider never calls back
        // into the budget.
        let snap = snapshotProvider()
        guard bytes <= grantableLocked(snap, basis: basis, gpuResident: gpuResident) else {
            return nil
        }
        pendingBytes += bytes
        if gpuResident { pendingGPUBytes += bytes }
        ownerBytes[owner, default: 0] += bytes
        return Reservation(owner: owner, bytes: bytes, gpuResident: gpuResident, budget: self)
    }

    public func grantableBytes(basis: Basis, gpuResident: Bool = false) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return grantableLocked(snapshotProvider(), basis: basis, gpuResident: gpuResident)
    }

    /// Pending plus committed bytes held by `owner`.
    public func reservedBytes(for owner: Owner) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return ownerBytes[owner] ?? 0
    }

    /// Registers a pressure handler. The first registration on the shared
    /// budget installs the system memory-pressure source.
    public func onPressure(_ owner: Owner,
                           _ handler: @escaping @Sendable (MemoryPressureLevel) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        handlers[owner, default: []].append(handler)
        if installsSystemPressureSource && pressureSource == nil {
            installPressureSourceLocked()
        }
    }

    /// Records `level` and calls handlers in `pressureOrder`, synchronously on
    /// the caller's thread. The system source calls it on the helper's queue;
    /// tests call it directly.
    func deliverPressure(_ newLevel: MemoryPressureLevel) {
        lock.lock()
        level = newLevel
        let ordered = Self.pressureOrder.flatMap { handlers[$0] ?? [] }
        lock.unlock()
        for handler in ordered { handler(newLevel) }
    }

    /// Called with `lock` held, so two first registrations cannot both create
    /// a source (an inactive dispatch source must never be dropped). The event
    /// handler runs later on `pressureQueue` and takes the lock itself.
    private func installPressureSourceLocked() {
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical], queue: pressureQueue)
        source.setEventHandler { [weak self, weak source] in
            guard let self, let event = source?.data else { return }
            let level: MemoryPressureLevel = event.contains(.critical) ? .critical
                : event.contains(.warning) ? .warning : .normal
            self.deliverPressure(level)
        }
        pressureSource = source
        source.activate()
    }

    private func grantableLocked(_ snap: LiveMemorySnapshot, basis: Basis,
                                 gpuResident: Bool) -> UInt64 {
        let available = basis == .includingFileCache
            ? snap.availableIncludingFileCache
            : snap.availableExcludingFileCache
        var grantable = Self.subtract(available, headroom(snap.physical), pendingBytes)
        if gpuResident {
            // Without a device there is no working-set figure: fail closed.
            guard let ceiling = snap.gpuRecommendedMaxWorkingSet else { return 0 }
            let gpuRoom = Self.subtract(ceiling, Self.gpuWorkingSetMargin,
                                        snap.gpuCurrentAllocated ?? 0, pendingGPUBytes)
            grantable = min(grantable, gpuRoom)
        }
        return grantable
    }

    private static func subtract(_ base: UInt64, _ values: UInt64...) -> UInt64 {
        values.reduce(base) { $0 > $1 ? $0 - $1 : 0 }
    }

    fileprivate func transition(_ reservation: Reservation, to next: Reservation.State) {
        lock.lock()
        defer { lock.unlock() }
        switch (reservation.state, next) {
        case (.pending, .committed):
            pendingBytes -= reservation.bytes
            if reservation.gpuResident { pendingGPUBytes -= reservation.bytes }
        case (.pending, .released):
            pendingBytes -= reservation.bytes
            if reservation.gpuResident { pendingGPUBytes -= reservation.bytes }
            ownerBytes[reservation.owner, default: 0] -= reservation.bytes
        case (.committed, .released):
            ownerBytes[reservation.owner, default: 0] -= reservation.bytes
        default:
            return   // idempotent: commit twice, commit after release, release twice
        }
        reservation.state = next
    }
}
