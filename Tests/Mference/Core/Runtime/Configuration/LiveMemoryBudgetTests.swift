import Foundation
import Testing
@testable import Mference

/// The shared live-memory budget that resident experts and the prefix cache
/// both draw from. Every budget test injects its snapshot and installs no
/// system pressure source; only `currentSnapshotIsSelfConsistent` reads this
/// host, and its assertions hold on any host.
@Suite struct LiveMemoryBudgetTests {

    static let gib = UInt64(1) << 30
    static let mib = UInt64(1) << 20

    /// Mutable injected snapshot: tests change the live figures between calls
    /// to model allocations landing in the kernel's counts.
    final class Host: @unchecked Sendable {
        private let lock = NSLock()
        private var value: LiveMemorySnapshot
        init(_ value: LiveMemorySnapshot) { self.value = value }
        var snapshot: LiveMemorySnapshot {
            lock.lock(); defer { lock.unlock() }
            return value
        }
        func set(_ value: LiveMemorySnapshot) {
            lock.lock(); self.value = value; lock.unlock()
        }
    }

    static func snap(physical: UInt64 = 32 * gib,
                     free: UInt64 = 0, inactive: UInt64 = 0,
                     purgeable: UInt64 = 0, speculative: UInt64 = 0,
                     external: UInt64? = nil,
                     gpuMax: UInt64? = nil, gpuNow: UInt64? = nil) -> LiveMemorySnapshot {
        LiveMemorySnapshot(physical: physical, free: free, inactive: inactive,
                           purgeable: purgeable, speculative: speculative,
                           external: external,
                           gpuRecommendedMaxWorkingSet: gpuMax,
                           gpuCurrentAllocated: gpuNow)
    }

    static func budget(_ host: Host) -> LiveMemoryBudget {
        LiveMemoryBudget(snapshot: { host.snapshot })
    }

    @Test func availableViewsSumTheRightPageClasses() {
        let s = Self.snap(free: 3 * Self.gib, inactive: 10 * Self.gib,
                          purgeable: 1 * Self.gib, speculative: 2 * Self.gib)
        // Purgeable pages may also be inactive: the union is bounded below
        // by the larger set, never by their sum.
        #expect(s.availableIncludingFileCache == 15 * Self.gib)
        // Inactive and speculative (file read-ahead) pages are left alone.
        #expect(s.availableExcludingFileCache == 4 * Self.gib)
        let mostlyPurgeable = Self.snap(free: 1 * Self.gib, inactive: 2 * Self.gib,
                                        purgeable: 5 * Self.gib)
        #expect(mostlyPurgeable.availableIncludingFileCache == 6 * Self.gib)
        #expect(mostlyPurgeable.availableExcludingFileCache == 6 * Self.gib)
    }

    @Test func availableViewsSaturateInsteadOfOverflowing() {
        let s = Self.snap(free: .max, inactive: 1)
        #expect(s.availableIncludingFileCache == .max)
    }

    @Test func headroomIsFourGiBOrAnEighthOfRAM() {
        #expect(LiveMemoryBudget.defaultHeadroom(physical: 16 * Self.gib) == 4 * Self.gib)
        #expect(LiveMemoryBudget.defaultHeadroom(physical: 32 * Self.gib) == 4 * Self.gib)
        #expect(LiveMemoryBudget.defaultHeadroom(physical: 64 * Self.gib) == 8 * Self.gib)
        #expect(LiveMemoryBudget.defaultHeadroom(physical: 128 * Self.gib) == 16 * Self.gib)
    }

    @Test func grantableSubtractsHeadroomUnderEitherBasis() {
        let host = Host(Self.snap(free: 6 * Self.gib, inactive: 16 * Self.gib))
        let budget = Self.budget(host)
        #expect(budget.grantableBytes(basis: .includingFileCache) == 18 * Self.gib)
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 2 * Self.gib)
    }

    @Test func reservationLargerThanGrantableIsRefused() {
        let host = Host(Self.snap(free: 6 * Self.gib))
        let budget = Self.budget(host)
        #expect(budget.reserve(2 * Self.gib + 1, for: .prefixCache,
                               basis: .excludingFileCache) == nil)
        let exact = budget.reserve(2 * Self.gib, for: .prefixCache, basis: .excludingFileCache)
        #expect(exact != nil)
        #expect(budget.reservedBytes(for: .prefixCache) == 2 * Self.gib)
        withExtendedLifetime(exact) {}
    }

    @Test func zeroByteReservationIsRefused() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        #expect(budget.reserve(0, for: .prefixCache, basis: .excludingFileCache) == nil)
        #expect(budget.reserve(0, for: .residentExperts, basis: .includingFileCache,
                               gpuResident: true) == nil)
    }

    /// Two owners cannot both claim the same free memory: a pending
    /// reservation by one owner reduces what the other can take.
    @Test func pendingReservationIsNotGrantedTwiceAcrossOwners() {
        let host = Host(Self.snap(free: 4 * Self.gib, inactive: 20 * Self.gib))
        let budget = Self.budget(host)
        let experts = budget.reserve(18 * Self.gib, for: .residentExperts,
                                     basis: .includingFileCache)
        #expect(experts != nil)
        #expect(budget.grantableBytes(basis: .includingFileCache) == 2 * Self.gib)
        #expect(budget.reserve(3 * Self.gib, for: .prefixCache,
                               basis: .includingFileCache) == nil)
        #expect(budget.reserve(2 * Self.gib, for: .prefixCache,
                               basis: .includingFileCache) != nil)
        withExtendedLifetime(experts) {}
    }

    /// Once committed, the allocation shows up in the live figures; the
    /// budget must stop subtracting it or it counts the memory twice.
    @Test func committedReservationIsNotSubtractedAgain() {
        let host = Host(Self.snap(free: 4 * Self.gib, inactive: 20 * Self.gib))
        let budget = Self.budget(host)
        let r = budget.reserve(10 * Self.gib, for: .residentExperts, basis: .includingFileCache)!
        #expect(budget.grantableBytes(basis: .includingFileCache) == 10 * Self.gib)
        // The owner allocates: the kernel now reports 10 GiB less available.
        host.set(Self.snap(free: 4 * Self.gib, inactive: 10 * Self.gib))
        r.commit()
        #expect(budget.grantableBytes(basis: .includingFileCache) == 10 * Self.gib)
        #expect(budget.reservedBytes(for: .residentExperts) == 10 * Self.gib)
        r.release()
        #expect(budget.reservedBytes(for: .residentExperts) == 0)
    }

    @Test func releaseAndCommitAreIdempotent() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        let r = budget.reserve(1 * Self.gib, for: .prefixCache, basis: .excludingFileCache)!
        r.commit(); r.commit()
        #expect(budget.reservedBytes(for: .prefixCache) == 1 * Self.gib)
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 6 * Self.gib)
        r.release(); r.release(); r.commit()
        #expect(budget.reservedBytes(for: .prefixCache) == 0)
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 6 * Self.gib)
    }

    @Test func releasingAPendingReservationReturnsItsBytes() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        let r = budget.reserve(5 * Self.gib, for: .prefixCache, basis: .excludingFileCache)!
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 1 * Self.gib)
        r.release()
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 6 * Self.gib)
    }

    @Test func droppingAReservationReleasesIt() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        do {
            let r = budget.reserve(5 * Self.gib, for: .prefixCache, basis: .excludingFileCache)
            #expect(r != nil)
        }
        #expect(budget.reservedBytes(for: .prefixCache) == 0)
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 6 * Self.gib)
    }

    @Test func reservedBytesSumsSeveralReservationsPerOwner() {
        let host = Host(Self.snap(free: 20 * Self.gib))
        let budget = Self.budget(host)
        let a = budget.reserve(640 * Self.mib, for: .prefixCache, basis: .excludingFileCache)!
        let b = budget.reserve(61 * Self.mib, for: .prefixCache, basis: .excludingFileCache)!
        a.commit()
        #expect(budget.reservedBytes(for: .prefixCache) == 701 * Self.mib)
        #expect(budget.reservedBytes(for: .residentExperts) == 0)
        b.release()
        #expect(budget.reservedBytes(for: .prefixCache) == 640 * Self.mib)
        withExtendedLifetime(a) {}
    }

    @Test func gpuResidentReservationFailsClosedWithoutADevice() {
        let host = Host(Self.snap(free: 30 * Self.gib))
        let budget = Self.budget(host)
        #expect(budget.grantableBytes(basis: .includingFileCache, gpuResident: true) == 0)
        #expect(budget.reserve(1, for: .residentExperts, basis: .includingFileCache,
                               gpuResident: true) == nil)
        #expect(budget.reserve(1, for: .prefixCache, basis: .includingFileCache) != nil)
    }

    /// Current Metal allocation already includes committed GPU memory, so the
    /// ceiling subtracts only pending GPU-resident reservations on top of it.
    @Test func gpuCeilingCountsPendingOnTopOfCurrentAllocation() {
        let host = Host(Self.snap(free: 4 * Self.gib, inactive: 26 * Self.gib,
                                  gpuMax: 22 * Self.gib, gpuNow: 2 * Self.gib))
        let budget = Self.budget(host)
        // RAM allows 26 GiB; the GPU allows 22 − 0.5 − 2 = 19.5 GiB.
        #expect(budget.grantableBytes(basis: .includingFileCache, gpuResident: true)
                == 19 * Self.gib + 512 * Self.mib)
        let pool = budget.reserve(18 * Self.gib, for: .residentExperts,
                                  basis: .includingFileCache, gpuResident: true)!
        #expect(budget.grantableBytes(basis: .includingFileCache, gpuResident: true)
                == 1 * Self.gib + 512 * Self.mib)
        // A CPU-only reservation is limited by RAM alone.
        #expect(budget.grantableBytes(basis: .includingFileCache) == 8 * Self.gib)
        // Allocation lands in both the kernel and Metal figures, then commit.
        host.set(Self.snap(free: 4 * Self.gib, inactive: 8 * Self.gib,
                           gpuMax: 22 * Self.gib, gpuNow: 20 * Self.gib))
        pool.commit()
        #expect(budget.grantableBytes(basis: .includingFileCache, gpuResident: true)
                == 1 * Self.gib + 512 * Self.mib)
        #expect(budget.grantableBytes(basis: .includingFileCache) == 8 * Self.gib)
    }

    @Test func pressureReachesThePrefixCacheBeforeResidentExperts() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        final class Log: @unchecked Sendable {
            let lock = NSLock()
            var events: [String] = []
            func add(_ e: String) { lock.lock(); events.append(e); lock.unlock() }
        }
        let log = Log()
        // Registration order is deliberately the reverse of the delivery order.
        let experts = budget.onPressure(.residentExperts) { log.add("experts:\($0)") }
        let prefix = budget.onPressure(.prefixCache) { log.add("prefix:\($0)") }
        defer { withExtendedLifetime((experts, prefix)) {} }
        #expect(budget.pressure == .normal)
        budget.deliverPressure(.warning)
        budget.deliverPressure(.critical)
        #expect(log.events == ["prefix:warning", "experts:warning",
                               "prefix:critical", "experts:critical"])
        #expect(budget.pressure == .critical)
        #expect(budget.snapshot().pressure == .critical)
        budget.deliverPressure(.normal)
        #expect(budget.snapshot().pressure == .normal)
    }

    /// Inactive anonymous pages (for example committed Metal buffers that
    /// aged out) are not reclaimable without swap and never count.
    @Test func inactiveAnonymousPagesAreNotAvailable() {
        let s = Self.snap(free: 2 * Self.gib, inactive: 12 * Self.gib,
                          speculative: 1 * Self.gib, external: 3 * Self.gib)
        #expect(s.availableIncludingFileCache == 6 * Self.gib)
        let purgeableWins = Self.snap(free: 2 * Self.gib, inactive: 12 * Self.gib,
                                      purgeable: 5 * Self.gib, external: 3 * Self.gib)
        #expect(purgeableWins.availableIncludingFileCache == 7 * Self.gib)
    }

    /// While pressure is above normal nothing is grantable; normal restores it.
    @Test func pressureAboveNormalRefusesEveryReservation() {
        let host = Host(Self.snap(free: 20 * Self.gib, gpuMax: 22 * Self.gib, gpuNow: 0))
        let budget = Self.budget(host)
        for level in [MemoryPressureLevel.warning, .critical] {
            budget.deliverPressure(level)
            #expect(budget.grantableBytes(basis: .excludingFileCache) == 0)
            #expect(budget.reserve(1, for: .prefixCache, basis: .excludingFileCache) == nil)
            #expect(budget.reserve(1, for: .residentExperts, basis: .includingFileCache,
                                   gpuResident: true) == nil)
        }
        budget.deliverPressure(.normal)
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 16 * Self.gib)
    }

    @Test func cancelledOrDroppedHandlersAreNotCalled() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        final class Count: @unchecked Sendable {
            let lock = NSLock(); var n = 0
            func bump() { lock.lock(); n += 1; lock.unlock() }
            var value: Int { lock.lock(); defer { lock.unlock() }; return n }
        }
        let kept = Count(), cancelled = Count(), dropped = Count()
        let keep = budget.onPressure(.residentExperts) { _ in kept.bump() }
        let cancel = budget.onPressure(.residentExperts) { _ in cancelled.bump() }
        do { _ = budget.onPressure(.prefixCache) { _ in dropped.bump() } }
        budget.deliverPressure(.warning)
        cancel.cancel(); cancel.cancel()
        budget.deliverPressure(.critical)
        #expect(kept.value == 2)
        #expect(cancelled.value == 1)
        #expect(dropped.value == 0)
        withExtendedLifetime(keep) {}
    }

    /// Handlers run outside the budget's lock, so one may call back in.
    @Test func handlerMayCallBackIntoTheBudget() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        final class Box: @unchecked Sendable {
            var grantable: UInt64 = .max
            var nested: LiveMemoryBudget.PressureRegistration?
        }
        let box = Box()
        let registration = budget.onPressure(.prefixCache) { _ in
            box.grantable = budget.grantableBytes(basis: .excludingFileCache)
            _ = budget.reserve(1, for: .prefixCache, basis: .excludingFileCache)
            box.nested = budget.onPressure(.residentExperts) { _ in }
        }
        budget.deliverPressure(.warning)
        #expect(box.grantable == 0)
        #expect(box.nested != nil)
        withExtendedLifetime(registration) {}
    }

    /// When RAM, not the GPU working set, is the tighter limit, it binds.
    @Test func ramBindsAGPUResidentReservationWhenTighter() {
        let host = Host(Self.snap(free: 4 * Self.gib, inactive: 6 * Self.gib,
                                  gpuMax: 22 * Self.gib, gpuNow: 1 * Self.gib))
        let budget = Self.budget(host)
        #expect(budget.grantableBytes(basis: .includingFileCache, gpuResident: true)
                == 6 * Self.gib)
        #expect(budget.reserve(6 * Self.gib + 1, for: .residentExperts,
                               basis: .includingFileCache, gpuResident: true) == nil)
    }

    /// Live memory shrinking below the pending total saturates at zero.
    @Test func grantableSaturatesWhenLiveMemoryShrinks() {
        let host = Host(Self.snap(free: 10 * Self.gib))
        let budget = Self.budget(host)
        let r = budget.reserve(5 * Self.gib, for: .prefixCache, basis: .excludingFileCache)!
        host.set(Self.snap(free: 3 * Self.gib))
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 0)
        #expect(budget.reserve(1, for: .prefixCache, basis: .excludingFileCache) == nil)
        withExtendedLifetime(r) {}
    }

    @Test func reservationThatOutlivesItsBudgetIsHarmless() {
        var r: LiveMemoryBudget.Reservation?
        do {
            let host = Host(Self.snap(free: 10 * Self.gib))
            let budget = Self.budget(host)
            r = budget.reserve(1 * Self.gib, for: .prefixCache, basis: .excludingFileCache)
        }
        r?.commit()
        r?.release()
        r = nil
        #expect(r == nil)
    }

    /// Concurrent draws never grant more than the budget holds.
    @Test func concurrentReservationsNeverOvergrant() {
        let host = Host(Self.snap(free: 14 * Self.gib))   // 10 GiB grantable
        let budget = Self.budget(host)
        final class Held: @unchecked Sendable {
            let lock = NSLock(); var items: [LiveMemoryBudget.Reservation] = []
            func add(_ r: LiveMemoryBudget.Reservation) { lock.lock(); items.append(r); lock.unlock() }
        }
        let held = Held()
        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            if let r = budget.reserve(1 * Self.gib, for: .prefixCache, basis: .excludingFileCache) {
                held.add(r)
            }
        }
        #expect(held.items.count == 10)
        #expect(budget.reservedBytes(for: .prefixCache) == 10 * Self.gib)
        #expect(budget.grantableBytes(basis: .excludingFileCache) == 0)
    }

    @Test func currentSnapshotIsSelfConsistent() {
        let s = LiveMemorySnapshot.current(device: nil)
        #expect(s.physical == ProcessInfo.processInfo.physicalMemory)
        #expect(s.availableIncludingFileCache >= s.availableExcludingFileCache)
        #expect(s.availableIncludingFileCache <= s.physical)
        #expect(s.external <= s.physical)
        #expect(s.gpuRecommendedMaxWorkingSet == nil)
    }
}
