import Darwin
import Foundation
import Synchronization

// MARK: The multi-prefix cache and live memory
//
// The cache's byte ceiling (`--prompt-cache-bytes`) stays its admission
// limit. `PrefixCacheMemoryGovernor` adds what a ceiling cannot:
//
// - no growth while the system reports memory pressure, and a shed of
//   everything but the live prefix when it does;
// - the same for swap growth after the cache grew, because on some hosts swap
//   grows while the pressure level still reads normal;
// - a shadow of full `LiveMemoryBudget` admission, computed and reported but
//   never applied, to measure what enforcing it would cost.
//
// Everything the cache retains is anonymous memory, filled as soon as it is
// mapped, so the budget's live figures already count it for other owners.

/// Why the governor shed retained state.
public enum PrefixCacheShedReason: String, Sendable, Equatable {
    /// Memory pressure rose above normal: everything but the live prefix.
    case pressure
    /// Swap grew after the cache grew: back to its size before that growth.
    case swapGrowth
}

/// What the governor has done and seen, for diagnostics.
public struct PrefixCacheMemoryReport: Sendable, Equatable, Encodable {
    public var pressureSheds = 0
    public var swapSheds = 0
    /// Bytes released by sheds, all reasons.
    public var shedBytes = 0
    /// True while swap growth is blocking growth.
    public var refusingForSwap = false
    /// Swap used at the last safe point.
    public var swapUsedBytes: UInt64 = 0
    /// Shadow admission: what `LiveMemoryBudget` would grant under
    /// `.includingFileCache` at the last allocation the governor admitted
    /// (nil before the first one).
    public var grantableBytes: UInt64?
    /// Allocations the governor admitted, and how many of them full budget
    /// admission would have refused.
    public var admitted = 0
    public var wouldRefuse = 0
}

/// The server's live-memory policy for one `PrefixStateCache`.
///
/// The owner calls `requestStarted` before the cache's `begin`, then
/// `requestEnded` once the request has finished or failed, and
/// `idleSafePoint` when a pressure event arrives (via `setRaiseAction`). Those are
/// the safe points, where no request is using the cache; `idleSafePoint` does
/// nothing while a request is running, and that request's end acts instead.
/// All three run on the owner's serial executor, like every other call into
/// the cache. Only the pressure handler runs elsewhere, and it only raises a
/// flag.
public final class PrefixCacheMemoryGovernor: PrefixCacheMemoryGate, Sendable {
    /// WS2's fill-guard threshold.
    public static let defaultSwapGrowthLimit: UInt64 = 256 << 20
    /// Swap has stopped growing once it grew by less than `swapSettledBytes`
    /// over at least `swapSettleSeconds`, measured from the trip or from the
    /// last safe point where it was still growing.
    public static let swapSettledBytes: UInt64 = 16 << 20
    public static let swapSettleSeconds: Double = 60

    private struct Sample { let swap: UInt64; let retained: Int }

    private struct State {
        var pressureRaised = false
        var requestRunning = false
        var refusingForSwap = false
        /// While refusing: when and at what swap reading swap was last seen
        /// growing.
        var stillGrowing: (seconds: Double, swap: UInt64)?
        var previous: Sample?
        /// The safe point just before the cache last grew.
        var beforeGrowth: Sample?
        var report = PrefixCacheMemoryReport()
    }

    private let budget: LiveMemoryBudget
    private let swapUsed: @Sendable () -> UInt64
    private let now: @Sendable () -> Double
    private let swapGrowthLimit: UInt64
    private let state = Mutex(State())
    private let registration: Mutex<LiveMemoryBudget.PressureRegistration?> = Mutex(nil)
    private let raiseAction: Mutex<(@Sendable () -> Void)?> = Mutex(nil)

    /// Registers with `budget` for pressure events for as long as the
    /// governor lives.
    public init(budget: LiveMemoryBudget = .shared,
                swapUsed: @escaping @Sendable () -> UInt64 = PrefixCacheMemoryGovernor.systemSwapUsed,
                now: @escaping @Sendable () -> Double = PrefixCacheMemoryGovernor.uptimeSeconds,
                swapGrowthLimit: UInt64 = PrefixCacheMemoryGovernor.defaultSwapGrowthLimit) {
        self.budget = budget
        self.swapUsed = swapUsed
        self.now = now
        self.swapGrowthLimit = swapGrowthLimit
        let token = budget.onPressure(.prefixCache) { [weak self] level in
            guard level != .normal, let self else { return }
            self.state.withLock { $0.pressureRaised = true }
            self.raiseAction.withLock { $0 }?()
        }
        registration.withLock { $0 = token }
    }

    /// Called on the budget's pressure queue each time pressure rises above
    /// `.normal`. It should schedule `idleSafePoint` on the owner's executor
    /// and return.
    public func setRaiseAction(_ action: @escaping @Sendable () -> Void) {
        raiseAction.withLock { $0 = action }
    }

    /// `vm.swapusage`'s used bytes; 0 when the query fails.
    public static func systemSwapUsed() -> UInt64 {
        var swap = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        return sysctlbyname("vm.swapusage", &swap, &size, nil, 0) == 0 ? swap.xsu_used : 0
    }

    public static func uptimeSeconds() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1e9
    }

    public var report: PrefixCacheMemoryReport { state.withLock { $0.report } }
    var isPressureRaised: Bool { state.withLock { $0.pressureRaised } }

    // MARK: Gate

    private final class Grant: PrefixCacheMemoryGrant {
        func commit() {}
        func release() {}
    }

    public var admitsGrowth: Bool {
        budget.pressure == .normal && !state.withLock { $0.refusingForSwap }
    }

    public func reserve(bytes: Int) -> (any PrefixCacheMemoryGrant)? {
        guard admitsGrowth else { return nil }
        let grantable = budget.grantableBytes(basis: .includingFileCache)
        state.withLock {
            $0.report.admitted += 1
            $0.report.grantableBytes = grantable
            if UInt64(bytes) > grantable { $0.report.wouldRefuse += 1 }
        }
        return Grant()
    }

    // MARK: Safe points

    /// Before the cache's `begin`. Returns what it shed, if anything.
    @discardableResult
    public func requestStarted(_ cache: PrefixStateCache) -> PrefixCacheShedReason? {
        let shed = safePoint(cache)
        state.withLock { $0.requestRunning = true }
        return shed
    }

    /// After the request finished (`didFinish`) or failed (`didFail`).
    @discardableResult
    public func requestEnded(_ cache: PrefixStateCache) -> PrefixCacheShedReason? {
        state.withLock { $0.requestRunning = false }
        return safePoint(cache)
    }

    /// From the pressure hop while the server may be idle. Defers to the
    /// running request's end when there is one.
    @discardableResult
    public func idleSafePoint(_ cache: PrefixStateCache) -> PrefixCacheShedReason? {
        guard !state.withLock({ $0.requestRunning }) else { return nil }
        return safePoint(cache)
    }

    private func safePoint(_ cache: PrefixStateCache) -> PrefixCacheShedReason? {
        let swap = swapUsed()
        let seconds = now()
        let before = cache.retainedBytes
        var reason: PrefixCacheShedReason?
        var shrinkTo: Int?
        state.withLock { s in
            s.report.swapUsedBytes = swap
            if s.pressureRaised {
                s.pressureRaised = false
                reason = .pressure
                shrinkTo = 0
            }
            if s.refusingForSwap {
                let mark = s.stillGrowing ?? (seconds, swap)
                if swap >= mark.swap + Self.swapSettledBytes {
                    s.stillGrowing = (seconds, swap)
                } else if seconds - mark.seconds >= Self.swapSettleSeconds {
                    s.refusingForSwap = false
                    s.stillGrowing = nil
                }
            } else {
                if let previous = s.previous, before > previous.retained {
                    s.beforeGrowth = previous
                }
                if let anchor = s.beforeGrowth, swap > anchor.swap + swapGrowthLimit {
                    s.refusingForSwap = true
                    s.stillGrowing = (seconds, swap)
                    s.beforeGrowth = nil
                    s.report.swapSheds += 1
                    if reason == nil {
                        reason = .swapGrowth
                        shrinkTo = anchor.retained
                    }
                }
            }
            if reason == .pressure { s.report.pressureSheds += 1 }
            s.report.refusingForSwap = s.refusingForSwap
        }
        switch reason {
        case .pressure: cache.shed()
        case .swapGrowth: cache.shrink(toRetainedBytes: shrinkTo ?? 0)
        case nil: break
        }
        let after = cache.retainedBytes
        state.withLock {
            $0.report.shedBytes += before - after
            $0.previous = Sample(swap: swap, retained: after)
        }
        return reason
    }
}
