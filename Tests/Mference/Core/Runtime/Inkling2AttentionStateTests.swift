import Testing
import Foundation
import Metal
@testable import Mference

@Suite struct Inkling2VisitPlanTests {
    @Test func fortyEightVisitsWithLoopSlotsAfterTheStoredLayers() {
        let plan = Inkling2VisitPlan.bySlot
        #expect(plan.count == 48)
        #expect(Set(plan.map(\.slot)) == Set(0..<48))
        #expect(Inkling2VisitPlan.windowSlots.count == 40)
        #expect(Inkling2VisitPlan.globalSlots == [5, 11, 17, 23, 29, 35, 41, 47])
        #expect(Inkling2VisitPlan.slot(layer: 6, pass: 1) == 24)
        #expect(Inkling2VisitPlan.slot(layer: 17, pass: 2) == 47)
        for visit in plan {
            #expect(Inkling2VisitPlan.slot(layer: visit.layer, pass: visit.pass) == visit.slot)
            #expect(visit.isGlobal == ((visit.layer + 1) % 6 == 0))
        }
        let order = Inkling2VisitPlan.executionOrder.map(\.slot)
        #expect(order == Array(0..<18) + Array(24..<36) + Array(36..<48) + Array(18..<24))
        #expect(Inkling2VisitPlan.executionOrder.map(\.layer)
                == Array(0..<6) + Array(6..<18) + Array(6..<18) + Array(6..<18) + Array(18..<24))
    }
}

@Suite(.serialized) struct Inkling2AttentionStateTests {
    static let config = Inkling2AttentionState.Configuration(maxContext: 64, maxChunkTokens: 8, window: 16,
                                                             globalGrowthStep: 16)
    static let rowHalves = 1024

    /// Writes a recognisable row for (slot, position, keys/values) the way
    /// the epilogue would, at the slot the view maps the position to.
    static func write(_ state: Inkling2AttentionState, slot: Int, position: Int, salt: Int) {
        let view = state.kv(slot: slot)
        let physical = view.ringCapacity == 0 ? position : position % view.ringCapacity
        for (which, ref) in [view.keys, view.values].enumerated() {
            let row = (ref.buffer.contents() + ref.offset + physical * Inkling2AttentionState.rowBytes)
                .bindMemory(to: Float16.self, capacity: rowHalves)
            for i in 0..<rowHalves { row[i] = Float16(Float((slot * 7 + position * 3 + which + salt + i) % 251) / 64) }
        }
    }

    static func read(_ state: Inkling2AttentionState, slot: Int, position: Int) -> [Float16] {
        let view = state.kv(slot: slot)
        let physical = view.ringCapacity == 0 ? position : position % view.ringCapacity
        return [view.keys, view.values].flatMap { ref in
            let row = (ref.buffer.contents() + ref.offset + physical * Inkling2AttentionState.rowBytes)
                .bindMemory(to: Float16.self, capacity: rowHalves)
            return Array(UnsafeBufferPointer(start: row, count: rowHalves))
        }
    }

    static func fillTails(_ state: Inkling2AttentionState, salt: Int) {
        let count = state.tailBuffer.length / 4
        let p = state.tailBuffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { p[i] = Float((i &* 13 &+ salt) % 997) }
    }

    static func tails(_ state: Inkling2AttentionState) -> [Float] {
        let count = state.tailBuffer.length / 4
        return Array(UnsafeBufferPointer(start: state.tailBuffer.contents().bindMemory(to: Float.self, capacity: count),
                                         count: count))
    }

    /// Runs `tokens` positions forward in chunks of 8, writing every visit's rows.
    static func run(_ state: Inkling2AttentionState, tokens: Int, salt: Int) throws {
        var left = tokens
        while left > 0 {
            let n = min(8, left)
            try state.ensureCapacity(tokens: state.position + n)
            for p in state.position..<(state.position + n) {
                for slot in 0..<48 { write(state, slot: slot, position: p, salt: salt) }
            }
            state.advance(by: n)
            left -= n
        }
    }

    @Test func layoutViewsAndGrowth() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        #expect(state.ringCapacity == 24)
        #expect(state.globalCapacity == 16)
        #expect(state.kv(slot: 0).ringCapacity == 24)
        #expect(state.kv(slot: 5).ringCapacity == 0)
        #expect(state.mask(slot: 29) == .globalDefault)
        #expect(state.mask(slot: 30) == .window(16))
        #expect(Self.tails(state).allSatisfy { $0 == 0 })
        #expect(Inkling2VisitTails.floatsPerVisit == 19_968)
        #expect(state.tails(slot: 2).moeOutput.offset == 2 * 19_968 * 4 + 13_056 * 4)

        try Self.run(state, tokens: 16, salt: 1)
        let before = Self.read(state, slot: 11, position: 3)
        #expect(try state.ensureCapacity(tokens: 17))
        #expect(state.globalCapacity == 32)
        #expect(Self.read(state, slot: 11, position: 3) == before)
        #expect(throws: PrefixSnapshotError.self) { try state.ensureCapacity(tokens: 65) }
    }

    @Test func fullSnapshotRestoresTailsRingWindowAndGlobalRows() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 20, salt: 5)
        Self.fillTails(state, salt: 3)
        let tailsAt20 = Self.tails(state)
        var rows: [Int: [Int: [Float16]]] = [:]
        for slot in 0..<48 {
            let first = Inkling2VisitPlan.bySlot[slot].isGlobal ? 0 : 20 - 15
            for p in first..<20 { rows[slot, default: [:]][p] = Self.read(state, slot: slot, position: p) }
        }
        let region = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: true))
        let snapshot = try state.takeSnapshot(into: region, includeGlobalRows: true)
        #expect(snapshot.position == 20)

        // Diverge: new rows past 20 (they overwrite ring slots), new tails, then a reset.
        try Self.run(state, tokens: 30, salt: 9)
        Self.fillTails(state, salt: 77)
        state.reset()
        #expect(state.position == 0)
        #expect(Self.tails(state).allSatisfy { $0 == 0 })
        #expect(state.canRestore(snapshot))
        try state.restoreSnapshot(snapshot, from: region)
        #expect(state.position == 20)
        #expect(Self.tails(state) == tailsAt20)
        for (slot, byPosition) in rows {
            for (p, row) in byPosition { #expect(Self.read(state, slot: slot, position: p) == row, "slot \(slot) p \(p)") }
        }
    }

    @Test func lightSnapshotRestoresOnlyWhileGlobalRowsAreIntact() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 10, salt: 2)
        let early = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: true))
        let atTen = try state.takeSnapshot(into: early, includeGlobalRows: true)
        try Self.run(state, tokens: 10, salt: 2)
        let lightRegion = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: false))
        let light = try state.takeSnapshot(into: lightRegion, includeGlobalRows: false)
        // Header, 48 visits of tails, 40 rings x (K, V) x the last 15 rows; no global rows.
        #expect(light.byteCount == 64 + 48 * 19_968 * 4 + 40 * 2 * 15 * Inkling2AttentionState.rowBytes)
        let global20 = Self.read(state, slot: 23, position: 19)

        // Forward only: still valid; restoring it is itself a rewind to 20.
        try Self.run(state, tokens: 12, salt: 4)
        #expect(state.canRestore(light))
        try state.restoreSnapshot(light, from: lightRegion)
        #expect(state.position == 20)
        #expect(Self.read(state, slot: 23, position: 19) == global20)
        #expect(state.canRestore(light))

        // A rewind below 20 invalidates it; the full snapshot at 10 stays usable.
        try state.restoreSnapshot(atTen, from: early)
        #expect(!state.canRestore(light))
        #expect(throws: PrefixSnapshotError.self) { try state.restoreSnapshot(light, from: lightRegion) }
        #expect(state.position == 10)
        state.reset()
        #expect(state.canRestore(atTen))
    }

    /// A full snapshot restores global rows from another timeline: it must
    /// invalidate light snapshots taken since, at any position.
    @Test func fullRestoreInvalidatesLaterLightSnapshots() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 30, salt: 1)                      // conversation X
        let fullRegion = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: true))
        let full = try state.takeSnapshot(into: fullRegion, includeGlobalRows: true)
        state.reset()
        try Self.run(state, tokens: 20, salt: 2)                      // conversation Y
        let lightRegion = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: false))
        let light = try state.takeSnapshot(into: lightRegion, includeGlobalRows: false)
        #expect(state.canRestore(light))
        try state.restoreSnapshot(full, from: fullRegion)             // back to X at 30
        #expect(!state.canRestore(light))
        #expect(throws: PrefixSnapshotError.self) { try state.restoreSnapshot(light, from: lightRegion) }
        #expect(state.position == 30)
    }

    /// A snapshot record paired with a region holding a different snapshot is refused.
    @Test func restoreRefusesARegionHoldingAnotherSnapshot() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 12, salt: 3)
        let region = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: true))
        let first = try state.takeSnapshot(into: region, includeGlobalRows: true)
        state.reset()
        try Self.run(state, tokens: 12, salt: 4)
        let second = try state.takeSnapshot(into: region, includeGlobalRows: true)   // overwrites the region
        #expect(first.position == second.position && first.byteCount == second.byteCount)
        #expect(throws: PrefixSnapshotError.self) { try state.restoreSnapshot(first, from: region) }
        try state.restoreSnapshot(second, from: region)
    }

    @Test func memoryEstimateAtProductionShape() {
        let config = Inkling2AttentionState.Configuration(maxContext: 32_768, maxChunkTokens: 2048)
        // 40 rings x (K, V) x 3072 slots x 2 KiB + 48 x 78 KiB of tails + 8 x (K, V) x 32K x 2 KiB.
        let expected = 40 * 2 * 3072 * 2048 + 48 * 19_968 * 4 + 8 * 2 * 32_768 * 2048
        #expect(Inkling2AttentionState.memoryBytes(configuration: config, tokens: 32_768) == expected)
    }
}
