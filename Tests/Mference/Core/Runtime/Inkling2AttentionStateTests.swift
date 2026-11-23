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

    // MARK: Prefix snapshots for the multi-prefix cache

    /// Every window ring's rows the next token can read at `position`, and
    /// every global visit's rows below it.
    static func prefixRows(_ state: Inkling2AttentionState, at position: Int) -> [Int: [Int: [Float16]]] {
        var rows: [Int: [Int: [Float16]]] = [:]
        for slot in 0..<48 {
            let first = Inkling2VisitPlan.bySlot[slot].isGlobal ? 0 : max(0, position - 15)
            for p in first..<position { rows[slot, default: [:]][p] = read(state, slot: slot, position: p) }
        }
        return rows
    }

    static func expectRows(_ state: Inkling2AttentionState, _ rows: [Int: [Int: [Float16]]],
                           sourceLocation: SourceLocation = #_sourceLocation) {
        var differing = 0
        for (slot, byPosition) in rows {
            for (p, row) in byPosition where read(state, slot: slot, position: p) != row { differing += 1 }
        }
        #expect(differing == 0, "\(differing) rows differ", sourceLocation: sourceLocation)
    }

    @Test func globalRowCopyHoldsEachVisitsKeysThenValues() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 20, salt: 6)
        let count = 12
        #expect(Inkling2AttentionState.globalRowBytes(tokens: count) == 8 * 2 * count * 2048)
        let region = try PrefixSnapshotRegion(byteCount: Inkling2AttentionState.globalRowBytes(tokens: count))
        try state.copyGlobalRows(count: count, into: region)
        let halves = region.base.bindMemory(to: Float16.self, capacity: 8 * 2 * count * Self.rowHalves)
        for (i, slot) in Inkling2VisitPlan.globalSlots.enumerated() {
            for p in 0..<count {
                let row = Self.read(state, slot: slot, position: p)
                let keys = Array(UnsafeBufferPointer(start: halves + ((i * 2) * count + p) * Self.rowHalves,
                                                     count: Self.rowHalves))
                let values = Array(UnsafeBufferPointer(start: halves + ((i * 2 + 1) * count + p) * Self.rowHalves,
                                                       count: Self.rowHalves))
                #expect(keys + values == row, "visit \(slot) position \(p)")
            }
        }
        #expect(throws: PrefixSnapshotError.self) { try state.copyGlobalRows(count: 21, into: region) }
        #expect(throws: PrefixSnapshotError.self) { try state.copyGlobalRows(count: 0, into: region) }
        let small = try PrefixSnapshotRegion(byteCount: Inkling2AttentionState.globalRowBytes(tokens: count) - 1)
        #expect(throws: PrefixSnapshotError.self) { try state.copyGlobalRows(count: count, into: small) }
    }

    /// The cache's restored hit: conversation X's image and light snapshot
    /// come back byte for byte after Y overwrote the rings, the tails and the
    /// global rows. Then the cache's later rewind inside X (a prefix whose
    /// rows the image brought back) restores too, although the rewind log
    /// refuses it to the snapshot API.
    @Test func prefixImageRestoresAnotherConversationAndItsEarlierPrefix() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 12, salt: 1)                      // conversation X
        Self.fillTails(state, salt: 3)
        let tailsAt12 = Self.tails(state), rowsAt12 = Self.prefixRows(state, at: 12)
        let light12 = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: false))
        let record12 = try state.takeSnapshot(into: light12, includeGlobalRows: false)
        try Self.run(state, tokens: 8, salt: 1)
        Self.fillTails(state, salt: 4)
        let tailsAt20 = Self.tails(state), rowsAt20 = Self.prefixRows(state, at: 20)
        let light20 = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: false))
        _ = try state.takeSnapshot(into: light20, includeGlobalRows: false)
        let image = try PrefixSnapshotRegion(byteCount: Inkling2AttentionState.globalRowBytes(tokens: 20))
        try state.copyGlobalRows(count: 20, into: image)

        state.reset()
        try Self.run(state, tokens: 30, salt: 9)                      // conversation Y wraps every ring
        Self.fillTails(state, salt: 77)
        state.reset()

        try state.restorePrefix(globalRows: (image, 20), position: 20, light: light20)
        #expect(state.position == 20)
        #expect(Self.tails(state) == tailsAt20)
        Self.expectRows(state, rowsAt20)

        // The image's earlier prefix: its rows below 12 are X's again.
        #expect(!state.canRestore(record12))
        try state.restorePrefix(globalRows: nil, position: 12, light: light12)
        #expect(state.position == 12)
        #expect(Self.tails(state) == tailsAt12)
        Self.expectRows(state, rowsAt12)

        // An image longer than the prefix restores it too (rows past 12 are
        // rewritten before anything reads them).
        state.reset()
        try state.restorePrefix(globalRows: (image, 20), position: 12, light: light12)
        #expect(state.position == 12)
        #expect(Self.tails(state) == tailsAt12)
        Self.expectRows(state, rowsAt12)
    }

    /// The cache's rewound hit: a branch past 20 wraps the rings over the
    /// prefix's window rows and advances the tails; the light snapshot
    /// brings them back, and the global rows below 20 were never touched.
    @Test func prefixRewindRestoresTheRingsAfterABranch() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 20, salt: 2)
        Self.fillTails(state, salt: 5)
        let tails = Self.tails(state), rows = Self.prefixRows(state, at: 20)
        let light = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: false))
        _ = try state.takeSnapshot(into: light, includeGlobalRows: false)

        try Self.run(state, tokens: 25, salt: 8)                      // positions 20-44: slots 0-19 rewritten
        Self.fillTails(state, salt: 99)
        #expect(Self.read(state, slot: 0, position: 19) != rows[0]![19]!)

        try state.restorePrefix(globalRows: nil, position: 20, light: light)
        #expect(state.position == 20)
        #expect(Self.tails(state) == tails)
        Self.expectRows(state, rows)
    }

    /// Every refusal comes before the first copy: the cursor and the tails
    /// stay as they were.
    @Test func prefixRestoreRefusesBeforeCopying() throws {
        let device = try #require(MetalContext.makeSystemDefaultDevice())
        let state = try Inkling2AttentionState(device: device, configuration: Self.config)
        try Self.run(state, tokens: 12, salt: 3)
        let light12 = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: false))
        _ = try state.takeSnapshot(into: light12, includeGlobalRows: false)
        let full12 = try PrefixSnapshotRegion(byteCount: state.snapshotBytes(includeGlobalRows: true))
        _ = try state.takeSnapshot(into: full12, includeGlobalRows: true)
        let image12 = try PrefixSnapshotRegion(byteCount: Inkling2AttentionState.globalRowBytes(tokens: 12))
        try state.copyGlobalRows(count: 12, into: image12)
        try Self.run(state, tokens: 4, salt: 3)
        Self.fillTails(state, salt: 11)
        let tails = Self.tails(state)
        let blank = try PrefixSnapshotRegion(byteCount: light12.byteCount)

        func refused(_ body: @escaping () throws -> Void, _ why: Comment) {
            #expect(throws: PrefixSnapshotError.self, why) { try body() }
            #expect(state.position == 16, why)
            #expect(Self.tails(state) == tails, why)
        }
        refused({ try state.restorePrefix(globalRows: nil, position: 11, light: light12) }, "another position")
        refused({ try state.restorePrefix(globalRows: nil, position: 12, light: full12) }, "a full snapshot")
        refused({ try state.restorePrefix(globalRows: nil, position: 12, light: blank) }, "no snapshot")
        refused({ try state.restorePrefix(globalRows: (image12, 11), position: 12, light: light12) },
                "an image shorter than the prefix")
        refused({ try state.restorePrefix(globalRows: (image12, 13), position: 12, light: light12) },
                "an image region too small for its count")
        refused({ try state.restorePrefix(globalRows: nil, position: 0, light: light12) }, "an empty prefix")

        // Past the cursor without rows: refused; with the image, restored.
        try state.restorePrefix(globalRows: nil, position: 12, light: light12)
        #expect(state.position == 12)
        state.reset()
        #expect(throws: PrefixSnapshotError.self) {
            try state.restorePrefix(globalRows: nil, position: 12, light: light12)
        }
        #expect(state.position == 0)
        try state.restorePrefix(globalRows: (image12, 12), position: 12, light: light12)
        #expect(state.position == 12)
    }

    @Test func memoryEstimateAtProductionShape() {
        let config = Inkling2AttentionState.Configuration(maxContext: 32_768, maxChunkTokens: 2048)
        // 40 rings x (K, V) x 3072 slots x 2 KiB + 48 x 78 KiB of tails + 8 x (K, V) x 32K x 2 KiB.
        let expected = 40 * 2 * 3072 * 2048 + 48 * 19_968 * 4 + 8 * 2 * 32_768 * 2048
        #expect(Inkling2AttentionState.memoryBytes(configuration: config, tokens: 32_768) == expected)
    }
}
