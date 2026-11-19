import Foundation
import Metal
import Testing
@testable import Mference

/// Reference KV store and memory planner semantics.
@Suite("ReferenceKVStoreTests", .serialized)
struct ReferenceKVStoreTests {
    static func store(context: Int = 200, verify: Int = 4, prefixBudget: UInt64 = 1 << 26,
                      device: MTLDevice) throws -> Native.ReferenceKVStore {
        let plan = try Native.ReferenceMemoryPlanner().resolve(
            config: RefTest.tiny, weightsMapped: 0, contextLength: context, precision: .f16, maxVerifyRows: verify,
            maxPrefillChunk: 64, activationBytes: 0, prefixCacheRAMBudget: prefixBudget, budget: 1 << 30)
        return try Native.ReferenceKVStore(config: RefTest.tiny, plan: plan, device: device)
    }

    static func fill(_ ref: Native.RecurrentStateRef, _ value: Float) {
        let p = ref.buffer.contents().advanced(by: ref.offset).assumingMemoryBound(to: Float.self)
        for i in 0..<ref.layout.layerBytes / 4 { p[i] = value }
    }

    static func first(_ ref: Native.RecurrentStateRef) -> Float {
        ref.buffer.contents().advanced(by: ref.offset + ref.layout.ssmOffset).load(as: Float.self)
    }

    static func sync(_ queue: MTLCommandQueue, _ body: (MTLCommandBuffer) throws -> Void) rethrows {
        let cb = queue.makeCommandBuffer()!
        try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    @Test func plannerAccountsAndRefuses() throws {
        let c = RefTest.tiny
        let planner = Native.ReferenceMemoryPlanner()
        let plan = try planner.resolve(config: c, weightsMapped: 1000, contextLength: 130, precision: .int8,
                                       maxVerifyRows: 3, maxPrefillChunk: 16, activationBytes: 77,
                                       prefixCacheRAMBudget: 5, budget: 1 << 30)
        let g = Native.ReferenceMemoryPlanner.geometry(config: c, precision: .int8)
        let l = Native.ReferenceMemoryPlanner.recurrentLayout(config: c)
        #expect(plan.planned.kvArena == UInt64(3 * g.pageBytes))
        #expect(plan.planned.recurrentState == UInt64(2 * 2 * l.layerBytes))
        #expect(plan.planned.verifyArena == UInt64(2 * 3 * l.layerBytes))
        #expect(plan.planned.weightsMapped == 1000 && plan.planned.activations == 77 && plan.planned.prefixCacheRAM == 5)
        #expect(throws: Native.NativeError.self) {
            _ = try planner.resolve(config: c, weightsMapped: 1000, contextLength: 130, precision: .int8,
                                    maxVerifyRows: 3, maxPrefillChunk: 16, activationBytes: 77,
                                    prefixCacheRAMBudget: 5, budget: plan.planned.total - 1)
        }
    }

    @Test func stepsBanksAndRollback() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 0, commandBuffer: ctx.queue.makeCommandBuffer()!) }
        try kv.beginStep(rows: 10, kind: .prefill)
        let s1 = kv.recurrentState(gdnLayer: 1)
        #expect(s1.input.offset != s1.output.offset)
        Self.fill(s1.output, 1)
        try kv.beginStep(rows: 1, kind: .decode)
        let s2 = kv.recurrentState(gdnLayer: 1)
        #expect(s2.input.offset == s1.output.offset && s2.output.offset == s1.input.offset)
        Self.fill(s2.output, 2)
        // Undo the decode step: the live bank goes back to the prefill's output.
        try Self.sync(ctx.queue) { try kv.rollback(to: 10, commandBuffer: $0) }
        #expect(kv.length == 10)
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 10, commandBuffer: ctx.queue.makeCommandBuffer()!) }
        try kv.beginStep(rows: 4, kind: .verify)
        let s3 = kv.recurrentState(gdnLayer: 1)
        #expect(s3.input.offset == s1.output.offset)
        #expect(Self.first(s3.input) == 1)
        // A verify step keeping rows 0...1 copies arena slot 1 into the output bank.
        let arena = kv.verifyArena(gdnLayer: 1)
        for r in 0..<4 {
            Self.fill(Native.RecurrentStateRef(layout: arena.layout, buffer: arena.buffer,
                                               offset: arena.offset + r * arena.slotStride), Float(10 + r))
        }
        Self.fill(s3.output, 99)
        try Self.sync(ctx.queue) { try kv.rollback(to: 12, commandBuffer: $0) }
        #expect(kv.length == 12)
        #expect(Self.first(s3.output) == 11)
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(kv.recurrentState(gdnLayer: 1).input.offset == s3.output.offset)
        #expect(throws: Native.NativeError.self) { try kv.beginStep(rows: 200, kind: .prefill) }
    }

    /// Keeping only some rows of a step is legal only after a verify step;
    /// after a prefill (or a kind-less step) the arena slots are stale and the
    /// store must refuse instead of restoring one.
    @Test func partialRollbackOnlyAfterVerify() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        try kv.beginStep(rows: 4, kind: .prefill)
        #expect(throws: Native.NativeError.self) {
            try kv.rollback(to: 2, commandBuffer: ctx.queue.makeCommandBuffer()!)
        }
        try Self.sync(ctx.queue) { try kv.rollback(to: 4, commandBuffer: $0) }
        #expect(kv.length == 4)
        try kv.beginStep(rows: 3)
        #expect(throws: Native.NativeError.self) {
            try kv.rollback(to: 5, commandBuffer: ctx.queue.makeCommandBuffer()!)
        }
        try Self.sync(ctx.queue) { try kv.rollback(to: 4, commandBuffer: $0) }
        try kv.beginStep(rows: 3, kind: .verify)
        try Self.sync(ctx.queue) { try kv.rollback(to: 5, commandBuffer: $0) }
        #expect(kv.length == 5)
    }

    @Test func resetAndPrefixCache() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        let tokens: [Int32] = Array(0..<70)
        try kv.beginStep(rows: 70, kind: .prefill)
        Self.fill(kv.recurrentState(gdnLayer: 0).output, 3)
        // Mark K of token 65 (logical page 1).
        let table = kv.pageTable()
        let page1 = Int(table.pageTable.contents().advanced(by: table.pageTableOffset + 4).load(as: UInt32.self))
        let off = page1 * kv.geometry.pageBytes + kv.geometry.valueOffset(attnLayer: 0, kv: 0, kvHead: 0, token: 1, dim: 0)
        kv.arena.contents().advanced(by: off).storeBytes(of: Float16(7.5), as: Float16.self)
        let handle = try Self.syncResult(ctx.queue) { try kv.insertPrefix(tokens, commandBuffer: $0) }
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        #expect(kv.length == 0)
        #expect(Self.first(kv.recurrentState(gdnLayer: 0).input) == 0)
        #expect(kv.lookupPrefix(Array(tokens.prefix(69))) == nil)
        let match = try #require(kv.lookupPrefix(tokens + [5, 6]))
        #expect(match.handle == handle && match.length == 70)
        try Self.sync(ctx.queue) { try kv.attachPrefix(match, commandBuffer: $0) }
        #expect(kv.length == 70)
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(Self.first(kv.recurrentState(gdnLayer: 0).input) == 3)
        let t2 = kv.pageTable()
        let p1 = Int(t2.pageTable.contents().advanced(by: t2.pageTableOffset + 4).load(as: UInt32.self))
        let off2 = p1 * kv.geometry.pageBytes + kv.geometry.valueOffset(attnLayer: 0, kv: 0, kvHead: 0, token: 1, dim: 0)
        #expect(kv.arena.contents().advanced(by: off2).load(as: Float16.self) == 7.5)
        #expect(kv.footprint.prefixCacheRAM > 0)
    }

    @Test func prefixBudgetIsEnforced() throws {
        let ctx = try MetalContext()
        // A snapshot larger than the whole budget is refused for that reason.
        let tiny = try Self.store(prefixBudget: 1024, device: ctx.device)
        try tiny.beginStep(rows: 3, kind: .prefill)
        do {
            _ = try Self.syncResult(ctx.queue) { try tiny.insertPrefix([1, 2, 3], commandBuffer: $0) }
            Issue.record("insertPrefix fit a 1 KiB budget")
        } catch Native.NativeError.memoryBudgetExceeded(_, let budget) {
            #expect(budget == 1024)
        }
        // A budget of exactly one snapshot keeps the most recent one.
        let probe = try Self.store(device: ctx.device)
        let one = UInt64(probe.geometry.pageBytes + probe.gdnLayerCount * probe.recurrentLayout.layerBytes)
        let kv = try Self.store(prefixBudget: one, device: ctx.device)
        try kv.beginStep(rows: 3, kind: .prefill)
        _ = try Self.syncResult(ctx.queue) { try kv.insertPrefix([1, 2, 3], commandBuffer: $0) }
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        try kv.beginStep(rows: 3, kind: .prefill)
        _ = try Self.syncResult(ctx.queue) { try kv.insertPrefix([4, 5, 6], commandBuffer: $0) }
        #expect(kv.lookupPrefix([1, 2, 3, 9]) == nil)
        #expect(kv.lookupPrefix([4, 5, 6, 9])?.length == 3)
        #expect(kv.footprint.prefixCacheRAM == one)
    }

    /// insertPrefix, attachPrefix and reset end the undoable step.
    @Test func rollbackIsRefusedAfterCacheOperationsAndReset() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        func refused() {
            #expect(throws: Native.NativeError.self) {
                try kv.rollback(to: kv.length - 1, commandBuffer: ctx.queue.makeCommandBuffer()!)
            }
        }
        try kv.beginStep(rows: 4, kind: .prefill)
        let handle = try Self.syncResult(ctx.queue) { try kv.insertPrefix([1, 2, 3, 4], commandBuffer: $0) }
        refused()
        try Self.sync(ctx.queue) { try kv.reset(commandBuffer: $0) }
        #expect(throws: Native.NativeError.self) { try kv.rollback(to: 0, commandBuffer: ctx.queue.makeCommandBuffer()!) }
        try Self.sync(ctx.queue) {
            try kv.attachPrefix(Native.PrefixMatch(handle: handle, length: 4, tier: .ram), commandBuffer: $0)
        }
        refused()
    }

    /// Consecutive steps use different page-table ring slots.
    @Test func ringSlotsAlternate() throws {
        let ctx = try MetalContext()
        let kv = try Self.store(device: ctx.device)
        try kv.beginStep(rows: 2, kind: .prefill)
        let a = kv.pageTable().pageTableOffset
        try kv.beginStep(rows: 1, kind: .decode)
        let b = kv.pageTable().pageTableOffset
        try kv.beginStep(rows: 1, kind: .decode)
        #expect(a != b && kv.pageTable().pageTableOffset == a)
    }

    static func syncResult<T>(_ queue: MTLCommandQueue, _ body: (MTLCommandBuffer) throws -> T) throws -> T {
        let cb = queue.makeCommandBuffer()!
        let v = try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        return v
    }

    /// Regression for the swap growth seen in the long golden runs: buffers
    /// from `makeBuffer` (and command buffers) are autoreleased, so a loop
    /// that does not drain a pool per iteration keeps every one alive (this
    /// loop then grows ~384 MiB per pass). With a pool per iteration, passes
    /// after the first must not grow.
    @Test func longEncodeLoopsDoNotRetainBuffers() throws {
        let ctx = try MetalContext()
        let glue = Native.ReferenceGlueOps(context: ctx)
        try glue.prepare(config: RefTest.tiny)
        let cols = 1 << 18  // 1 MiB of F32 per buffer
        func loop() throws {
            for _ in 0..<192 {
                // No pool of its own: buffers made inside RefTest.run's body
                // are drained by RefTest.run's pool (the production pattern).
                try RefTest.run(ctx) { enc in
                    let a = RefTest.tensor(ctx.device, floats: [Float](repeating: 1, count: cols), rows: 1, cols: cols)
                    let b = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: cols), rows: 1, cols: cols)
                    try glue.encodeCopy(enc, source: a, destination: b)
                }
            }
        }
        let start = RefTest.physicalFootprint()
        try loop()
        let warm = RefTest.physicalFootprint()
        try loop()
        try loop()
        let end = RefTest.physicalFootprint()
        let growth = Int64(end) - Int64(warm)
        print("footprint: start \(start >> 20) MiB, after one loop \(warm >> 20) MiB, after three \(end >> 20) MiB")
        // Each loop allocates ~384 MiB; retained buffers would add that twice.
        #expect(growth < 96 << 20, "footprint grew by \(growth >> 20) MiB after warm-up")
    }
}
