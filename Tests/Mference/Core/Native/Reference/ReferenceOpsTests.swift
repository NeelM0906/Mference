import Foundation
import Metal
import Testing
@testable import Mference

@Suite("ReferenceOpsTests", .serialized)
struct ReferenceOpsTests {
    @Test func glueOpsMatchCPUTwins() throws {
        let ctx = try MetalContext()
        let glue = Native.ReferenceGlueOps(context: ctx)
        try glue.prepare(config: RefTest.tiny)
        var rng = RefTest.SplitMix64(5)
        let rows = 5, cols = 256
        let a = (0..<rows * cols).map { _ in rng.unit() * 3 }
        let b = (0..<rows * cols).map { _ in rng.unit() * 3 }
        let w = (0..<cols).map { _ in rng.unit() }
        let ta = RefTest.tensor(ctx.device, floats: a, rows: rows, cols: cols, rowStride: cols * 4 + 32)
        let tb = RefTest.tensor(ctx.device, floats: b, rows: rows, cols: cols)
        let norm = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * cols), rows: rows, cols: cols)
        let sum = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * cols), rows: rows, cols: cols)
        let sw = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * cols), rows: rows, cols: cols)
        let copy = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * cols), rows: rows, cols: cols,
                                  rowStride: cols * 4 + 16, offset: 48)
        try RefTest.run(ctx) { enc in
            try glue.encodeRMSNorm(enc, input: ta, weight: RefTest.f32Weight(ctx.device, w), output: norm)
            try glue.encodeAdd(enc, a: ta, b: tb, output: sum)
            try glue.encodeSwiGLU(enc, gate: ta, up: tb, output: sw)
            try glue.encodeCopy(enc, source: ta, destination: copy)
        }
        let wantNorm = Native.ReferenceCPU.rmsNorm(a, rows: rows, cols: cols, weight: w, eps: 1e-6)
        #expect(RefTest.maxRelativeError(RefTest.floats(norm), wantNorm) < 1e-6)
        #expect(RefTest.floats(sum) == zip(a, b).map { $0 + $1 })
        #expect(RefTest.maxRelativeError(RefTest.floats(sw), Native.ReferenceCPU.swiglu(gate: a, up: b)) < 1e-6)
        #expect(RefTest.floats(copy) == a)
    }

    /// Attention over several steps (prefill chunk, decode, verify) through a
    /// reference KV store whose page table is never the identity, against the
    /// CPU twin, for F16 and INT8 pages.
    @Test(arguments: [Native.KVPrecision.f16, .int8])
    func attentionMatchesCPUTwinAcrossSteps(precision: Native.KVPrecision) throws {
        let ctx = try MetalContext()
        let c = RefTest.tiny
        let planner = Native.ReferenceMemoryPlanner()
        let plan = try planner.resolve(config: c, weightsMapped: 0, contextLength: 256, precision: precision,
                                       maxVerifyRows: 4, maxPrefillChunk: 80, activationBytes: 0,
                                       prefixCacheRAMBudget: 0, budget: 1 << 30)
        let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
        let op = Native.ReferenceAttentionOp(context: ctx)
        try op.prepare(config: c, geometry: kv.geometry)
        var rng = RefTest.SplitMix64(42)
        let qNorm = (0..<32).map { _ in 1 + rng.unit() * 0.5 }
        let kNorm = (0..<32).map { _ in 1 + rng.unit() * 0.5 }
        let weights = Native.AttentionLayerWeights(qNorm: RefTest.f32Weight(ctx.device, qNorm),
                                                   kNorm: RefTest.f32Weight(ctx.device, kNorm))
        let cpu = Native.ReferenceCPU.Attention(headCount: 4, kvHeadCount: 2, headDim: 32, ropeDims: 16, eps: 1e-6,
                                                freqBase: 10_000, precision: precision)
        // 70 rows of prefill crosses a page; then decode, then a verify of 3.
        for (kind, rows) in [(Native.StepKind.prefill, 70), (.decode, 1), (.verify, 3), (.prefill, 1)] {
            let start = kv.length
            try kv.beginStep(rows: rows, kind: kind)
            let qg = (0..<rows * 256).map { _ in rng.unit() * 2 }
            let k = (0..<rows * 64).map { _ in rng.unit() * 2 }
            let v = (0..<rows * 64).map { _ in rng.unit() * 2 }
            let tq = RefTest.tensor(ctx.device, floats: qg, rows: rows, cols: 256)
            let tk = RefTest.tensor(ctx.device, floats: k, rows: rows, cols: 64)
            let tv = RefTest.tensor(ctx.device, floats: v, rows: rows, cols: 64)
            let out = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * 128), rows: rows, cols: 128)
            let table = kv.pageTable()
            try RefTest.run(ctx) { enc in
                switch kind {
                case .decode:
                    try op.encodeDecode(enc, attnLayer: 1, weights: weights, qGate: tq, k: tk, v: tv,
                                        startPosition: start, kv: table, output: out)
                case .verify:
                    try op.encodeVerify(enc, attnLayer: 1, weights: weights, qGate: tq, k: tk, v: tv,
                                        startPosition: start, kv: table, output: out)
                case .prefill:
                    try op.encodePrefill(enc, attnLayer: 1, weights: weights, qGate: tq, k: tk, v: tv,
                                         startPosition: start, kv: table, output: out)
                }
            }
            let want = cpu.step(qGate: qg, k: k, v: v, rows: rows, qNorm: qNorm, kNorm: kNorm)
            let err = RefTest.maxRelativeError(RefTest.floats(out), want)
            // ulp-level differences between Metal's and libm's pow/cos/sin can
            // flip an F16 rounding of K (one F16 ulp is ~1e-3 relative).
            #expect(err < 2e-4, "\(precision) \(kind) rows \(rows): \(err)")
        }
        #expect(kv.physicalPages.count == 2)
        #expect(kv.physicalPages != Array(0..<kv.physicalPages.count))

        // Stored K for position 5 matches the twin's stored form (page bytes).
        let g = kv.geometry
        let table = kv.pageTable()
        let page = table.pageTable.contents().advanced(by: table.pageTableOffset)
            .assumingMemoryBound(to: UInt32.self)[0]
        let base = kv.arena.contents().advanced(by: Int(page) * g.pageBytes)
        for d in 0..<32 {
            let off = g.valueOffset(attnLayer: 1, kv: 0, kvHead: 1, token: 5, dim: d)
            let stored: Float
            if precision == .f16 {
                stored = Float(base.load(fromByteOffset: off, as: Float16.self))
            } else {
                let s = Float(base.load(fromByteOffset: g.scaleOffset(attnLayer: 1, kv: 0, kvHead: 1, token: 5),
                                        as: Float16.self))
                stored = s * Float(base.load(fromByteOffset: off, as: Int8.self))
            }
            #expect(abs(stored - cpu.keys[5][32 + d]) <= 1e-5 * max(1, abs(stored)))
        }
    }

    /// The ops read their small F32 weights only in the .gguf layout; a tiled
    /// one is refused instead of being read as row-major.
    @Test func gdnRefusesTiledSmallWeights() throws {
        let ctx = try MetalContext()
        let c = RefTest.tiny
        let plan = try Native.ReferenceMemoryPlanner().resolve(
            config: c, weightsMapped: 0, contextLength: 64, precision: .f16, maxVerifyRows: 1, maxPrefillChunk: 4,
            activationBytes: 0, prefixCacheRAMBudget: 0, budget: 1 << 30)
        let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
        let op = Native.ReferenceGDNOp(context: ctx)
        try op.prepare(config: c, layout: kv.recurrentLayout)
        let C = c.ssmConvChannels, K = c.ssmConvKernel, H = c.ssmTimeStepRank, d = c.ssmStateSize
        let conv = try RefTest.weight(ctx.device, name: "conv", type: .f32, rows: C, cols: K,
                                      layout: Native.TileLayout(rowsPerTile: 4, blocksPerTile: 2, planes: false),
                                      gguf: [UInt8](repeating: 0, count: C * K * 4))
        let ones = RefTest.f32Weight(ctx.device, [Float](repeating: 1, count: H))
        let weights = Native.GDNLayerWeights(conv1d: conv, dtBias: ones, a: ones,
                                             norm: RefTest.f32Weight(ctx.device, [Float](repeating: 1, count: d)))
        try kv.beginStep(rows: 1, kind: .decode)
        let banks = kv.recurrentState(gdnLayer: 0)
        func zeros(_ cols: Int) -> Native.TensorRef {
            RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: cols), rows: 1, cols: cols)
        }
        try RefTest.run(ctx) { enc in
            #expect(throws: Native.NativeError.self) {
                try op.encodeDecode(enc, weights: weights, qkv: zeros(C), z: zeros(H * d), alpha: zeros(H),
                                    beta: zeros(H), stateIn: banks.input, stateOut: banks.output, output: zeros(H * d))
            }
            // Overlapping input and output banks are refused (the op reads one
            // bank while it writes the other).
            let plain = Native.GDNLayerWeights(
                conv1d: Native.QuantTensor(name: "conv", type: .f32, shape: [K, C], layout: .gguf,
                                           buffer: RefTest.buffer(ctx.device, floats: [Float](repeating: 0, count: C * K)),
                                           offset: 0, byteCount: C * K * 4, sourceSHA256: ""),
                dtBias: ones, a: ones, norm: RefTest.f32Weight(ctx.device, [Float](repeating: 1, count: d)))
            let shifted = Native.RecurrentStateRef(layout: banks.input.layout, buffer: banks.input.buffer,
                                                   offset: banks.input.offset + 256)
            #expect(throws: Native.NativeError.self) {
                try op.encodeDecode(enc, weights: plain, qkv: zeros(C), z: zeros(H * d), alpha: zeros(H),
                                    beta: zeros(H), stateIn: banks.input, stateOut: shifted, output: zeros(H * d))
            }
        }
    }

    /// INT8 pages: an all-zero row stores scale 0 and every q 0; a row whose
    /// max|x| / 127 exceeds F16 stores the largest finite F16 scale, so loads
    /// stay finite (no inf scale, no NaN).
    @Test func int8ZeroAndOverflowRows() throws {
        let ctx = try MetalContext()
        let c = RefTest.tiny
        let plan = try Native.ReferenceMemoryPlanner().resolve(
            config: c, weightsMapped: 0, contextLength: 64, precision: .int8, maxVerifyRows: 1, maxPrefillChunk: 4,
            activationBytes: 0, prefixCacheRAMBudget: 0, budget: 1 << 30)
        let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
        let op = Native.ReferenceAttentionOp(context: ctx)
        try op.prepare(config: c, geometry: kv.geometry)
        let weights = Native.AttentionLayerWeights(qNorm: RefTest.f32Weight(ctx.device, [Float](repeating: 1, count: 32)),
                                                   kNorm: RefTest.f32Weight(ctx.device, [Float](repeating: 1, count: 32)))
        try kv.beginStep(rows: 2, kind: .prefill)
        // Row 0: zero K and V. Row 1: V of 1e7 (K is normed, so test V).
        let v = [Float](repeating: 0, count: 64) + [Float](repeating: 1e7, count: 64)
        let out = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: 2 * 128), rows: 2, cols: 128)
        try RefTest.run(ctx) {
            try op.encodePrefill($0, attnLayer: 0, weights: weights,
                                 qGate: RefTest.tensor(ctx.device, floats: [Float](repeating: 0.5, count: 2 * 256), rows: 2, cols: 256),
                                 k: RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: 2 * 64), rows: 2, cols: 64),
                                 v: RefTest.tensor(ctx.device, floats: v, rows: 2, cols: 64),
                                 startPosition: 0, kv: kv.pageTable(), output: out)
        }
        let g = kv.geometry
        let table = kv.pageTable()
        let page = Int(table.pageTable.contents().advanced(by: table.pageTableOffset).load(as: UInt32.self))
        let base = kv.arena.contents().advanced(by: page * g.pageBytes)
        for kvIndex in 0..<2 {
            #expect(base.load(fromByteOffset: g.scaleOffset(attnLayer: 0, kv: kvIndex, kvHead: 0, token: 0),
                              as: Float16.self) == 0)
            for dim in 0..<32 {
                #expect(base.load(fromByteOffset: g.valueOffset(attnLayer: 0, kv: kvIndex, kvHead: 0, token: 0, dim: dim),
                                  as: Int8.self) == 0)
            }
        }
        let big = base.load(fromByteOffset: g.scaleOffset(attnLayer: 0, kv: 1, kvHead: 0, token: 1), as: Float16.self)
        #expect(big == Float16.greatestFiniteMagnitude)
        #expect(RefTest.floats(out).allSatisfy { $0.isFinite })
    }

    /// GDN prefill, decode and verify (with arena slots) against the CPU twin,
    /// threading state through the double-banked store.
    @Test func gdnMatchesCPUTwinAcrossSteps() throws {
        let ctx = try MetalContext()
        let c = RefTest.tiny
        let plan = try Native.ReferenceMemoryPlanner().resolve(
            config: c, weightsMapped: 0, contextLength: 128, precision: .f16, maxVerifyRows: 4, maxPrefillChunk: 32,
            activationBytes: 0, prefixCacheRAMBudget: 0, budget: 1 << 30)
        let kv = try Native.ReferenceKVStore(config: c, plan: plan, device: ctx.device)
        let op = Native.ReferenceGDNOp(context: ctx)
        try op.prepare(config: c, layout: kv.recurrentLayout)
        let layout = kv.recurrentLayout
        var rng = RefTest.SplitMix64(7)
        let C = c.ssmConvChannels, K = c.ssmConvKernel, H = c.ssmTimeStepRank, d = c.ssmStateSize
        let conv1d = (0..<C * K).map { _ in rng.unit() * 0.5 }
        let dtBias = (0..<H).map { _ in rng.unit() }
        let a = (0..<H).map { _ in -abs(rng.unit()) - 0.1 }
        let norm = (0..<d).map { _ in 1 + rng.unit() * 0.3 }
        let convT = Native.QuantTensor(name: "conv", type: .f32, shape: [K, C], layout: .gguf,
                                       buffer: RefTest.buffer(ctx.device, floats: conv1d), offset: 0,
                                       byteCount: C * K * 4, sourceSHA256: "")
        let weights = Native.GDNLayerWeights(conv1d: convT, dtBias: RefTest.f32Weight(ctx.device, dtBias),
                                             a: RefTest.f32Weight(ctx.device, a),
                                             norm: RefTest.f32Weight(ctx.device, norm))
        let shape = Native.ReferenceCPU.GDNShape(convKernel: K, convChannels: C, valueHeads: H,
                                                 groups: c.ssmGroupCount, dim: d, eps: c.rmsNormEpsilon)
        var state = Native.ReferenceCPU.GDNState(conv: [Float](repeating: 0, count: (K - 1) * C),
                                                 ssm: [Float](repeating: 0, count: H * d * d))
        func read(_ ref: Native.RecurrentStateRef) -> Native.ReferenceCPU.GDNState {
            let p = ref.buffer.contents().advanced(by: ref.offset)
            let conv = Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: (K - 1) * C))
            let ssm = Array(UnsafeBufferPointer(start: p.advanced(by: layout.ssmOffset).assumingMemoryBound(to: Float.self),
                                                count: H * d * d))
            return .init(conv: conv, ssm: ssm)
        }
        func close(_ x: Native.ReferenceCPU.GDNState, _ y: Native.ReferenceCPU.GDNState) -> Bool {
            RefTest.maxRelativeError(x.conv, y.conv) < 1e-5 && RefTest.maxRelativeError(x.ssm, y.ssm) < 1e-4
        }
        for (kind, rows) in [(Native.StepKind.prefill, 9), (.decode, 1), (.verify, 4), (.prefill, 2)] {
            try kv.beginStep(rows: rows, kind: kind)
            let qkv = (0..<rows * C).map { _ in rng.unit() * 2 }
            let z = (0..<rows * H * d).map { _ in rng.unit() * 2 }
            let alpha = (0..<rows * H).map { _ in rng.unit() * 2 }
            let beta = (0..<rows * H).map { _ in rng.unit() * 2 }
            let out = RefTest.tensor(ctx.device, floats: [Float](repeating: 0, count: rows * H * d), rows: rows,
                                     cols: H * d)
            let banks = kv.recurrentState(gdnLayer: 1)
            let arena = kv.verifyArena(gdnLayer: 1)
            let tq = RefTest.tensor(ctx.device, floats: qkv, rows: rows, cols: C)
            let tz = RefTest.tensor(ctx.device, floats: z, rows: rows, cols: H * d)
            let ta = RefTest.tensor(ctx.device, floats: alpha, rows: rows, cols: H)
            let tb = RefTest.tensor(ctx.device, floats: beta, rows: rows, cols: H)
            try RefTest.run(ctx) { enc in
                switch kind {
                case .decode:
                    try op.encodeDecode(enc, weights: weights, qkv: tq, z: tz, alpha: ta, beta: tb,
                                        stateIn: banks.input, stateOut: banks.output, output: out)
                case .verify:
                    try op.encodeVerify(enc, weights: weights, qkv: tq, z: tz, alpha: ta, beta: tb,
                                        stateIn: banks.input, stateOut: banks.output, arena: arena, output: out)
                case .prefill:
                    try op.encodePrefill(enc, weights: weights, qkv: tq, z: tz, alpha: ta, beta: tb,
                                         stateIn: banks.input, stateOut: banks.output, output: out)
                }
            }
            let want = Native.ReferenceCPU.gdn(shape: shape, qkv: qkv, z: z, alpha: alpha, beta: beta, rows: rows,
                                               conv1d: conv1d, dtBias: dtBias, a: a, norm: norm, state: state)
            let err = RefTest.maxRelativeError(RefTest.floats(out), want.output)
            #expect(err < 1e-4, "\(kind) rows \(rows): \(err)")
            #expect(close(read(banks.output), want.states.last!), "\(kind) state")
            if kind == .verify {
                for r in 0..<rows {
                    let slot = Native.RecurrentStateRef(layout: layout, buffer: arena.buffer,
                                                        offset: arena.offset + r * arena.slotStride)
                    #expect(close(read(slot), want.states[r]), "verify slot \(r)")
                }
            }
            state = want.states.last!
        }
    }
}
