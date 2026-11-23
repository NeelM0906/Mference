import Foundation
import Metal

/// Where each layer's routed experts come from. Prefill visits are begun and
/// ended in visit order around the encoder that reads the pool (a staged
/// layer waits there for its slab and releases it after); decode visits read
/// `decodePool`. K2's `Inkling2ExpertStore` (resident and staged layers) for
/// an install, `Inkling2FixedExperts` for buffers already in memory.
protocol Inkling2ExpertSource: AnyObject {
    var shape: Inkling2MoEShape { get }
    func beginPrefillVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws -> Inkling2ExpertPool
    func endPrefillVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws
    func decodePool(layer: Int) -> Inkling2ExpertPool
    /// Releases visits begun and not ended (their command buffer is lost).
    func abandonOpenVisits() throws
    /// Set when staging failed; the source is then unusable.
    var stagingFailure: String? { get }
}

extension Inkling2ExpertStore: Inkling2ExpertSource {
    var stagingFailure: String? { staging?.error.map { "\($0)" } }
}

/// One pool per stored layer, all in memory already (a synthetic model).
final class Inkling2FixedExperts: Inkling2ExpertSource {
    let pools: [Inkling2ExpertPool]
    let shape: Inkling2MoEShape

    init(pools: [Inkling2ExpertPool]) throws {
        guard let first = pools.first, pools.allSatisfy({ $0.shape == first.shape }) else {
            throw Inkling2ForwardRunnerError.invalidInput("Inkling 2 experts: no pools, or pools of differing shapes")
        }
        self.pools = pools
        self.shape = first.shape
    }

    func beginPrefillVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws -> Inkling2ExpertPool { pools[layer] }
    func endPrefillVisit(layer: Int, commandBuffer: MTLCommandBuffer) throws {}
    func decodePool(layer: Int) -> Inkling2ExpertPool { pools[layer] }
    func abandonOpenVisits() throws {}
    var stagingFailure: String? { nil }
}

/// Every tensor the Inkling 2 runner reads, as buffer views. Built from an
/// install (`Model`'s `inkling2*` accessors) or, in tests, from synthetic
/// buffers. Dense Q8_0 and BF16 weights are byte offsets into `dense`, the
/// buffer K3's stacked projection reads; F32 tensors are buffer references.
struct Inkling2RunnerWeights {
    struct Loop {
        /// `loop_msg_query` F32 `[4][d]`: query, key, value, MoE.
        var messageQueries: Inkling2BufferRef
        /// Per controller (query, key, value, MoE): `down` F32 `[rank][2d]`,
        /// `gain` and `forget` F32 `[d][rank]`.
        var down: [Inkling2BufferRef]
        var gain: [Inkling2BufferRef]
        var forget: [Inkling2BufferRef]
    }

    struct Layer {
        var isGlobal: Bool
        /// Sliding layers: `attn_norm`. Global layers: q, k, v input norms.
        var attnNorm: Inkling2BufferRef?
        var inputNorms: [Inkling2BufferRef]
        // Q8_0 byte offsets in `dense`.
        var q: Int, k: Int, v: Int, rel: Int, gate: Int, output: Int
        // BF16 byte offsets in `dense`.
        var sinkProj: Int
        var kConvProj: Int?
        var vConvProj: Int?
        // F32.
        var qClip: Inkling2BufferRef
        var relClip: Inkling2BufferRef
        var relBias: Inkling2BufferRef
        var kConv: Inkling2BufferRef
        var vConv: Inkling2BufferRef
        var attnPostConv: Inkling2BufferRef
        var ffnPostConv: Inkling2BufferRef
        /// `[1][d]` on sliding layers, `[3][d]` on global layers, nil on layer 0.
        var attnResQuery: Inkling2BufferRef?
        var ffnResQuery: Inkling2BufferRef
        var ffnNorm: Inkling2BufferRef
        var router: Inkling2BufferRef
        var routerBias: Inkling2BufferRef
        /// Core layers only.
        var loop: Loop?
    }

    var dense: MTLBuffer
    /// Q8_0 `[vocabRows][d]` in `dense`.
    var embedding: Int
    var embeddingNorm: Inkling2BufferRef
    var outputResQuery: Inkling2BufferRef
    var outputNorm: Inkling2BufferRef
    /// Q8_0 `[vocabRows][d]` in `dense`.
    var output: Int
    var layers: [Layer]
    /// The routed experts of every layer.
    var experts: any Inkling2ExpertSource
    /// Rows of the embedding and head, and logits the head produces.
    var vocabRows: Int
    var vocab: Int
    var gateRank: Int
    var routeScale: Float
    var finalLogitSoftcap: Float
    var rmsEps: Float
}

enum Inkling2ForwardRunnerError: Error, CustomStringConvertible {
    case invalidInput(String)
    case commandFailed(String)
    case allocationFailed(String)

    var description: String {
        switch self {
        case .invalidInput(let s), .commandFailed(let s), .allocationFailed(let s): return s
        }
    }
}

/// The native Inkling 2 Mini runner: per token, the embedding, the prelude
/// (stored layers 0 to 5), the core (6 to 17) three times, the coda (18 to
/// 23), the final depth mix, norm, head and softcap. Each of the 48 visits
/// has its own KV and convolution state (`Inkling2AttentionState`, indexed by
/// cache slot). Prefill is layer-major over chunks: one visit over every
/// token of the chunk, then the next; decode is a chunk of one.
///
/// Per visit: depth mix of the completed blocks and the running sum (K4),
/// message gates on passes 2 and 3 (K4), input norms, stacked projections,
/// the Q/K/V epilogue with the short convolutions, attention with the
/// relative bias and sink, output projection and post-attention convolution
/// (K3), then the MoE input mix and gate (K4), the router and routed experts
/// (K2) and the post-MoE convolution (K3). Checked against
/// `Inkling2ReferenceModel`.
final class Inkling2ForwardRunner: ContinuableLogitProducer, ChunkedPrefillRunner,
    ContextWindowReporting, @unchecked Sendable {

    static let d = Inkling2AttentionShape.hidden
    static let qWidth = Inkling2AttentionShape.qWidth
    static let kvWidth = Inkling2AttentionShape.kvWidth
    static let relWidth = Inkling2AttentionShape.relWidth
    static let heads = Inkling2AttentionShape.numQHeads
    static let blockSize = 6
    static let coreStart = 6
    static let coreEnd = 18
    static let passes = 3

    // Columns of the stacked projection output Y (FP32 per token).
    static let colQ = 0
    static let colRel = qWidth                          // 3072
    static let colSink = colRel + relWidth              // 3456
    static let colGate = colSink + heads                // 3480
    static let colK = colGate + qWidth                  // 6552
    static let colV = colK + kvWidth                    // 7576
    static let colKTaps = colV + kvWidth                // 8600
    static let colVTaps = colKTaps + 32                 // 8632
    static let yStride = colVTaps + 32                  // 8664

    let maxContext: Int
    let maxChunkTokens: Int
    private let w: Inkling2RunnerWeights
    private let ctx: MetalContext
    private let inputNorm: Inkling2InputNorm
    private let projection: Inkling2Projection
    private let epilogue: Inkling2QKVEpilogue
    private let convTail: Inkling2ConvTail
    private let shortConv: Inkling2ShortConv
    private let prefillAttention: Inkling2PrefillAttention
    private let decodeAttention: Inkling2DecodeAttention
    private let moe: Inkling2MoEKernels
    private let kernels: Inkling2RunnerKernels
    private let state: Inkling2AttentionState
    private let moeShape: Inkling2MoEShape
    private let scratch: Scratch
    private var chunkState = PrefillChunkCommitState()

    /// Diagnostic and golden hook: while set, every chunk copies each
    /// visit's output, the running block sum (FP32 `[48][rows][d]`), and the
    /// experts each token chose (UInt32 `[48][rows][topK]`) into these
    /// buffers, visits in execution order, a token at row `position`
    /// (positions `0 ..< rows`). Off on every production path.
    struct VisitTrace {
        let outputs: MTLBuffer
        let routes: MTLBuffer
        let rows: Int
    }
    var visitTrace: VisitTrace?

    init(weights: Inkling2RunnerWeights, context: MetalContext, maxContext: Int, maxChunkTokens: Int) throws {
        try Self.validate(weights)
        guard maxContext > 0, maxChunkTokens > 0 else {
            throw Inkling2ForwardRunnerError.invalidInput(
                "Inkling 2 runner needs a positive context (\(maxContext)) and chunk (\(maxChunkTokens))")
        }
        self.w = weights
        self.ctx = context
        self.maxContext = maxContext
        self.maxChunkTokens = maxChunkTokens
        let library = try Inkling2KernelLibrary(context: context)
        inputNorm = try Inkling2InputNorm(library: library)
        projection = try Inkling2Projection(library: library)
        epilogue = try Inkling2QKVEpilogue(library: library)
        convTail = try Inkling2ConvTail(library: library)
        shortConv = try Inkling2ShortConv(library: library)
        prefillAttention = try Inkling2PrefillAttention(library: library)
        decodeAttention = try Inkling2DecodeAttention(library: library)
        moe = try Inkling2MoEKernels(context: context)
        kernels = try Inkling2RunnerKernels(device: context.device)
        state = try Inkling2AttentionState(device: context.device,
                                           configuration: .init(maxContext: maxContext,
                                                                maxChunkTokens: maxChunkTokens))
        moeShape = weights.experts.shape
        scratch = try Scratch(device: context.device, tokens: maxChunkTokens, maxContext: maxContext,
                              moeShape: moeShape)
    }

    /// What the schedule and the kernels assume of the weights, checked
    /// once: the layer kinds the visit plan (and K3's state) use, the
    /// tensors each kind and the core need, the vocabulary, the gate rank
    /// the message-gate kernel holds, and the experts' hidden width.
    static func validate(_ w: Inkling2RunnerWeights) throws {
        func invalid(_ detail: String) -> Inkling2ForwardRunnerError { .invalidInput("Inkling 2 weights: " + detail) }
        guard w.layers.count == Inkling2VisitPlan.storedLayers else {
            throw invalid("\(w.layers.count) layers, not \(Inkling2VisitPlan.storedLayers)")
        }
        guard w.vocab > 0, w.vocabRows >= w.vocab else { throw invalid("vocabulary \(w.vocab) of \(w.vocabRows) rows") }
        guard (1...Inkling2RunnerKernels.maxGateRank).contains(w.gateRank) else {
            throw invalid("gate rank \(w.gateRank)")
        }
        guard w.experts.shape.d == d else { throw invalid("expert hidden \(w.experts.shape.d), hidden \(d)") }
        for (layer, lw) in w.layers.enumerated() {
            guard lw.isGlobal == Inkling2VisitPlan.isGlobal(layer: layer) else {
                throw invalid("layer \(layer) is \(lw.isGlobal ? "global" : "sliding"), the visit plan says otherwise")
            }
            if lw.isGlobal {
                guard lw.inputNorms.count == 3, lw.kConvProj != nil, lw.vConvProj != nil else {
                    throw invalid("global layer \(layer) needs q, k, v input norms and both tap projections")
                }
            } else {
                guard lw.attnNorm != nil else { throw invalid("sliding layer \(layer) has no attn_norm") }
            }
            guard (lw.attnResQuery == nil) == (layer == 0) else {
                throw invalid("layer \(layer): attn_res_query is absent exactly on layer 0")
            }
            guard (lw.loop != nil) == Inkling2VisitPlan.isLoopLayer(layer) else {
                throw invalid("layer \(layer): loop gates are present exactly on the core layers")
            }
            if let loop = lw.loop, loop.down.count != 4 || loop.gain.count != 4 || loop.forget.count != 4 {
                throw invalid("core layer \(layer) needs four message gates")
            }
        }
    }

    // MARK: Scratch

    /// Every buffer of one chunk, sized for `tokens` rows. The arena holds
    /// the block states (FP32 `[tokens][d]` per slot): a slot keeps its value
    /// while any block list, message bank or the prelude snapshot names it.
    final class Scratch {
        static let arenaSlots = 12
        let tokens: Int
        let arena: MTLBuffer
        let mix: MTLBuffer          // FP32 [3][T][d]
        let messages: MTLBuffer     // FP32 [4][T][d]
        let gated: MTLBuffer        // FP32 [3][T][d]
        let moeIn: MTLBuffer        // FP32 [T][d]
        let moeInHalf: MTLBuffer    // FP16 [T][d]
        let ffnMix: MTLBuffer       // FP32 [2][T][d]
        let h: MTLBuffer            // FP16 [3][T][d] normed q/k/v inputs
        let y: MTLBuffer            // FP32 [T][yStride]
        let q: MTLBuffer            // FP16 [T][qWidth]
        let rel: MTLBuffer          // FP32 [T][relWidth]
        let attn: MTLBuffer         // FP16 [T][qWidth]
        let o: MTLBuffer            // FP32 [T][d]
        let moeOut: MTLBuffer       // FP32 [T][d]
        let ids: MTLBuffer          // UInt32 [T]
        let finalRow: MTLBuffer     // FP32 [2][d]
        let partials: MTLBuffer
        let decodeActivation: MTLBuffer
        let moeScratch: Inkling2MoEPrefillScratch

        init(device: MTLDevice, tokens: Int, maxContext: Int, moeShape: Inkling2MoEShape) throws {
            self.tokens = tokens
            let d = Inkling2ForwardRunner.d
            func make(_ bytes: Int, _ name: String) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: max(16, bytes), options: .storageModePrivate) else {
                    throw Inkling2ForwardRunnerError.allocationFailed("inkling2.runner.\(name)")
                }
                b.label = "inkling2.runner.\(name)"
                return b
            }
            arena = try make(Self.arenaSlots * tokens * d * 4, "arena")
            mix = try make(3 * tokens * d * 4, "mix")
            messages = try make(4 * tokens * d * 4, "messages")
            gated = try make(3 * tokens * d * 4, "gated")
            moeIn = try make(tokens * d * 4, "moe-in")
            moeInHalf = try make(tokens * d * 2, "moe-in-half")
            ffnMix = try make(2 * tokens * d * 4, "ffn-mix")
            h = try make(3 * tokens * d * 2, "inputs")
            y = try make(tokens * Inkling2ForwardRunner.yStride * 4, "projections")
            q = try make(tokens * Inkling2ForwardRunner.qWidth * 2, "queries")
            rel = try make(tokens * Inkling2ForwardRunner.relWidth * 4, "relative")
            attn = try make(tokens * Inkling2ForwardRunner.qWidth * 2, "attention")
            o = try make(tokens * d * 4, "attention-out")
            moeOut = try make(tokens * d * 4, "moe-out")
            guard let idsBuffer = device.makeBuffer(length: max(16, tokens * 4), options: .storageModeShared) else {
                throw Inkling2ForwardRunnerError.allocationFailed("inkling2.runner.ids")
            }
            ids = idsBuffer
            finalRow = try make(2 * d * 4, "final")
            partials = try make(Inkling2DecodeAttention.partialsLength(keys: maxContext), "partials")
            decodeActivation = try make(moeShape.topK * moeShape.f * 2, "decode-activation")
            moeScratch = try Inkling2MoEPrefillScratch(device: device, shape: moeShape, maxTokens: tokens)
        }

        func arenaRef(_ slot: Int) -> Inkling2BufferRef {
            .init(arena, offset: slot * tokens * Inkling2ForwardRunner.d * 4)
        }
        /// Float offset of a slot's row 0, for the depth-mix kernel.
        func arenaFloats(_ slot: Int) -> Int { slot * tokens * Inkling2ForwardRunner.d }
    }

    /// Reference-counted arena slots.
    private struct Slots {
        var refs = [Int](repeating: 0, count: Scratch.arenaSlots)
        mutating func take() -> Int {
            guard let i = refs.firstIndex(of: 0) else { fatalError("Inkling 2 arena exhausted") }
            refs[i] = 1
            return i
        }
        mutating func retain(_ i: Int) { refs[i] += 1 }
        mutating func release(_ i: Int) { precondition(refs[i] > 0); refs[i] -= 1 }
        mutating func release(_ list: [Int]) { list.forEach { release($0) } }
    }

    // MARK: LogitProducer

    func reset() {
        chunkState.reset()
        state.reset()
    }

    var continuationPosition: Int { state.position }

    func prepareForContinuation(expectedPosition: Int) throws {
        try chunkState.requireClean(operation: "prepareForContinuation")
        guard expectedPosition > 0, expectedPosition == state.position else {
            throw PrefillError.prefillCursorMismatch(
                "Inkling 2 continuation cursor \(expectedPosition) does not match \(state.position)")
        }
    }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try chunkState.requireClean(operation: "produce")
        try Task.checkCancellation()
        guard position == state.position else {
            throw Inkling2ForwardRunnerError.invalidInput(
                "Inkling 2 position \(position) does not match its cursor \(state.position)")
        }
        try run(tokens: [token], start: position, logits: logits)
    }

    func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig, into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        var execution = PrefillExecutionReport()
        try chunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported("Inkling 2 prefillChunked requires chunked mode")
        }
        guard startPosition == state.position else {
            throw PrefillError.chunkedUnsupported(
                "Inkling 2 prefill cursor \(state.position) != startPosition \(startPosition)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten, execution: execution)
        }
        // Check the whole range before the first span commits anything.
        guard startPosition + tokens.count <= maxContext else {
            throw Inkling2ForwardRunnerError.invalidInput(
                "Inkling 2 prompt of \(tokens.count) at \(startPosition) exceeds the context \(maxContext)")
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < w.vocab }) else {
            throw Inkling2ForwardRunnerError.invalidInput("Inkling 2 token outside the vocabulary")
        }
        guard logits.length >= w.vocabRows * 2 else {
            throw Inkling2ForwardRunnerError.invalidInput("Inkling 2 logits buffer is too small")
        }
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count, startPosition: startPosition,
                                              chunkTokens: min(config.chunkTokens, maxChunkTokens))
        for (i, span) in spans.enumerated() {
            try Task.checkCancellation()
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try run(tokens: Array(tokens[lower..<upper]), start: span.startPosition,
                    logits: i == spans.count - 1 ? logits : nil)
            execution.recordBatch(span.tokenCount)
            onProgress(span.completedCount)
        }
        return PrefillResult(newPosition: startPosition + tokens.count, seed: .logitsWritten,
                             execution: execution)
    }

    // MARK: One chunk

    /// Runs `tokens` at `start ..< start + count` through all 48 visits and,
    /// when `logits` is given, writes the last token's FP16 logits (the
    /// padded rows past `vocab` get a large negative value). The chunk goes
    /// out as several command buffers; the state advances only when every
    /// one completed, and a failure leaves the chunk state dirty (`reset`).
    private func run(tokens: [Int32], start: Int, logits: MTLBuffer?) throws {
        let t = tokens.count
        guard t > 0, t <= maxChunkTokens, start + t <= maxContext else {
            throw Inkling2ForwardRunnerError.invalidInput("Inkling 2 chunk of \(t) at \(start) out of range")
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < w.vocab }) else {
            throw Inkling2ForwardRunnerError.invalidInput("Inkling 2 token outside the vocabulary")
        }
        if let logits, logits.length < w.vocabRows * 2 {
            throw Inkling2ForwardRunnerError.invalidInput("Inkling 2 logits buffer is too small")
        }
        try state.ensureCapacity(tokens: start + t)
        chunkState.markDirty(startPosition: start, tokenCount: t)

        let s = scratch
        let d = Self.d
        tokens.withUnsafeBytes { raw in
            let ids = s.ids.contents().bindMemory(to: UInt32.self, capacity: t)
            for i in 0..<t { ids[i] = UInt32(bitPattern: raw.load(fromByteOffset: i * 4, as: Int32.self)) }
        }

        var slots = Slots()
        var cb = try commandBuffer()
        var committed: [MTLCommandBuffer] = []
        var pending: MTLCommandBuffer? = cb
        /// Commits the buffer being encoded and starts the next.
        func nextBuffer() throws {
            cb.commit()
            committed.append(cb)
            pending = nil
            cb = try commandBuffer()
            pending = cb
        }
        /// Commits what is encoded and waits for every buffer. Encoded work
        /// is never dropped: a staged expert slab's release is in it.
        func finish() {
            if let open = pending {
                open.commit()
                committed.append(open)
                pending = nil
            }
            committed.forEach { $0.waitUntilCompleted() }
        }
        func encode(_ body: (MTLComputeCommandEncoder) -> Void) {
            let enc = cb.makeComputeCommandEncoder()!
            body(enc)
            enc.endEncoding()
        }
        let topK = moeShape.topK
        if let trace = visitTrace {
            precondition(start + t <= trace.rows && trace.outputs.length >= 48 * trace.rows * d * 4
                         && trace.routes.length >= 48 * trace.rows * topK * 4, "visit trace too small")
        }
        var visitIndex = 0
        func record(_ output: Int) {
            defer { visitIndex += 1 }
            guard let trace = visitTrace, let blit = cb.makeBlitCommandEncoder() else { return }
            let row = visitIndex * trace.rows + start
            blit.copy(from: s.arena, sourceOffset: s.arenaRef(output).offset, to: trace.outputs,
                      destinationOffset: row * d * 4, size: t * d * 4)
            blit.copy(from: s.moeScratch.ids, sourceOffset: 0, to: trace.routes,
                      destinationOffset: row * topK * 4, size: t * topK * 4)
            blit.endEncoding()
        }

        do {
            // Embedding.
            let embedded = slots.take()
            encode { enc in
                kernels.encodeEmbedding(enc, ids: s.ids, table: w.dense, tableOffset: w.embedding,
                                        rowBytes: d / 32 * 34, weight: w.embeddingNorm.buffer,
                                        weightOffset: w.embeddingNorm.offset, eps: w.rmsEps,
                                        out: s.arena, outOffset: s.arenaRef(embedded).offset,
                                        outStride: d, d: d, tokens: t)
            }

            var cur = embedded
            var blocks: [Int] = []
            // Prelude.
            for layer in 0..<Self.coreStart {
                cur = try encodeVisit(cb, layer: layer, slot: layer, tokens: t, start: start,
                                      cur: cur, blocks: &blocks, bank: nil, slots: &slots)
                record(cur)
            }
            let preludeCur = cur
            let preludeBlocks = blocks
            slots.retain(preludeCur)
            preludeBlocks.forEach { slots.retain($0) }

            var bank: [Int]? = nil
            for pass in 0..<Self.passes {
                if pass > 0 {
                    // The previous pass's bank: its first core block and its output.
                    let newBank = Array(blocks[(preludeBlocks.count + 1)...]) + [cur]
                    newBank.forEach { slots.retain($0) }
                    if let old = bank { slots.release(old) }
                    bank = newBank
                    slots.release(blocks)
                    slots.release(cur)
                    cur = preludeCur
                    blocks = preludeBlocks
                    slots.retain(cur)
                    blocks.forEach { slots.retain($0) }
                }
                for layer in Self.coreStart..<Self.coreEnd {
                    let slot = Inkling2VisitPlan.slot(layer: layer, pass: pass)
                    cur = try encodeVisit(cb, layer: layer, slot: slot, tokens: t, start: start,
                                          cur: cur, blocks: &blocks, bank: bank, slots: &slots)
                    record(cur)
                }
                // Keep each command buffer to about a third of the visits.
                try nextBuffer()
            }
            if let old = bank { slots.release(old) }
            slots.release(preludeCur)
            slots.release(preludeBlocks)
            // Coda.
            for layer in Self.coreEnd..<24 {
                cur = try encodeVisit(cb, layer: layer, slot: layer, tokens: t, start: start,
                                      cur: cur, blocks: &blocks, bank: nil, slots: &slots)
                record(cur)
            }

            if let logits {
                // The head reads only the last token: mix its block states, norm, project.
                let last = t - 1
                encode { enc in
                    kernels.encodeDepthMix(enc, arena: s.arena,
                                           stateOffsets: (blocks + [cur]).map { s.arenaFloats($0) + last * d },
                                           stateStride: d, queries: w.outputResQuery.buffer,
                                           queriesOffset: w.outputResQuery.offset, queryCount: 1,
                                           out: s.finalRow, outStride: d, outQueryStride: d, d: d, tokens: 1)
                    kernels.encodeRMSNorm(enc, x: s.finalRow, xStride: d, weight: w.outputNorm.buffer,
                                          weightOffset: w.outputNorm.offset, eps: w.rmsEps,
                                          out: s.finalRow, outOffset: d * 4, outStride: d, d: d, tokens: 1)
                    kernels.encodeHead(enc, x: s.finalRow, xOffset: d * 4, table: w.dense, tableOffset: w.output,
                                       rowBytes: d / 32 * 34, rows: w.vocab, softcap: w.finalLogitSoftcap,
                                       logits: logits, d: d)
                }
                if w.vocabRows > w.vocab, let blit = cb.makeBlitCommandEncoder() {
                    // 0xFBFB is about -65376 in FP16: never sampled, never the argmax.
                    blit.fill(buffer: logits, range: (w.vocab * 2)..<(w.vocabRows * 2), value: 0xFB)
                    blit.endEncoding()
                }
            }
        } catch {
            // A visit begun and not ended would hold its slab forever.
            try? w.experts.abandonOpenVisits()
            finish()
            throw error
        }
        finish()
        if let failure = w.experts.stagingFailure {
            throw Inkling2ForwardRunnerError.commandFailed("Inkling 2 expert staging failed: \(failure)")
        }
        if let failed = committed.first(where: { $0.status != .completed }) {
            throw Inkling2ForwardRunnerError.commandFailed(
                failed.error?.localizedDescription ?? "Inkling 2 command buffer did not complete")
        }
        state.advance(by: t)
        chunkState.markCommitted()
    }

    // MARK: One visit

    /// Encodes one layer visit over the chunk and returns the arena slot of
    /// the new running sum. `blocks` gains the visit's input at a block start.
    private func encodeVisit(_ cb: MTLCommandBuffer, layer: Int, slot: Int, tokens t: Int, start: Int,
                             cur: Int, blocks: inout [Int], bank: [Int]?, slots: inout Slots) throws -> Int {
        let s = scratch
        let d = Self.d
        let lw = w.layers[layer]
        let plane = t * d                       // floats in one [T][d] plane
        func encode(_ body: (MTLComputeCommandEncoder) -> Void) {
            let enc = cb.makeComputeCommandEncoder()!
            body(enc)
            enc.endEncoding()
        }

        // 1. Attention inputs: the depth mix (layer 0 reads the embedding),
        //    then the q/k/v message gates on later passes.
        var inputs: [Inkling2BufferRef]      // FP32 [T][d] each, q k v
        if let queries = lw.attnResQuery {
            let count = lw.isGlobal ? 3 : 1
            encode { enc in
                kernels.encodeDepthMix(enc, arena: s.arena,
                                       stateOffsets: (blocks + [cur]).map { s.arenaFloats($0) },
                                       stateStride: d, queries: queries.buffer, queriesOffset: queries.offset,
                                       queryCount: count, out: s.mix, outStride: d, outQueryStride: plane,
                                       d: d, tokens: t)
            }
            let mixed = (0..<count).map { Inkling2BufferRef(s.mix, offset: $0 * plane * 4) }
            inputs = count == 3 ? mixed : [mixed[0], mixed[0], mixed[0]]
        } else {
            let x = s.arenaRef(cur)
            inputs = [x, x, x]
        }
        if let bank, let loop = lw.loop {
            encode { enc in
                kernels.encodeDepthMix(enc, arena: s.arena, stateOffsets: bank.map { s.arenaFloats($0) },
                                       stateStride: d, queries: loop.messageQueries.buffer,
                                       queriesOffset: loop.messageQueries.offset, queryCount: 4,
                                       out: s.messages, outStride: d, outQueryStride: plane, d: d, tokens: t)
                for j in 0..<3 {
                    kernels.encodeMessageGate(enc, x: inputs[j].buffer, xOffset: inputs[j].offset, xStride: d,
                                              message: s.messages, messageOffset: j * plane * 4, messageStride: d,
                                              down: loop.down[j].buffer, downOffset: loop.down[j].offset,
                                              gain: loop.gain[j].buffer, gainOffset: loop.gain[j].offset,
                                              forget: loop.forget[j].buffer, forgetOffset: loop.forget[j].offset,
                                              rank: w.gateRank, out: s.gated, outOffset: j * plane * 4,
                                              outStride: d, d: d, tokens: t)
                }
            }
            inputs = (0..<3).map { Inkling2BufferRef(s.gated, offset: $0 * plane * 4) }
        }

        // 2. Input norms to FP16. Without messages a sliding layer has one input.
        let distinct = lw.isGlobal || bank != nil
        let h = (0..<3).map { Inkling2BufferRef(s.h, offset: $0 * plane * 2) }
        for j in 0..<(distinct ? 3 : 1) {
            let weight = lw.isGlobal ? lw.inputNorms[j] : lw.attnNorm!
            inputNorm.encode(commandBuffer: cb, x: inputs[j], xStride: d, weight: weight,
                             output: h[j], outputStride: d, rows: t, eps: w.rmsEps)
        }
        // Projection segments address the FP16 inputs in halves.
        let hq = 0, hk = distinct ? plane : 0, hv = distinct ? 2 * plane : 0

        // 3. Stacked projections into Y.
        var segments: [Inkling2ProjectionSegment] = [
            .init(weightOffset: lw.q, inputOffset: hq, inputStride: d, rows: Self.qWidth,
                  outColumn: Self.colQ, format: .q8_0),
            .init(weightOffset: lw.rel, inputOffset: hq, inputStride: d, rows: Self.relWidth,
                  outColumn: Self.colRel, format: .q8_0),
            .init(weightOffset: lw.sinkProj, inputOffset: hq, inputStride: d, rows: Self.heads,
                  outColumn: Self.colSink, format: .bf16),
            .init(weightOffset: lw.gate, inputOffset: hq, inputStride: d, rows: Self.qWidth,
                  outColumn: Self.colGate, format: .q8_0),
            .init(weightOffset: lw.k, inputOffset: hk, inputStride: d, rows: Self.kvWidth,
                  outColumn: Self.colK, format: .q8_0),
            .init(weightOffset: lw.v, inputOffset: hv, inputStride: d, rows: Self.kvWidth,
                  outColumn: Self.colV, format: .q8_0),
        ]
        if lw.isGlobal {
            segments.append(.init(weightOffset: lw.kConvProj!, inputOffset: hk, inputStride: d,
                                  rows: 32, outColumn: Self.colKTaps, format: .bf16))
            segments.append(.init(weightOffset: lw.vConvProj!, inputOffset: hv, inputStride: d,
                                  rows: 32, outColumn: Self.colVTaps, format: .bf16))
        }
        projection.encode(commandBuffer: cb, weights: w.dense, input: s.h, output: s.y,
                          tokens: t, inputs: d, outStride: Self.yStride, segments: segments)

        // 4. Clips, K/V convolutions, V norm, cache store; then the K/V tails.
        let kv = state.kv(slot: slot)
        let tails = state.tails(slot: slot)
        epilogue.encode(commandBuffer: cb, .init(
            projections: .init(s.y), projectionStride: Self.yStride,
            columns: .init(queries: Self.colQ, relative: Self.colRel, keys: Self.colK, values: Self.colV,
                           keyTaps: lw.isGlobal ? Self.colKTaps : nil,
                           valueTaps: lw.isGlobal ? Self.colVTaps : nil),
            queryScale: lw.qClip, relativeScale: lw.relClip,
            keyConv: lw.kConv, valueConv: lw.vConv,
            keyTail: tails.keys, valueTail: tails.values,
            queries: .init(s.q), queryStride: Self.qWidth,
            relative: .init(s.rel), relativeStride: Self.relWidth,
            cache: kv, rows: t, startPosition: start))
        convTail.encode(commandBuffer: cb, x: .init(s.y), xStride: Self.yStride, rows: t, tails: tails.keys,
                        runs: tails.keyValueRuns(keyColumn: Self.colK, valueColumn: Self.colV))

        // 5. Attention with the relative bias, sink and output gate.
        let mask = state.mask(slot: slot)
        let extent = Inkling2AttentionShape.relExtent
        if t == 1 {
            decodeAttention.encode(commandBuffer: cb, .init(
                queries: .init(s.q), cache: kv, relative: .init(s.rel), table: lw.relBias,
                sink: .init(s.y, offset: Self.colSink * 4), gate: .init(s.y, offset: Self.colGate * 4),
                output: .init(s.attn), partials: .init(s.partials), position: start, mask: mask,
                relExtent: extent))
        } else {
            prefillAttention.encode(commandBuffer: cb, .init(
                queries: .init(s.q), queryStride: Self.qWidth, cache: kv,
                relative: .init(s.rel), relativeStride: Self.relWidth, table: lw.relBias,
                sink: .init(s.y, offset: Self.colSink * 4), sinkStride: Self.yStride,
                gate: .init(s.y, offset: Self.colGate * 4), gateStride: Self.yStride,
                output: .init(s.attn), rows: t, startPosition: start, mask: mask))
        }

        // 6. Output projection, post-attention convolution (with the block
        //    residual unless this layer starts a block), its tail.
        projection.encode(commandBuffer: cb, weights: w.dense, input: s.attn, output: s.o,
                          tokens: t, inputs: Self.qWidth, outStride: d,
                          segments: [.init(weightOffset: lw.output, inputStride: Self.qWidth, rows: d,
                                           outColumn: 0, format: .q8_0)])
        let blockStart = layer % Self.blockSize == 0
        let sum = slots.take()
        shortConv.encode(commandBuffer: cb, x: .init(s.o), xStride: d, weight: lw.attnPostConv,
                         tail: tails.attentionOutput,
                         residual: blockStart ? nil : s.arenaRef(cur), residualStride: d,
                         output: s.arenaRef(sum), outputStride: d, rows: t)
        convTail.encode(commandBuffer: cb, x: .init(s.o), xStride: d, rows: t, tails: tails.attentionOutput,
                        runs: [.init(xColumn: 0, tailOffset: 0, channels: d)])
        if blockStart {
            blocks.append(cur)          // the visit's input joins the completed blocks
        } else {
            slots.release(cur)
        }

        // 7. MoE input: depth mix (+ the MoE message gate), norm to FP32 and FP16.
        var moeInput = Inkling2BufferRef(s.ffnMix)
        encode { enc in
            kernels.encodeDepthMix(enc, arena: s.arena, stateOffsets: (blocks + [sum]).map { s.arenaFloats($0) },
                                   stateStride: d, queries: lw.ffnResQuery.buffer,
                                   queriesOffset: lw.ffnResQuery.offset, queryCount: 1,
                                   out: s.ffnMix, outStride: d, outQueryStride: plane, d: d, tokens: t)
            if bank != nil, let loop = lw.loop {
                kernels.encodeMessageGate(enc, x: s.ffnMix, xStride: d,
                                          message: s.messages, messageOffset: 3 * plane * 4, messageStride: d,
                                          down: loop.down[3].buffer, downOffset: loop.down[3].offset,
                                          gain: loop.gain[3].buffer, gainOffset: loop.gain[3].offset,
                                          forget: loop.forget[3].buffer, forgetOffset: loop.forget[3].offset,
                                          rank: w.gateRank, out: s.ffnMix, outOffset: plane * 4,
                                          outStride: d, d: d, tokens: t)
                moeInput = .init(s.ffnMix, offset: plane * 4)
            }
            kernels.encodeRMSNormDual(enc, x: moeInput.buffer, xOffset: moeInput.offset, xStride: d,
                                      weight: lw.ffnNorm.buffer, weightOffset: lw.ffnNorm.offset, eps: w.rmsEps,
                                      out: s.moeIn, outStride: d, outHalf: s.moeInHalf, halfStride: d,
                                      d: d, tokens: t)
            // 8. Router and routed experts.
            moe.encodeRouter(enc, shape: moeShape, x: s.moeIn, xStride: d,
                             weight: lw.router.buffer, weightOffset: lw.router.offset,
                             bias: lw.routerBias.buffer, biasOffset: lw.routerBias.offset,
                             ids: s.moeScratch.ids, routeWeights: s.moeScratch.routeWeights,
                             tokens: t, routeScale: w.routeScale)
            if t == 1 {
                moe.encodeDecode(enc, shape: moeShape, x: s.moeInHalf, ids: s.moeScratch.ids,
                                 routeWeights: s.moeScratch.routeWeights, pool: w.experts.decodePool(layer: layer),
                                 activation: s.decodeActivation, out: s.moeOut)
            }
        }
        if t > 1 {
            // A staged layer's command buffer waits here for its slab and
            // releases it after the experts' encoder.
            let pool = try w.experts.beginPrefillVisit(layer: layer, commandBuffer: cb)
            encode { enc in
                moe.encodePrefill(enc, hidden: s.moeInHalf, hiddenStride: d, pool: pool,
                                  scratch: s.moeScratch, tokens: t, out: s.moeOut, outStride: d)
            }
            try w.experts.endPrefillVisit(layer: layer, commandBuffer: cb)
        }

        // 9. Post-MoE convolution with the residual, its tail.
        let next = slots.take()
        shortConv.encode(commandBuffer: cb, x: .init(s.moeOut), xStride: d, weight: lw.ffnPostConv,
                         tail: tails.moeOutput, residual: s.arenaRef(sum), residualStride: d,
                         output: s.arenaRef(next), outputStride: d, rows: t)
        convTail.encode(commandBuffer: cb, x: .init(s.moeOut), xStride: d, rows: t, tails: tails.moeOutput,
                        runs: [.init(xColumn: 0, tailOffset: 0, channels: d)])
        slots.release(sum)
        return next
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Inkling2ForwardRunnerError.commandFailed("unable to create an Inkling 2 command buffer")
        }
        return cb
    }
}
