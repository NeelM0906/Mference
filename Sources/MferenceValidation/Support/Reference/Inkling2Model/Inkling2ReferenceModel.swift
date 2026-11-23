import Foundation

/// Whole-model CPU reference for Inkling 2 Mini (`inkling2_loop`): the
/// oracle every GPU path of the native runner is checked against.
///
/// Per token: normalized embedding, prelude layers `0 ..< coreStart`, the
/// core `coreStart ..< coreEnd` once per pass (every pass restarts from the
/// prelude output; passes after the first gate in messages read from the
/// previous pass's two core blocks), the coda, a final depth mix, norm, head
/// and logit softcap. Each of the `numVisits` visits keeps its own keys,
/// values and four convolution tails (`Inkling2ReferenceConfig.cacheSlot`).
///
/// `forward(tokens:)` runs a chunk layer-major, as chunked prefill does: one
/// visit over every token of the chunk, then the next visit. Chunk size does
/// not change the result, so a chunk of one token is decode. Clarity over
/// speed: plain loops, double-precision accumulation, full key history.
public final class Inkling2ReferenceModel {

    public let config: Inkling2ReferenceConfig
    public let weights: Inkling2ReferenceWeights

    /// Tokens consumed since the last `reset()`; the next token's position.
    public var position: Int { state.position }

    /// When true, the next `forward` records an `Inkling2ReferenceTrace`.
    public var captureTrace = false

    /// Where the arithmetic rounds. `.fp32` is the vendor's FP32 semantics.
    /// `.metalRunner` also rounds to FP16 wherever `Inkling2ForwardRunner`
    /// keeps a value in half: the normalized projection inputs, the clipped
    /// queries, the cached keys and values, the gated attention output, and
    /// the experts' input and activation (the router still reads FP32). A
    /// comparison with the runner then measures accumulation order, not the
    /// runner's storage formats. `.llamaCPU` rounds as llama.cpp's CPU backend
    /// does in K0's CPU-only traces (G6a): the input of every Q8_0 and NVFP4
    /// product (dense projections, experts' input and activation, the head)
    /// to Q8_0 blocks (ggml's `vec_dot_type`), the input of a BF16 product
    /// (sink and tap projections) to BF16, and the cached keys and values
    /// to FP16 (llama.cpp's default cache type). F32 products (router, loop
    /// gates) stay FP32.
    public enum Precision: Sendable { case fp32, metalRunner, llamaCPU }
    public var precision: Precision = .fp32
    /// The trace of the most recent `forward` with `captureTrace` set.
    public private(set) var lastTrace: Inkling2ReferenceTrace?
    /// Expert choices for the next `forward`, `[visit order][token]`, as
    /// another implementation made them: each router takes these experts,
    /// weighted by its own scores, instead of its own top-k, so a comparison
    /// is not thrown off by near-ties broken differently. The trace records
    /// each forced choice's shortfall. `forward` clears it.
    public var forcedRoutes: [[[Int]]]?
    /// Teacher forcing for the next `forward`, `[visit order][token][hidden]`:
    /// after each visit the running sum continues from this value (another
    /// implementation's output of the same visit), so the block list, the
    /// message bank, the next visit's input and the state of every later
    /// visit derive from it. The trace still records the reference's own
    /// output of each visit: each visit is compared alone, from the same
    /// input, without the error the visits before it amplify. `forward`
    /// clears it.
    public var forcedVisitOutputs: [[[Float]]]?
    /// Which features the computation exercised since the last `reset()`.
    public private(set) var coverage = Inkling2ReferenceCoverage()

    private var state: Inkling2ReferenceState

    public init(config: Inkling2ReferenceConfig, weights: Inkling2ReferenceWeights) throws {
        try weights.validate(against: config)
        self.config = config
        self.weights = weights
        self.state = Inkling2ReferenceState(config: config)
    }

    /// Forget every token: empty caches, zero convolution tails, position 0.
    public func reset() {
        state = Inkling2ReferenceState(config: config)
        coverage = Inkling2ReferenceCoverage()
        lastTrace = nil
    }

    /// A copy of the complete numerical state, for prefix-snapshot tests.
    public func saveState() -> Inkling2ReferenceState { state }

    /// Put back a state taken by `saveState()` from a model of the same config.
    public func restoreState(_ saved: Inkling2ReferenceState) {
        precondition(saved.slots.count == config.numVisits, "state from another config")
        state = saved
    }

    public enum LogitsMode: Sendable {
        /// Logits for every token of the chunk.
        case all
        /// Logits for the last token only (one entry).
        case last
        /// No head evaluation.
        case none
    }

    /// Runs `tokens` at positions `position ..< position + tokens.count` and
    /// returns their logits (`vocabUnpadded` values each) per `logits`.
    @discardableResult
    public func forward(tokens: [Int32], logits mode: LogitsMode = .last) -> [[Float]] {
        precondition(!tokens.isEmpty, "empty chunk")
        let c = config
        let w = weights
        let start = state.position
        let n = tokens.count
        var trace = captureTrace ? Inkling2ReferenceTrace(chunkStart: start) : nil
        let forced = forcedRoutes
        let forcedOutputs = forcedVisitOutputs
        forcedRoutes = nil
        forcedVisitOutputs = nil
        if let forced {
            precondition(forced.count == c.numVisits && forced.allSatisfy { $0.count == n },
                         "forced routes are not [visit][token]")
        }
        if let forcedOutputs {
            precondition(forcedOutputs.count == c.numVisits
                         && forcedOutputs.allSatisfy { $0.count == n && $0.allSatisfy { $0.count == c.hiddenSize } },
                         "forced visit outputs are not [visit][token][hidden]")
        }

        // Per token of the chunk: the running block sum and the completed blocks.
        var cur: [[Float]] = tokens.map { token in
            precondition(token >= 0 && Int(token) < c.vocabPadded, "token \(token) outside the vocabulary")
            return Inkling2ReferenceOps.rmsNorm(w.embedding.row(Int(token)), weight: w.embeddingNorm, eps: c.rmsEps)
        }
        var blocks = [[[Float]]](repeating: [], count: n)

        func run(_ visit: Inkling2ReferenceVisit, messages: [[[Float]]]?) {
            var routes: [Inkling2Route] = []
            var shortfalls: [Float] = []
            for t in 0..<n {
                let (out, route, shortfall) = layerStep(visit: visit, token: t, chunkStart: start,
                                                        cur: cur[t], blocks: &blocks[t],
                                                        messages: messages?[t],
                                                        forcedExperts: forced?[visit.order][t])
                cur[t] = out
                routes.append(route)
                shortfalls.append(shortfall)
            }
            trace?.visitOutputs.append(cur)
            trace?.routes.append(routes)
            trace?.routeShortfalls.append(shortfalls)
            if let forcedOutputs { cur = forcedOutputs[visit.order] }
        }

        let schedule = c.schedule
        var cursor = 0
        // Prelude.
        while cursor < c.coreStart {
            run(schedule[cursor], messages: nil)
            cursor += 1
        }
        let preludeCur = cur
        let preludeBlocks = blocks
        // Core passes.
        for pass in 0..<c.numPasses {
            var messageStates: [[[Float]]]? = nil
            if pass > 0 {
                // The previous pass's bank: its blocks after the prelude's and
                // the first core block's start (the first core block sum), and
                // its output (the second core block sum).
                messageStates = (0..<n).map { t in
                    Array(blocks[t][(preludeBlocks[t].count + 1)...]) + [cur[t]]
                }
                cur = preludeCur
                blocks = preludeBlocks
            }
            for layer in c.coreStart..<c.coreEnd {
                let visit = schedule[cursor]
                precondition(visit.layer == layer && visit.pass == pass)
                var messages: [[[Float]]]? = nil
                if let bank = messageStates {
                    let queries = w.loopLayers[layer - c.coreStart].messageQueries
                    messages = bank.map { Inkling2ReferenceOps.mix(states: $0, queries: queries) }
                }
                run(visit, messages: messages)
                cursor += 1
            }
        }
        // Coda.
        while cursor < schedule.count {
            run(schedule[cursor], messages: nil)
            cursor += 1
        }

        let hidden = (0..<n).map { t in
            Inkling2ReferenceOps.rmsNorm(
                Inkling2ReferenceOps.mix(states: blocks[t] + [cur[t]], queries: [w.outputResQuery])[0],
                weight: w.outputNorm, eps: c.rmsEps)
        }
        trace?.finalHidden = hidden
        state.position += n
        if captureTrace { lastTrace = trace }

        let wanted: [Int]
        switch mode {
        case .all: wanted = Array(0..<n)
        case .last: wanted = [n - 1]
        case .none: wanted = []
        }
        return wanted.map { logits(hidden: hidden[$0]) }
    }

    /// Head over a final hidden state: the first `vocabUnpadded` rows of
    /// `output`, then the logit softcap.
    public func logits(hidden: [Float]) -> [Float] {
        let input = precision == .llamaCPU ? Inkling2Q8_0.roundTrip(hidden) : hidden
        return (0..<config.vocabUnpadded).map {
            let raw = Inkling2ReferenceOps.dot(weights.output.row($0), input)
            if config.finalLogitSoftcap > 0 && abs(raw) > config.finalLogitSoftcap {
                coverage.saturatedLogits += 1
            }
            return Inkling2ReferenceOps.softcap(raw, config.finalLogitSoftcap)
        }
    }

    // MARK: One visit, one token

    private func layerStep(visit: Inkling2ReferenceVisit, token t: Int, chunkStart: Int,
                           cur: [Float], blocks: inout [[Float]],
                           messages: [[Float]]?, forcedExperts: [Int]?) -> ([Float], Inkling2Route, Float) {
        typealias Op = Inkling2ReferenceOps
        let c = config
        let il = visit.layer
        let lw = weights.layers[il]
        let sliding = c.slidingLayers[il]
        let pos = chunkStart + t
        let hd = c.headDim
        var slot = state.slots[visit.cacheSlot]
        let loop = messages == nil ? nil : weights.loopLayers[il - c.coreStart]
        let precision = self.precision
        func toHalf(_ v: [Float]) -> [Float] { v.map { Float(Float16($0)) } }
        /// The input of a Q8_0 or NVFP4 product.
        func q8In(_ v: [Float]) -> [Float] {
            switch precision {
            case .fp32: return v
            case .metalRunner: return toHalf(v)
            case .llamaCPU: return Inkling2Q8_0.roundTrip(v)
            }
        }
        /// The input of a BF16 product.
        func bf16In(_ v: [Float]) -> [Float] {
            switch precision {
            case .fp32: return v
            case .metalRunner: return toHalf(v)
            case .llamaCPU: return v.map(Op.bf16Round)
            }
        }
        /// A cached key or value row, and a query.
        func cached(_ v: [Float]) -> [Float] { precision == .fp32 ? v : toHalf(v) }
        func queryIn(_ v: [Float]) -> [Float] { precision == .metalRunner ? toHalf(v) : v }
        let roundActivation: (([Float]) -> [Float])? = {
            switch precision {
            case .fp32: return nil
            case .metalRunner: return { $0.map { Float(Float16($0)) } }
            case .llamaCPU: return { Inkling2Q8_0.roundTrip($0) }
            }
        }()

        // Attention inputs: the depth mix of the earlier blocks and the
        // running sum (one mix on sliding layers, one each for Q, K and V on
        // global layers); layer 0 reads the embedding directly.
        let inpSA = cur
        var inputs = [cur, cur, cur]
        if !lw.attnResQueries.isEmpty {
            let mixed = Op.mix(states: blocks + [cur], queries: lw.attnResQueries)
            inputs = sliding ? [mixed[0], mixed[0], mixed[0]] : mixed
        }
        if let msgs = messages, let lp = loop {
            for j in 0..<3 {
                inputs[j] = Op.control(inputs[j], message: msgs[j], down: lp.controlDown[j],
                                       gain: lp.controlGain[j], forget: lp.controlForget[j])
            }
            coverage.messageGates += 3
        }
        if sliding {
            inputs = inputs.map { Op.rmsNorm($0, weight: lw.attnNorm!, eps: c.rmsEps) }
        } else {
            inputs = [Op.rmsNorm(inputs[0], weight: lw.qInputNorm!, eps: c.rmsEps),
                      Op.rmsNorm(inputs[1], weight: lw.kInputNorm!, eps: c.rmsEps),
                      Op.rmsNorm(inputs[2], weight: lw.vInputNorm!, eps: c.rmsEps)]
        }
        let (xq, xk, xv) = (q8In(inputs[0]), q8In(inputs[1]), q8In(inputs[2]))
        let (bq, bk, bv) = (bf16In(inputs[0]), bf16In(inputs[1]), bf16In(inputs[2]))

        // Projections. K and V pass through their short convolutions (with
        // input-dependent taps on global layers) before the clip and norm.
        let qRaw = lw.wq.apply(xq)
        let kPre = lw.wk.apply(xk)
        let vPre = lw.wv.apply(xv)
        let rRaw = lw.rel.apply(xq)
        let sinks = lw.sinkProj.apply(bq)
        let kConv = Op.shortConv(kPre, tail: slot.tails[0], weights: lw.kConv,
                                 dynamicTaps: lw.kConvProj?.apply(bk), heads: c.numKVHeads)
        let vConv = Op.shortConv(vPre, tail: slot.tails[1], weights: lw.vConv,
                                 dynamicTaps: lw.vConvProj?.apply(bv), heads: c.numKVHeads)
        slot.tails[0] = Op.advance(tail: slot.tails[0], with: kPre)
        slot.tails[1] = Op.advance(tail: slot.tails[1], with: vPre)

        let tau: Float = sliding ? 1 : Op.logScale(position: pos, floor: c.logScaleFloor, alpha: c.logScaleAlpha)
        if !sliding && tau != 1 { coverage.logScaledQueries += 1 }
        var q = [[Float]](), r = [[Float]]()
        for h in 0..<c.numHeads {
            let qh = Array(qRaw[(h * hd)..<((h + 1) * hd)])
            let qs = Array(lw.qClipScale[(h * hd)..<((h + 1) * hd)])
            if Op.l2ClipActive(qh, scale: qs, threshold: c.qkClip) { coverage.clippedQueries += 1 }
            q.append(queryIn(Op.l2Clip(qh, scale: qs, threshold: c.qkClip)).map { $0 * tau })
            let rh = Array(rRaw[(h * c.relDim)..<((h + 1) * c.relDim)])
            let rs = Array(lw.relClipScale[(h * c.relDim)..<((h + 1) * c.relDim)])
            if Op.l2ClipActive(rh, scale: rs, threshold: c.relClip) { coverage.clippedRelStates += 1 }
            r.append(Op.l2Clip(rh, scale: rs, threshold: c.relClip).map { $0 * tau })
        }
        var key = [Float](), value = [Float]()
        for g in 0..<c.numKVHeads {
            let kh = Array(kConv[(g * hd)..<((g + 1) * hd)])
            if Op.l2ClipActive(kh, threshold: c.qkClip) { coverage.clippedKeys += 1 }
            key += cached(Op.l2Clip(kh, threshold: c.qkClip))
            value += cached(Op.rmsNorm(Array(vConv[(g * hd)..<((g + 1) * hd)]), eps: c.rmsEps))
        }
        precondition(slot.keys.count == pos, "visit \(visit.order) cache at \(slot.keys.count), token at \(pos)")
        slot.keys.append(key)
        slot.values.append(value)

        // Attention with the relative bias and the sink; sliding layers see
        // backward distances 0 ..< slidingWindow.
        let extent = c.relativeExtent(layer: il)
        var visible: [Int] = []
        for j in 0...pos {
            let distance = pos - j
            if sliding && distance >= c.slidingWindow {
                coverage.windowMaskedKeys += 1
                continue
            }
            if distance >= extent { coverage.keysPastBiasExtent += 1 }
            visible.append(j)
        }
        var attn = [Float]()
        let scale = Float(1.0 / Double(hd).squareRoot())
        for h in 0..<c.numHeads {
            let g = h / c.headsPerKVHead
            let keys = visible.map { Array(slot.keys[$0][(g * hd)..<((g + 1) * hd)]) }
            let values = visible.map { Array(slot.values[$0][(g * hd)..<((g + 1) * hd)]) }
            let biases = visible.map { Op.relativeBias(table: lw.relBias, relState: r[h], distance: pos - $0) }
            attn += Op.attendHead(query: q[h], keys: keys, values: values, biases: biases,
                                  sink: sinks[h], scale: scale)
        }
        let gate = lw.outputGate.apply(xq).map(Op.sigmoid)
        let gated = (0..<attn.count).map { attn[$0] * gate[$0] }
        let projected = lw.wo.apply(q8In(gated))
        let attnOut = Op.shortConv(projected, tail: slot.tails[2], weights: lw.attnPostConv)
        slot.tails[2] = Op.advance(tail: slot.tails[2], with: projected)

        // A block's first layer starts a new running sum from its attention
        // output alone; its input joins the completed blocks.
        var sum: [Float]
        if il % c.attnResBlockSize == 0 {
            blocks.append(inpSA)
            sum = attnOut
        } else {
            sum = Op.add(inpSA, attnOut)
        }
        let ffnInp = sum

        // Mixture of experts on the depth mix (gated with the MoE message in
        // later passes).
        var m = Op.mix(states: blocks + [sum], queries: [lw.ffnResQuery])[0]
        if let msgs = messages, let lp = loop {
            m = Op.control(m, message: msgs[3], down: lp.controlDown[3],
                           gain: lp.controlGain[3], forget: lp.controlForget[3])
            coverage.messageGates += 1
        }
        m = Op.rmsNorm(m, weight: lw.ffnNorm, eps: c.rmsEps)
        var route = Op.route(m, router: lw.router, bias: lw.routerBias, topK: c.topK, routeScale: c.routeScale)
        let unbiased = Op.route(m, router: lw.router, bias: [Float](repeating: 0, count: c.numExperts),
                                topK: c.topK, routeScale: c.routeScale)
        if Set(unbiased.experts) != Set(route.experts) { coverage.biasChangedRoutes += 1 }
        var shortfall: Float = 0
        if let forcedExperts {
            precondition(forcedExperts.count == c.topK && Set(forcedExperts).count == c.topK
                         && forcedExperts.allSatisfy { (0..<c.numExperts).contains($0) }, "forced route")
            (route, shortfall) = Op.route(m, router: lw.router, bias: lw.routerBias, experts: forcedExperts,
                                          routeScale: c.routeScale)
        }
        let ff = c.expertFF
        let moe = Op.moe(q8In(m), route: route, gateUp: lw.expertGateUp, down: lw.expertDown,
                         gateSoftcap: c.gateSoftcap, upSoftcap: c.upSoftcap,
                         roundActivation: roundActivation) { gu in
            for i in 0..<ff {
                if abs(gu[i]) > c.gateSoftcap { self.coverage.saturatedGates += 1 }
                if abs(gu[ff + i]) > c.upSoftcap { self.coverage.saturatedUps += 1 }
            }
        }
        let ffnOut = Op.shortConv(moe, tail: slot.tails[3], weights: lw.ffnPostConv)
        slot.tails[3] = Op.advance(tail: slot.tails[3], with: moe)
        sum = Op.add(ffnInp, ffnOut)

        state.slots[visit.cacheSlot] = slot
        return (sum, route, shortfall)
    }
}

/// The numerical state of an `Inkling2ReferenceModel` (value type: a copy
/// is a snapshot).
public struct Inkling2ReferenceState: Sendable {
    public struct Slot: Sendable {
        /// Clipped keys, one `kvDim` row per position.
        public var keys: [[Float]] = []
        /// Normalized values, one `kvDim` row per position.
        public var values: [[Float]] = []
        /// The four convolution tails (K, V, attention output, MoE output),
        /// each the last `convKernel - 1` pre-convolution inputs, oldest first.
        public var tails: [[[Float]]]
    }

    public internal(set) var position: Int = 0
    /// One per logical cache layer, indexed by `Inkling2ReferenceVisit.cacheSlot`.
    public internal(set) var slots: [Slot]

    init(config c: Inkling2ReferenceConfig) {
        let taps = c.convKernel - 1
        func zeros(_ channels: Int) -> [[Float]] {
            [[Float]](repeating: [Float](repeating: 0, count: channels), count: taps)
        }
        slots = (0..<c.numVisits).map { _ in
            Slot(tails: [zeros(c.kvDim), zeros(c.kvDim), zeros(c.hiddenSize), zeros(c.hiddenSize)])
        }
    }
}

/// What one `forward` computed, per visit in execution order.
public struct Inkling2ReferenceTrace: Sendable {
    /// Position of the chunk's first token.
    public let chunkStart: Int
    /// `[visit order][token][hidden]`: the running block sum after each visit.
    public var visitOutputs: [[[Float]]] = []
    /// `[visit order][token]`: the experts each token's router used.
    public var routes: [[Inkling2Route]] = []
    /// `[visit order][token]`: each forced route's shortfall below the
    /// reference's own top-k cutoff (0 without `forcedRoutes`).
    public var routeShortfalls: [[Float]] = []
    /// `[token][hidden]`: the normalized state the head reads.
    public var finalHidden: [[Float]] = []

    init(chunkStart: Int) { self.chunkStart = chunkStart }
}

/// Counts of feature paths taken, so tests can show the toy checkpoint
/// exercises every one of them.
public struct Inkling2ReferenceCoverage: Sendable, Equatable {
    public var clippedQueries = 0
    public var clippedKeys = 0
    public var clippedRelStates = 0
    public var logScaledQueries = 0
    public var windowMaskedKeys = 0
    public var keysPastBiasExtent = 0
    public var messageGates = 0
    public var biasChangedRoutes = 0
    /// Expert gate pre-activations past `gateSoftcap`.
    public var saturatedGates = 0
    /// Expert `up` pre-activations past `upSoftcap`.
    public var saturatedUps = 0
    /// Raw logits past `finalLogitSoftcap`.
    public var saturatedLogits = 0

    public init() {}
}
