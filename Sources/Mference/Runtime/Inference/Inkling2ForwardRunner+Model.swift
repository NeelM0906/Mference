import Foundation
import Metal

enum Inkling2ModelBindingError: Error, CustomStringConvertible, Equatable {
    case unsupported(String)
    case tensor(String)
    case experts(String)

    var description: String {
        switch self {
        case .unsupported(let s): return "Inkling 2 runner: " + s
        case .tensor(let s): return "Inkling 2 runner: tensor " + s
        case .experts(let s): return "Inkling 2 runner: experts " + s
        }
    }
}

extension Inkling2RunnerWeights {
    /// The runner's view of a loaded `inkling2Loop` install: every resident
    /// tensor in place in the mapped weights buffer (dense Q8_0 and BF16
    /// tensors as byte offsets into it, the one buffer K3's stacked
    /// projection reads), each tensor's format, size and alignment checked
    /// against what the kernels read; and the routed experts from K2's
    /// `Inkling2ExpertStore`, which reads the layer files itself under its
    /// residency plan for this host (on 32 GB: the looped layers 6-17
    /// resident, 6.9 GB; the rest staged through two slabs for prefill and
    /// mapped for decode). The `Model`'s own expert streamers are never
    /// opened, so load it with `.pread` streaming: `.resident` would warm the
    /// whole 15.8 GB pool at load.
    ///
    /// The kernels are built for Inkling 2 Mini's shape, so the config must
    /// be the baseline's.
    init(model: Model, physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) throws {
        let c = model.config
        try Self.checkShape(c)
        let d = c.hiddenSize
        let embedding = try model.inkling2Embedding
        let dense = embedding.buffer

        func q8(_ view: TensorView, _ name: String, rows: Int, cols: Int) throws -> Int {
            guard view.dtype == GGMLBlockFormat.q8_0.residentDtype else {
                throw Inkling2ModelBindingError.tensor("\(name) is dtype \(view.dtype), not Q8_0")
            }
            return try place(view, name, bytes: rows * cols / 32 * 34, alignment: 2)
        }
        func bf16(_ view: TensorView, _ name: String, rows: Int, cols: Int) throws -> Int {
            guard view.dtype == 1 else { throw Inkling2ModelBindingError.tensor("\(name) is dtype \(view.dtype), not BF16") }
            return try place(view, name, bytes: rows * cols * 2, alignment: 8)
        }
        /// A dense tensor's offset in `dense`, the buffer every projection reads.
        func place(_ view: TensorView, _ name: String, bytes: Int, alignment: Int) throws -> Int {
            guard view.buffer === dense else {
                throw Inkling2ModelBindingError.tensor("\(name) is not in the mapped weights buffer")
            }
            guard Int(view.length) == bytes else {
                throw Inkling2ModelBindingError.tensor("\(name) is \(view.length) bytes, expected \(bytes)")
            }
            guard Int(view.offset) % alignment == 0, Int(view.offset) + bytes <= dense.length else {
                throw Inkling2ModelBindingError.tensor("\(name) at offset \(view.offset) is misplaced")
            }
            return Int(view.offset)
        }
        /// An F32 tensor of `count` values (16-byte aligned: the router and
        /// the mixes read float4).
        func f32(_ view: TensorView, _ name: String, count: Int) throws -> Inkling2BufferRef {
            guard view.dtype == 3 else { throw Inkling2ModelBindingError.tensor("\(name) is dtype \(view.dtype), not F32") }
            guard Int(view.length) == count * 4 else {
                throw Inkling2ModelBindingError.tensor("\(name) is \(view.length) bytes, expected \(count * 4)")
            }
            guard Int(view.offset) % 16 == 0, Int(view.offset) + count * 4 <= view.buffer.length else {
                throw Inkling2ModelBindingError.tensor("\(name) at offset \(view.offset) is misplaced")
            }
            return Inkling2BufferRef(view.buffer, offset: Int(view.offset))
        }

        let heads = c.numHeads, kv = c.numKVHeads * c.headDim, q = heads * c.headDim
        let rel = heads * c.relativePosition.dRel, extent = c.relativePosition.extent
        let taps = c.numKVHeads * c.sconvKernelSize, k = c.sconvKernelSize
        let rank = c.inkling2.loopGateRank
        let shape = Inkling2MoEShape(d: d, f: c.moeIntermediateSize, experts: c.numExperts, topK: c.topKExperts)
        let expertLayout = try model.inkling2ExpertLayout()
        var specs: [Inkling2ExpertLayerSpec] = []
        for L in 0..<c.numLayers {
            let blob = expertLayout.layers[L]
            let encoding: Inkling2MoEEncoding = blob.format == .nvfp4 ? .nvfp4 : .q8_0
            let stride = Int(blob.expertStride)
            let entry = model.packedExpertsLayout.layers[L]
            // K2's pools read blob `e` at `e * stride`, gate-up at 0 and down
            // right after it (K1's 256-byte boundary falls exactly there).
            guard blob.gateUp.offset == 0, Int(blob.down.offset) == shape.gateUpBytes(encoding),
                  entry.experts.count == c.numExperts,
                  entry.experts.enumerated().allSatisfy({ $0.element.offset == UInt64($0.offset * stride) }) else {
                throw Inkling2ModelBindingError.experts("layer \(L) is not laid out at its stride as K2's pools read it")
            }
            var gateUp: (buffer: MTLBuffer, offset: Int)?, down: (buffer: MTLBuffer, offset: Int)?
            if encoding == .nvfp4 {
                let g = try f32(model.inkling2ExpertGateUpScale(layer: L), "blk.\(L).ffn_gate_up_exps.scale",
                                count: c.numExperts)
                let h = try f32(model.inkling2ExpertDownScale(layer: L), "blk.\(L).ffn_down_exps.scale",
                                count: c.numExperts)
                gateUp = (g.buffer, g.offset)
                down = (h.buffer, h.offset)
            }
            let path = model.directoryURL.appendingPathComponent("packed_experts")
                .appendingPathComponent(entry.file).path
            specs.append(Inkling2ExpertLayerSpec(
                file: Inkling2ExpertLayerFile(path: path, offset: 0, bytes: c.numExperts * stride),
                encoding: encoding, expertStride: stride, gateUpScales: gateUp, downScales: down))
        }
        let plan = Inkling2ExpertResidencyPlan.make(
            layerBytes: specs.map(\.file.bytes),
            loopedLayers: c.inkling2.loopCoreStart..<c.inkling2.loopCoreEnd, physicalMemory: physicalMemory)
        let store: Inkling2ExpertStore
        do {
            store = try Inkling2ExpertStore(device: model.device, shape: shape, layers: specs, plan: plan)
        } catch {
            throw Inkling2ModelBindingError.experts("\(error)")
        }

        var layers: [Layer] = []
        for L in 0..<c.numLayers {
            let n = "blk.\(L)."
            let global = c.fullAttentionLayerMask[L] == 1
            var loop: Loop?
            if c.inkling2.layerIsLooped(L) {
                let gates = Inkling2MessageGate.allCases
                loop = Loop(
                    messageQueries: try f32(model.inkling2LoopMessageQuery(layer: L), n + "loop_msg_query", count: 4 * d),
                    down: try gates.map { try f32(model.inkling2LoopGateDown(layer: L, $0), n + "loop_down", count: rank * 2 * d) },
                    gain: try gates.map { try f32(model.inkling2LoopGateGain(layer: L, $0), n + "loop_gain", count: d * rank) },
                    forget: try gates.map { try f32(model.inkling2LoopGateForget(layer: L, $0), n + "loop_forget", count: d * rank) })
            }
            layers.append(Layer(
                isGlobal: global,
                attnNorm: global ? nil : try f32(model.inkling2AttnNorm(layer: L), n + "attn_norm", count: d),
                inputNorms: global ? [
                    try f32(model.inkling2AttnQInputNorm(layer: L), n + "attn_q_inp_norm", count: d),
                    try f32(model.inkling2AttnKInputNorm(layer: L), n + "attn_k_inp_norm", count: d),
                    try f32(model.inkling2AttnVInputNorm(layer: L), n + "attn_v_inp_norm", count: d),
                ] : [],
                q: try q8(model.inkling2AttnQ(layer: L), n + "attn_q", rows: q, cols: d),
                k: try q8(model.inkling2AttnK(layer: L), n + "attn_k", rows: kv, cols: d),
                v: try q8(model.inkling2AttnV(layer: L), n + "attn_v", rows: kv, cols: d),
                rel: try q8(model.inkling2AttnRel(layer: L), n + "attn_rel", rows: rel, cols: d),
                gate: try q8(model.inkling2AttnGate(layer: L), n + "attn_gate", rows: q, cols: d),
                output: try q8(model.inkling2AttnOutput(layer: L), n + "attn_output", rows: d, cols: q),
                sinkProj: try bf16(model.inkling2AttnSinkProj(layer: L), n + "attn_sink_proj", rows: heads, cols: d),
                kConvProj: global ? try bf16(model.inkling2AttnKConvProj(layer: L), n + "attn_k_conv_proj", rows: taps, cols: d) : nil,
                vConvProj: global ? try bf16(model.inkling2AttnVConvProj(layer: L), n + "attn_v_conv_proj", rows: taps, cols: d) : nil,
                qClip: try f32(model.inkling2AttnQClip(layer: L), n + "attn_q_clip", count: q),
                relClip: try f32(model.inkling2AttnRelClip(layer: L), n + "attn_rel_clip", count: rel),
                relBias: try f32(model.inkling2AttnRelBias(layer: L), n + "attn_rel_bias", count: extent * (rel / heads)),
                kConv: try f32(model.inkling2AttnKConv(layer: L), n + "attn_k_conv", count: kv * k),
                vConv: try f32(model.inkling2AttnVConv(layer: L), n + "attn_v_conv", count: kv * k),
                attnPostConv: try f32(model.inkling2AttnPostConv(layer: L), n + "attn_post_conv", count: d * k),
                ffnPostConv: try f32(model.inkling2FFNPostConv(layer: L), n + "ffn_post_conv", count: d * k),
                attnResQuery: L == 0 ? nil : try f32(model.inkling2AttnResQuery(layer: L), n + "attn_res_query",
                                                     count: (global ? 3 : 1) * d),
                ffnResQuery: try f32(model.inkling2FFNResQuery(layer: L), n + "ffn_res_query", count: d),
                ffnNorm: try f32(model.inkling2FFNNorm(layer: L), n + "ffn_norm", count: d),
                router: try f32(model.inkling2Router(layer: L), n + "ffn_gate_inp", count: c.numExperts * d),
                routerBias: try f32(model.inkling2RouterSelectionBias(layer: L), n + "exp_probs_b", count: c.numExperts),
                loop: loop))
        }
        self.init(
            dense: dense,
            embedding: try q8(embedding, "token_embd", rows: c.vocabSize, cols: d),
            embeddingNorm: try f32(model.inkling2EmbeddingNorm, "token_embd_norm", count: d),
            outputResQuery: try f32(model.inkling2OutputResQuery, "output_res_query", count: d),
            outputNorm: try f32(model.inkling2OutputNorm, "output_norm", count: d),
            output: try q8(model.inkling2OutputHead, "output", rows: c.vocabSize, cols: d),
            layers: layers,
            experts: store,
            vocabRows: c.vocabSize,
            vocab: c.unpaddedVocabSize > 0 ? c.unpaddedVocabSize : c.vocabSize,
            gateRank: rank,
            routeScale: Float(c.routedScalingFactor),
            finalLogitSoftcap: Float(c.finalLogitSoftcap),
            rmsEps: Inkling2AttentionShape.rmsEps)
    }

    /// The runner's kernels are built for Inkling 2 Mini: K3's attention
    /// shape, K2's 128-expert router, the loop and block layout, and the
    /// constants the kernels hold (clips, log scale, SITU caps). Anything else
    /// is refused by name rather than run wrong.
    static func checkShape(_ c: ArchConfig) throws {
        let m = ArchConfig.inkling2Mini
        let shape = Inkling2AttentionShape.self
        var differs: [String] = []
        func check(_ name: String, _ ok: Bool) { if !ok { differs.append(name) } }
        check("family", c.family == .inkling2Loop)
        check("hiddenSize", c.hiddenSize == shape.hidden)
        check("heads", c.numHeads == shape.numQHeads && c.numKVHeads == shape.numKVHeads
              && c.headDim == shape.headDim)
        check("numLayers", c.numLayers == Inkling2VisitPlan.storedLayers)
        check("fullAttentionLayerMask", c.fullAttentionLayerMask == m.fullAttentionLayerMask
              && (0..<c.numLayers).allSatisfy { (c.fullAttentionLayerMask[$0] == 1) == Inkling2VisitPlan.isGlobal(layer: $0) })
        check("slidingWindow", c.slidingWindow == shape.slidingWindow)
        check("relativePosition", c.relativePosition == m.relativePosition
              && c.relativePosition.dRel == shape.dRel && c.relativePosition.extent == shape.relExtent)
        check("sconvKernelSize", c.sconvKernelSize == shape.convKernel)
        check("experts", c.numExperts == 128 && c.topKExperts == m.topKExperts
              && c.moeIntermediateSize == m.moeIntermediateSize)
        check("inkling2", c.inkling2 == m.inkling2)
        check("vocab", c.vocabSize == m.vocabSize && c.unpaddedVocabSize == m.unpaddedVocabSize)
        check("routedScalingFactor", c.routedScalingFactor == m.routedScalingFactor)
        check("finalLogitSoftcap", c.finalLogitSoftcap == m.finalLogitSoftcap)
        guard differs.isEmpty else {
            throw Inkling2ModelBindingError.unsupported(
                "the kernels are built for Inkling 2 Mini; this install differs in \(differs.joined(separator: ", "))")
        }
    }
}

extension Inkling2ForwardRunner {
    /// The runner over a loaded install. Its chunk is the runtime
    /// configuration's prefill chunk (the scratch is sized for it).
    convenience init(model: Model, context: MetalContext, maxContext: Int,
                     runtimeConfiguration: RuntimeConfiguration) throws {
        try self.init(weights: Inkling2RunnerWeights(model: model), context: context,
                      maxContext: maxContext,
                      maxChunkTokens: runtimeConfiguration.prefillChunkTokens)
    }
}

extension Model {
    /// Stored layer `layer`'s expert file mapped read-only for CPU reads (the
    /// whole-model CPU reference): expert `e` at `e * expertStride`, its
    /// gate-up then down rows as `inkling2ExpertLayout()` describes. Pages
    /// fault in only where read. The runner itself takes its experts from
    /// `Inkling2ExpertStore`, not from this.
    public func inkling2MappedExpertLayer(_ layer: Int) throws -> (buffer: MTLBuffer, expertStride: Int) {
        let stride = Int(try inkling2ExpertLayout().layers[layer].expertStride)
        let name = "packed_experts/" + packedExpertsLayout.layers[layer].file
        let path = directoryURL.appendingPathComponent(name).path
        let fd = Darwin.open(path, O_RDONLY)
        guard fd >= 0 else { throw ModelError.missingFile(name: name) }
        defer { close(fd) }
        let pageSize = Int(getpagesize())
        let length = (config.numExperts * stride + pageSize - 1) / pageSize * pageSize
        guard let base = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else {
            throw ModelError.residentBufferWrapFailed
        }
        let mapping = Inkling2ReadOnlyMapping(base: base, length: length)
        nonisolated(unsafe) let start = base
        guard let buffer = device.makeBuffer(bytesNoCopy: start, length: length, options: .storageModeShared,
                                             deallocator: { _, _ in _ = mapping }) else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = "inkling2.reference.experts.layer\(layer)"
        return (buffer, stride)
    }
}

/// Keeps a read-only mapping alive while a buffer over it exists.
private final class Inkling2ReadOnlyMapping: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int
    init(base: UnsafeMutableRawPointer, length: Int) {
        self.base = base
        self.length = length
    }
    deinit { munmap(base, length) }
}
