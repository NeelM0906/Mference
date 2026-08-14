import Foundation
import Metal

public enum Qwen38ForwardRunnerError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    case invalidInput(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let message), .invalidInput(let message),
             .commandFailed(let message):
            return message
        }
    }
}

/// Qwen 3.8 dense decode pass. One instance owns mutable KV / GDN / scratch
/// state and is serial.
///
/// The architecture is the Qwen 3.6 layer graph with the MoE branch deleted:
///
///   embed_lookup_int4(token)                       // out_scale 1.0
///   for L in 0..<64:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     attn = (mask 2) gated-DeltaNet linear attention(a)      // FP32 state
///          | (mask 1) gated full attention(a)                 // packed
///            [query ; gate] q_proj, per-head q/k norm, NeoX sub-dim RoPE
///            over 64 of 256 dims, sigmoid(gate) before o_proj
///     h += attn
///     m = rmsnorm_bf16w(h, post_attention_layernorm)
///     h += SwiGLU(m)                               // one dense MLP per layer
///   head = greedy fused lm_head | full-logits GEMV over rmsnorm(h)
///
/// Every weight is resident (`numExperts == 0`: the expert streamer never
/// opens), so a decode step encodes into a single command buffer with no
/// CPU readback between layers. Prefill v1 is sequential decode replay —
/// exact by construction; layer-major chunking is a follow-up.
public final class Qwen38ForwardRunner: ContinuableLogitProducer, ContextWindowReporting,
    ChunkedPrefillRunner, HeadlessSequentialPrefillRunner, ExactPrefillLogitProducer,
    FusedHeadLogitProducer, @unchecked Sendable {

    /// Every per-layer TensorView, resolved once at init so the decode hot
    /// path never touches the resident-index dictionary.
    private struct LayerTensors {
        let inputNorm: TensorView
        let postAttnNorm: TensorView
        // Full-attention layers (mask 1) only.
        let q: TensorView?
        let k: TensorView?
        let v: TensorView?
        let o: TensorView?
        let qNorm: TensorView?
        let kNorm: TensorView?
        // Gated-DeltaNet layers (mask 2) only.
        let linQKV: TensorView?
        let linZ: TensorView?
        let linA: TensorView?
        let linB: TensorView?
        let linOut: TensorView?
        let linConv: TensorView?
        let linALog: TensorView?
        let linDtBias: TensorView?
        let linNorm: TensorView?
        // Dense SwiGLU MLP (every layer; served by the sharedExpert accessors).
        let mlpGate: SharedExpertProjection
        let mlpUp: SharedExpertProjection
        let mlpDown: SharedExpertProjection
        let isLinear: Bool
    }

    private let model: Model
    private let ctx: MetalContext
    private let cfg: ArchConfig
    private let kv: KVCacheManager
    private let gdnState: GDNStateManager

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let elementwise: Elementwise
    private let rope: RoPE
    private let gdn: GDN
    private let mlp: SharedExpertRuntime
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let layers: [LayerTensors]

    // Decode scratch, allocated once. FP16 unless noted. At production shape
    // (D 5120, F 17408, qDim 24*256 = 6144, gdn qkvDim 10240, valueDim 6144)
    // the whole set is ~293 KB:
    //   5 x D vectors (hidden/normed/mlpX/oOut/mlpOut)          50 KB
    //   packed q + q + gate + attn out (2*qDim + 3*qDim)        60 KB
    //   3 x F MLP intermediates                                102 KB
    //   gdn qkv + conv (2 x qkvDim) + z/y/out (3 x valueDim)
    //     + a/b (2 x numVHeads)                                 76 KB
    private let hidden: MTLBuffer         // [D]
    private let normed: MTLBuffer         // [D] input_layernorm output
    private let mlpX: MTLBuffer           // [D] post_attention_layernorm output
    private let oOut: MTLBuffer           // [D] attention-branch output
    private let mlpOut: MTLBuffer         // [D] dense MLP output
    private let qPackedScratch: MTLBuffer // [2 * qDim] packed [query ; gate]
    private let qScratch: MTLBuffer       // [qDim]
    private let attnGateScratch: MTLBuffer // [qDim]
    private let attnOut: MTLBuffer        // [qDim]
    private let mlpScratchGate: MTLBuffer // [F]
    private let mlpScratchUp: MTLBuffer   // [F]
    private let mlpScratchAct: MTLBuffer  // [F]
    private let gdnQKVRaw: MTLBuffer      // [qkvDim] raw in_proj_qkv output
    private let gdnConvOut: MTLBuffer     // [qkvDim] conv + SiLU output
    private let gdnZ: MTLBuffer           // [valueDim]
    private let gdnA: MTLBuffer           // [numVHeads]
    private let gdnB: MTLBuffer           // [numVHeads]
    private let gdnY: MTLBuffer           // [valueDim] delta-rule output
    private let gdnOut: MTLBuffer         // [valueDim] gated-norm output
    private let greedyTokenBuf: MTLBuffer // [1] UInt32 fused-head output

    public let maxContext: Int
    private let useFusedGreedyHead: Bool
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }

    private static let epsilon: Float = 1e-6

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        let cfg = model.config
        try Self.validate(config: cfg, maxContext: maxContext)
        self.model = model
        self.ctx = context
        self.cfg = cfg
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: runtimeConfiguration.fp16RingEnabled,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: runtimeConfiguration.prefillConfig.chunkTokens)
        self.gdnState = try GDNStateManager(device: context.device, config: cfg)

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context,
                                        additionalShapes: cfg.decodeInt4GEMVShapes)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.rope = try RoPE(context: context)
        self.gdn = try GDN(context: context, config: cfg.linearAttention,
                           specializedHiddenSize: cfg.hiddenSize)
        // The dense MLP shares the attention quant (the manifest's
        // sharedExpert slot is deliberately absent for this family); the
        // quant-less toy manifest keeps the INT8 default.
        self.mlp = try SharedExpertRuntime(context: context,
                                           weightBits: model.manifest.quant?.attention.weightBits ?? 8,
                                           siluActivation: cfg.hiddenActivation == "silu",
                                           specializedD: cfg.hiddenSize,
                                           specializedF: cfg.intermediateSize)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)

        let device = context.device
        func buf(_ elements: Int, _ stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(elements, 1) * stride,
                                               options: .storageModeShared) else {
                throw Qwen38ForwardRunnerError.invalidConfiguration(
                    "unable to allocate Qwen 3.8 runtime scratch")
            }
            return made
        }
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let qDim = cfg.numHeads * cfg.fullHeadDim
        let la = cfg.linearAttention
        self.hidden = try buf(D)
        self.normed = try buf(D)
        self.mlpX = try buf(D)
        self.oOut = try buf(D)
        self.mlpOut = try buf(D)
        self.qPackedScratch = try buf(2 * qDim)
        self.qScratch = try buf(qDim)
        self.attnGateScratch = try buf(qDim)
        self.attnOut = try buf(qDim)
        self.mlpScratchGate = try buf(F)
        self.mlpScratchUp = try buf(F)
        self.mlpScratchAct = try buf(F)
        self.gdnQKVRaw = try buf(la.qkvDim)
        self.gdnConvOut = try buf(la.qkvDim)
        self.gdnZ = try buf(la.valueDim)
        self.gdnA = try buf(la.numVHeads)
        self.gdnB = try buf(la.numVHeads)
        self.gdnY = try buf(la.valueDim)
        self.gdnOut = try buf(la.valueDim)
        self.greedyTokenBuf = try buf(1, MemoryLayout<UInt32>.stride)

        func projection(_ view: TensorView, rows: Int, cols: Int) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                   scales: view.buffer,
                                   biases: view.buffer,
                                   weightsOffset: Int(view.offset),
                                   scalesOffset: Int(view.scaleOffset),
                                   biasesOffset: Int(view.biasOffset),
                                   rows: UInt32(rows),
                                   cols: UInt32(cols))
        }
        self.layers = try (0..<cfg.numLayers).map { L in
            let isLinear = cfg.layerIsLinear(L)
            return LayerTensors(
                inputNorm: try model.inputNorm(layer: L),
                postAttnNorm: try model.postAttnNorm(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear ? nil : try model.vProj(layer: L),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil,
                mlpGate: projection(try model.sharedExpertGate(layer: L), rows: F, cols: D),
                mlpUp: projection(try model.sharedExpertUp(layer: L), rows: F, cols: D),
                mlpDown: projection(try model.sharedExpertDown(layer: L), rows: D, cols: F),
                isLinear: isLinear)
        }
    }

    private static func validate(config: ArchConfig, maxContext: Int) throws {
        guard config.family == .qwen38 else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "Qwen38ForwardRunner requires the qwen38 family")
        }
        guard config.numExperts == 0, config.attnOutputGate,
              config.hasLinearAttentionLayers, config.ropeNeoxSubdim,
              !config.ffnSandwichNorms, !config.sharedExpertGated,
              config.intermediateSize > 0 else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "model does not match the dense Qwen 3.8 layer graph")
        }
        guard maxContext > 0 else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "Qwen 3.8 runtime context must be positive")
        }
    }

    // MARK: - LogitProducer

    public func reset() {
        kv.reset()
        gdnState.reset()
    }

    public var continuationPosition: Int { kv.position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, expectedPosition == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "Qwen 3.8 continuation cursor \(expectedPosition) does not match \(kv.position)")
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produceToken(token: token, position: position, into: logits,
                               emitHead: true, outputMode: .greedyIfAvailable)
    }

    func produceWithoutLogits(token: Int32, position: Int) async throws {
        try await produceToken(token: token, position: position, into: nil,
                               emitHead: false, outputMode: .logits)
    }

    func produceExactPrefill(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produceToken(token: token, position: position, into: logits,
                               emitHead: true, outputMode: .logits)
    }

    // MARK: - Prefill (v1: sequential decode replay)

    /// Sequential decode-path replay: each prompt token runs the full decode
    /// step, heads suppressed until the last token. Exact by construction —
    /// prefill and decode share every kernel and all recurrent state. The
    /// chunked layer-major path is a follow-up; `config.chunkTokens` only
    /// affects progress granularity here.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "Qwen 3.8 prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0, startPosition == kv.position else {
            throw PrefillError.chunkedUnsupported(
                "Qwen 3.8 prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "Qwen 3.8 prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        var position = startPosition
        var index = 0
        for token in tokens {
            try Task.checkCancellation()
            let isLast = index == tokens.count - 1
            try await produceToken(token: token, position: position,
                                   into: logits,
                                   emitHead: isLast,
                                   outputMode: outputMode)
            position += 1
            index += 1
            if index % 16 == 0 { onProgress(index) }
        }
        onProgress(tokens.count)
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: position, seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: position, seed: .logitsWritten)
    }

    // MARK: - Decode step

    /// One full decode step, encoded into a single command buffer: no router
    /// readback exists in this architecture, so nothing forces a mid-layer
    /// CPU round-trip.
    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer?,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        try Task.checkCancellation()
        guard position == kv.position, position >= 0, position < maxContext else {
            throw Qwen38ForwardRunnerError.invalidInput(
                "Qwen 3.8 position \(position) does not match its KV cursor \(kv.position)")
        }
        guard token >= 0, token < Int32(cfg.vocabSize) else {
            throw Qwen38ForwardRunnerError.invalidInput(
                "Qwen 3.8 token is outside the vocabulary")
        }
        let emitLogitsHead = emitHead
            && !(useFusedGreedyHead && outputMode == .greedyIfAvailable)
        if emitLogitsHead {
            guard let logits,
                  logits.length >= cfg.vocabSize * MemoryLayout<Float16>.stride else {
                throw Qwen38ForwardRunnerError.invalidInput(
                    "Qwen 3.8 logits buffer is too small")
            }
        }

        let D = UInt32(cfg.hiddenSize)
        let cb = try commandBuffer()

        let emb = model.embedding
        embedInt4.encode(commandBuffer: cb,
                         table: emb.buffer, tableOffset: Int(emb.offset),
                         scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                         biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                         out: hidden,
                         tokenId: UInt32(bitPattern: token),
                         d: D,
                         outScale: 1.0)

        for (index, layer) in layers.enumerated() {
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: layer.inputNorm.buffer,
                            weightOffset: Int(layer.inputNorm.offset),
                            out: normed,
                            d: D, eps: Self.epsilon)
            if layer.isLinear {
                encodeLinearAttentionDecode(cb, layer: layer, layerIndex: index)
            } else {
                encodeGatedFullAttentionDecode(cb, layer: layer,
                                               layerIndex: index,
                                               position: position,
                                               seqLen: UInt32(position + 1))
            }
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: hidden,
                                          delta: oOut,
                                          count: cfg.hiddenSize)
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: layer.postAttnNorm.buffer,
                            weightOffset: Int(layer.postAttnNorm.offset),
                            out: mlpX,
                            d: D, eps: Self.epsilon)
            try mlp.encode(commandBuffer: cb,
                           x: mlpX,
                           gate: layer.mlpGate,
                           up: layer.mlpUp,
                           down: layer.mlpDown,
                           y: mlpOut,
                           scratchGate: mlpScratchGate,
                           scratchUp: mlpScratchUp,
                           scratchAct: mlpScratchAct)
            elementwise.encodeResidualAdd(commandBuffer: cb,
                                          hidden: hidden,
                                          delta: mlpOut,
                                          count: cfg.hiddenSize)
        }

        if emitHead {
            let fNorm = model.finalNorm
            let lm = model.lmHead
            if emitLogitsHead, let logits {
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: fNorm.buffer,
                                weightOffset: Int(fNorm.offset),
                                out: normed,
                                d: D, eps: Self.epsilon)
                int4.encode(commandBuffer: cb,
                            weights: lm.buffer, weightsOffset: Int(lm.offset),
                            scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                            biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                            x: normed, y: logits,
                            m: UInt32(cfg.vocabSize), n: D)
            } else {
                fusionHead.encodeGreedyDecode(commandBuffer: cb,
                                              hidden: hidden,
                                              normWeight: fNorm.buffer,
                                              normOffset: Int(fNorm.offset),
                                              weights: lm.buffer,
                                              weightsOffset: Int(lm.offset),
                                              scales: lm.buffer,
                                              scalesOffset: Int(lm.scaleOffset),
                                              biases: lm.buffer,
                                              biasesOffset: Int(lm.biasOffset),
                                              outToken: greedyTokenBuf,
                                              d: D, vocab: UInt32(cfg.vocabSize),
                                              rmsEps: Self.epsilon)
            }
        }

        try finish(cb)
        if emitHead, !emitLogitsHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
        kv.advance()
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step. Reads
    /// `normed`, updates the layer's recurrent state + conv tail in place,
    /// and leaves the attention-branch output in `oOut`. Hv 48 != the fused
    /// `_qwen` kernels' Hv 32, so `encodeDeltaGatedDecode` declines and the
    /// generic delta step + gated norm run instead.
    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer,
                                             layer: LayerTensors,
                                             layerIndex: Int) {
        guard let qkvW = layer.linQKV, let zW = layer.linZ,
              let aW = layer.linA, let bW = layer.linB,
              let outW = layer.linOut, let convW = layer.linConv,
              let aLog = layer.linALog, let dtBias = layer.linDtBias,
              let gatedNormW = layer.linNorm else {
            preconditionFailure("linear-attention layer without GDN tensors")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)

        // One dispatch over the concatenated qkv/z/a/b row space instead of
        // four separate GEMVs.
        gdn.encodeInputProjections(commandBuffer: cb,
                                   x: normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)
        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: layerIndex),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        let usedFusedDeltaNorm = gdn.encodeDeltaGatedDecode(
            commandBuffer: cb,
            convOut: gdnConvOut,
            aProj: gdnA,
            bProj: gdnB,
            aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
            dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
            state: gdnState.stateBuffer(layer: layerIndex),
            z: gdnZ,
            weight: gatedNormW.buffer, weightOffset: Int(gatedNormW.offset),
            out: gdnOut)
        if !usedFusedDeltaNorm {
            gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                      convOut: gdnConvOut,
                                      aProj: gdnA,
                                      bProj: gdnB,
                                      aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                      dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                      state: gdnState.stateBuffer(layer: layerIndex),
                                      y: gdnY)
            gdn.encodeGatedNorm(commandBuffer: cb,
                                y: gdnY,
                                z: gdnZ,
                                weight: gatedNormW.buffer,
                                weightOffset: Int(gatedNormW.offset),
                                out: gdnOut)
        }
        int4.encode(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE over headDim * partialRotaryFactor
    /// dims, full attention at the configured scale, sigmoid output gate,
    /// then o_proj into `oOut`.
    private func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer: LayerTensors,
                                                layerIndex: Int,
                                                position: Int,
                                                seqLen: UInt32) {
        guard let q = layer.q, let k = layer.k, let v = layer.v, let o = layer.o,
              let qNormW = layer.qNorm, let kNormW = layer.kNorm else {
            preconditionFailure("full-attention layer without attention tensors")
        }
        let D = UInt32(cfg.hiddenSize)
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: layerIndex, position: position)
        let vSlot = kv.vSlot(layer: layerIndex, position: position)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        fusedQKVGEMV.encode(commandBuffer: cb,
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            x: normed,
                            qOut: qPackedScratch,
                            kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                            qRows: 2 * qDim,
                            kvRows: kvDim,
                            n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: qScratch,
                               weight: qNormW.buffer,
                               weightOffset: Int(qNormW.offset),
                               out: qScratch,
                               headDim: UInt32(headDim),
                               numHeads: cfg.numHeads,
                               eps: Self.epsilon)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kSlot.buffer, xOffset: kSlot.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kSlot.buffer, outOffset: kSlot.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: Self.epsilon)
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: qScratch,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: kSlot.buffer,
                              dataOffset: kSlot.offset,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        attention.encodeFull(commandBuffer: cb,
                             q: qScratch,
                             k: kSlot.buffer, kOffset: 0,
                             v: vSlot.buffer, vOffset: 0,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: seqLen,
                             scale: Float(cfg.attentionScale))
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        int4.encode(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create Qwen 3.8 command buffer")
        }
        return cb
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw Qwen38ForwardRunnerError.commandFailed(
                cb.error?.localizedDescription ?? "Qwen 3.8 command buffer did not complete")
        }
    }
}
