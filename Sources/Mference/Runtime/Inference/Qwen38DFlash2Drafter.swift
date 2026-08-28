import Foundation
import Metal

/// DFlash2 block-diffusion drafter for the Qwen 3.8 dense family.
///
/// z-lab `model_mlx.py` semantics: five BF16 sliding-attention layers
/// predict a whole block of masked future tokens in one forward pass.
/// Conditioning is "KV injection" — target hidden states from five tap
/// layers, concatenated and projected (`fc` + `hidden_norm`), enter every
/// draft layer's K/V and persist in a per-layer rotating context cache.
/// DFlash2 adds a grouped dynamic conv around attention and MLP and a
/// top-16 path selector over the drafted block.
///
/// The drafter supplies *draft tokens only*. The MTP speculator's verify /
/// accept / rollback machinery owns correctness: emitted bytes are
/// identical to plain decode for any draft quality, so everything here
/// trades only acceptance length.
final class Qwen38DFlash2Drafter {

    /// Projection storage. `.int4` (default) quantizes at load into the
    /// production multi-x GEMV format — 3.85 GB BF16 would both thrash a
    /// 24 GB host against the 15 GB target and read 4x the bytes per round.
    /// `.bf16` (MFERENCE_DFLASH2_BF16=1, and the parity tests) runs the
    /// checkpoint unquantized for reference comparison.
    enum Precision {
        case int4
        case bf16

        static var fromEnvironment: Precision {
            ProcessInfo.processInfo.environment["MFERENCE_DFLASH2_BF16"] == "1"
                ? .bf16 : .int4
        }
    }

    struct DrafterConfig: Decodable {
        struct DFlash: Decodable {
            let block_size: Int
            let conv_group_size: Int
            let conv_kernel_size: Int
            let mask_token_id: Int
            let selector_rank: Int
            let selector_top_k: Int
            let target_layer_ids: [Int]
        }
        let dflash_config: DFlash
        let head_dim: Int
        let hidden_size: Int
        let intermediate_size: Int
        let num_attention_heads: Int
        let num_hidden_layers: Int
        let num_key_value_heads: Int
        let num_target_layers: Int
        let rms_norm_eps: Float
        let sliding_window: Int
        let vocab_size: Int
        let rope_parameters: RopeParameters
        struct RopeParameters: Decodable { let rope_theta: Float }
    }

    private struct LayerWeights {
        let inputNorm: DFlash2Weights.Tensor
        let postAttnNorm: DFlash2Weights.Tensor
        let q: DFlash2Weights.Tensor
        let k: DFlash2Weights.Tensor
        let v: DFlash2Weights.Tensor
        let o: DFlash2Weights.Tensor
        let qNorm: DFlash2Weights.Tensor
        let kNorm: DFlash2Weights.Tensor
        let mlpGate: DFlash2Weights.Tensor
        let mlpUp: DFlash2Weights.Tensor
        let mlpDown: DFlash2Weights.Tensor
        let attnConvBase: DFlash2Weights.Tensor      // [2, K, H]
        let attnConvProj: DFlash2Weights.Tensor      // [2*K*G, H]
        let mlpConvBase: DFlash2Weights.Tensor
        let mlpConvProj: DFlash2Weights.Tensor
    }

    let config: DrafterConfig
    /// Tap ordinals by target layer index (e.g. 5 -> 0, 19 -> 1, ...).
    let tapOrdinal: [Int: Int]
    /// Maximum pending tap rows: the drafter context window plus one block.
    let tapCapacity: Int

    private let ctx: MetalContext
    private let weights: DFlash2Weights
    private let precision: Precision
    /// BF16-mode GPU wrap of the checkpoint mapping; nil in INT4 mode.
    private let weightsGPU: MTLBuffer?
    /// INT4 mode: quantized projections; nil in BF16 mode.
    private let slab: DFlash2Int4Slab?
    /// Anonymous copies of the small tensors (norms, conv base kernels) so
    /// production command buffers never touch file-backed memory.
    private let smallBuf: MTLBuffer
    private let smallOffsets: [Int: Int]
    private let kernels: DFlash2Kernels
    private let prefillRMS: PrefillRMSNorm
    private let prefillRope: PrefillRoPE
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let elementwise: Elementwise
    private let multix: DequantInt4GEMVMultiX
    private let siluMulPSO: MTLComputePipelineState

    private let fc: DFlash2Weights.Tensor
    private let hiddenNorm: DFlash2Weights.Tensor
    private let finalNorm: DFlash2Weights.Tensor
    private let selectorHiddenProj: DFlash2Weights.Tensor
    private let predecessorCodebook: DFlash2Weights.Tensor
    private let successorCodebook: DFlash2Weights.Tensor
    private let layers: [LayerWeights]

    // Target-side shared pieces (embedding / LM head, INT4).
    private let targetEmbedding: TensorView
    private let targetLMHead: TensorView
    private let targetVocab: Int

    // Rotating per-layer context K/V rings (ping/pong for compaction).
    private let ringCapacity: Int
    private var ctxK: [[MTLBuffer]]   // [layer][pingpong]
    private var ctxV: [[MTLBuffer]]
    private var activeRing = 0
    private var ringStart = 0         // first kept row inside the ring
    private(set) var ctxKept = 0      // kept context rows (<= window - 1)
    private(set) var ctxTotal = 0     // total appended rows (RoPE base)

    // Pending target-tap rows, concatenated [row, tapCount * D].
    let tapStaging: MTLBuffer
    private(set) var pendingTapRows = 0

    // Scratch.
    private let ctxFeat: MTLBuffer    // [tapCapacity, D]
    private let hCtx: MTLBuffer       // [tapCapacity, D]
    private let ctxKNew: MTLBuffer    // [tapCapacity, kvDim]
    private let ctxVNew: MTLBuffer    // [tapCapacity, kvDim]
    private let blockIds: MTLBuffer
    private let blockH: MTLBuffer     // [B, D]
    private let blockNormed: MTLBuffer
    private let blockConvX: MTLBuffer
    private let blockDyn: MTLBuffer   // [B, 2*K*G]
    private let blockQ: MTLBuffer     // [B, qDim]
    private let blockK: MTLBuffer     // [B, kvDim]
    private let blockV: MTLBuffer
    private let blockAttnOut: MTLBuffer
    private let blockO: MTLBuffer     // [B, D]
    private let blockConvOut: MTLBuffer
    private let mlpGateBuf: MTLBuffer // [B, F]
    private let mlpUpBuf: MTLBuffer
    private let mlpActBuf: MTLBuffer
    private let mlpOutBuf: MTLBuffer
    private let normedFinal: MTLBuffer // [B, D]
    private let logitsBuf: MTLBuffer   // [B-1, vocab]
    private let candIdxBuf: MTLBuffer  // [B-1, 16] u32
    private let candValBuf: MTLBuffer  // [B-1, 16] f32
    private let hprojBuf: MTLBuffer    // [B-1, rank]

    /// Builds a drafter when `MFERENCE_DFLASH2_DIR` points at a directory
    /// holding the checkpoint's `config.json` + `model.safetensors`.
    static func probe(context: MetalContext,
                      model: Model,
                      targetConfig: ArchConfig) throws -> Qwen38DFlash2Drafter? {
        guard let dir = ProcessInfo.processInfo.environment["MFERENCE_DFLASH2_DIR"],
              !dir.isEmpty else { return nil }
        return try Qwen38DFlash2Drafter(
            context: context,
            directory: URL(fileURLWithPath: dir),
            embedding: model.embedding,
            lmHead: model.lmHead,
            targetConfig: targetConfig,
            precision: .fromEnvironment)
    }

    init(context: MetalContext,
         directory: URL,
         embedding: TensorView,
         lmHead: TensorView,
         targetConfig: ArchConfig,
         precision: Precision = .fromEnvironment) throws {
        self.precision = precision
        self.ctx = context
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        self.config = try JSONDecoder().decode(DrafterConfig.self, from: configData)
        guard config.vocab_size == targetConfig.vocabSize else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "dflash2 drafter vocab \(config.vocab_size) != target \(targetConfig.vocabSize)")
        }
        guard config.num_target_layers == targetConfig.numLayers else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "dflash2 drafter expects \(config.num_target_layers) target layers, "
                + "target has \(targetConfig.numLayers)")
        }
        // The drafter shares the target's embedding/LM head and consumes tap
        // captures of the target's hidden rows, so a width mismatch would
        // mis-stride those kernels instead of failing cleanly.
        guard config.hidden_size == targetConfig.hiddenSize else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "dflash2 drafter hidden size \(config.hidden_size) "
                + "!= target \(targetConfig.hiddenSize)")
        }
        self.tapOrdinal = Dictionary(
            uniqueKeysWithValues: config.dflash_config.target_layer_ids.enumerated()
                .map { ($0.element, $0.offset) })
        self.weights = try DFlash2Weights(
            safetensorsURL: directory.appendingPathComponent("model.safetensors"),
            device: context.device)
        self.kernels = try DFlash2Kernels(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillRope = try PrefillRoPE(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.elementwise = try Elementwise(context: context)
        self.multix = try DequantInt4GEMVMultiX(context: context)
        self.siluMulPSO = try context.pipeline("silu_mul_fp16")
        self.targetEmbedding = embedding
        self.targetLMHead = lmHead
        self.targetVocab = targetConfig.vocabSize

        let D = config.hidden_size
        let taps = config.dflash_config.target_layer_ids.count
        let B = config.dflash_config.block_size
        let F = config.intermediate_size
        let qDim = config.num_attention_heads * config.head_dim
        let kvDim = config.num_key_value_heads * config.head_dim
        let K = config.dflash_config.conv_kernel_size
        let G = D / config.dflash_config.conv_group_size

        self.fc = try weights.tensor("fc.weight", shape: [D, taps * D])
        self.hiddenNorm = try weights.tensor("hidden_norm.weight", shape: [D])
        self.finalNorm = try weights.tensor("norm.weight", shape: [D])
        self.selectorHiddenProj = try weights.tensor(
            "candidate_selector.hidden_projection.weight",
            shape: [config.dflash_config.selector_rank, D])
        self.predecessorCodebook = try weights.tensor(
            "candidate_selector.predecessor_codebook",
            shape: [config.vocab_size, config.dflash_config.selector_rank])
        self.successorCodebook = try weights.tensor(
            "candidate_selector.successor_codebook",
            shape: [config.vocab_size, config.dflash_config.selector_rank])
        let weightsFile = self.weights
        let headDim = config.head_dim
        self.layers = try (0..<config.num_hidden_layers).map { l in
            func t(_ suffix: String, _ shape: [Int]) throws -> DFlash2Weights.Tensor {
                try weightsFile.tensor("layers.\(l).\(suffix)", shape: shape)
            }
            return LayerWeights(
                inputNorm: try t("input_layernorm.weight", [D]),
                postAttnNorm: try t("post_attention_layernorm.weight", [D]),
                q: try t("self_attn.q_proj.weight", [qDim, D]),
                k: try t("self_attn.k_proj.weight", [kvDim, D]),
                v: try t("self_attn.v_proj.weight", [kvDim, D]),
                o: try t("self_attn.o_proj.weight", [D, qDim]),
                qNorm: try t("self_attn.q_norm.weight", [headDim]),
                kNorm: try t("self_attn.k_norm.weight", [headDim]),
                mlpGate: try t("mlp.gate_proj.weight", [F, D]),
                mlpUp: try t("mlp.up_proj.weight", [F, D]),
                mlpDown: try t("mlp.down_proj.weight", [D, F]),
                attnConvBase: try t("attention_conv.base_kernel", [2, K, D]),
                attnConvProj: try t("attention_conv.kernel_projection.weight", [2 * K * G, D]),
                mlpConvBase: try t("mlp_conv.base_kernel", [2, K, D]),
                mlpConvProj: try t("mlp_conv.kernel_projection.weight", [2 * K * G, D]))
        }

        // Small tensors -> one anonymous GPU buffer (norms, conv bases).
        var smallTensors: [DFlash2Weights.Tensor] = [hiddenNorm, finalNorm]
        for layer in layers {
            smallTensors.append(contentsOf: [layer.inputNorm, layer.postAttnNorm,
                                             layer.qNorm, layer.kNorm,
                                             layer.attnConvBase, layer.mlpConvBase])
        }
        var smallTotal = 0
        var smallMap: [Int: Int] = [:]
        for t in smallTensors {
            smallMap[t.offset] = smallTotal
            smallTotal += (t.byteCount + 63) / 64 * 64
        }
        guard let small = context.device.makeBuffer(length: max(smallTotal, 64),
                                                    options: .storageModeShared) else {
            throw Qwen38ForwardRunnerError.invalidConfiguration(
                "unable to allocate dflash2 small-tensor buffer")
        }
        small.label = "dflash2.small"
        for t in smallTensors {
            small.contents().advanced(by: smallMap[t.offset]!)
                .copyMemory(from: UnsafeRawPointer(weightsFile.cpuPointer(of: t)),
                            byteCount: t.byteCount)
        }
        self.smallBuf = small
        self.smallOffsets = smallMap

        switch precision {
        case .bf16:
            self.weightsGPU = try weightsFile.gpuBuffer()
            self.slab = nil
        case .int4:
            self.weightsGPU = nil
            var quantList: [(name: String, base: UnsafePointer<UInt16>, shape: [Int])] = [
                ("fc", weightsFile.cpuPointer(of: fc), fc.shape),
                ("sel", weightsFile.cpuPointer(of: selectorHiddenProj),
                 selectorHiddenProj.shape),
            ]
            for (l, layer) in layers.enumerated() {
                for (suffix, t) in [("q", layer.q), ("k", layer.k), ("v", layer.v),
                                    ("o", layer.o), ("gate", layer.mlpGate),
                                    ("up", layer.mlpUp), ("down", layer.mlpDown),
                                    ("aconv", layer.attnConvProj),
                                    ("mconv", layer.mlpConvProj)] {
                    quantList.append(("L\(l).\(suffix)",
                                      weightsFile.cpuPointer(of: t), t.shape))
                }
            }
            let attrs = try? FileManager.default.attributesOfItem(
                atPath: directory.appendingPathComponent("model.safetensors").path)
            let size = (attrs?[.size] as? Int) ?? 0
            let mtime = (attrs?[.modificationDate] as? Date)?
                .timeIntervalSince1970 ?? 0
            self.slab = try DFlash2Int4Slab(
                tensors: quantList,
                cacheURL: directory.appendingPathComponent("int4-slab.cache"),
                cacheKey: "v1:\(size):\(Int(mtime))",
                device: context.device)
        }

        let window = config.sliding_window
        self.tapCapacity = window - 1 + B
        self.ringCapacity = 4096
        precondition(ringCapacity >= window - 1 + B,
                     "context ring must hold a full window plus one block")

        let device = context.device
        func buf(_ elements: Int, _ label: String,
                 stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(elements, 1) * stride,
                                               options: .storageModeShared) else {
                throw Qwen38ForwardRunnerError.invalidConfiguration(
                    "unable to allocate dflash2 scratch (\(label))")
            }
            made.label = "dflash2.\(label)"
            return made
        }
        self.ctxK = []
        self.ctxV = []
        for l in 0..<config.num_hidden_layers {
            self.ctxK.append([try buf(ringCapacity * kvDim, "ctxK.\(l).a"),
                              try buf(ringCapacity * kvDim, "ctxK.\(l).b")])
            self.ctxV.append([try buf(ringCapacity * kvDim, "ctxV.\(l).a"),
                              try buf(ringCapacity * kvDim, "ctxV.\(l).b")])
        }
        self.tapStaging = try buf(tapCapacity * taps * D, "taps")
        self.ctxFeat = try buf(tapCapacity * D, "ctxFeat")
        self.hCtx = try buf(tapCapacity * D, "hCtx")
        self.ctxKNew = try buf(tapCapacity * kvDim, "ctxKNew")
        self.ctxVNew = try buf(tapCapacity * kvDim, "ctxVNew")
        self.blockIds = try buf(B, "blockIds", stride: MemoryLayout<UInt32>.stride)
        self.blockH = try buf(B * D, "blockH", stride: MemoryLayout<Float>.stride)
        self.blockNormed = try buf(B * D, "blockNormed")
        self.blockConvX = try buf(B * D, "blockConvX")
        self.blockDyn = try buf(B * 2 * K * G, "blockDyn")
        self.blockQ = try buf(B * qDim, "blockQ")
        self.blockK = try buf(B * kvDim, "blockK")
        self.blockV = try buf(B * kvDim, "blockV")
        self.blockAttnOut = try buf(B * qDim, "blockAttnOut")
        self.blockO = try buf(B * D, "blockO", stride: MemoryLayout<Float>.stride)
        self.blockConvOut = try buf(B * D, "blockConvOut", stride: MemoryLayout<Float>.stride)
        self.mlpGateBuf = try buf(B * F, "mlpGate")
        self.mlpUpBuf = try buf(B * F, "mlpUp")
        self.mlpActBuf = try buf(B * F, "mlpAct")
        self.mlpOutBuf = try buf(B * D, "mlpOut", stride: MemoryLayout<Float>.stride)
        self.normedFinal = try buf(B * D, "normedFinal")
        self.logitsBuf = try buf((B - 1) * targetVocab, "logits")
        self.candIdxBuf = try buf((B - 1) * 16, "candIdx",
                                  stride: MemoryLayout<UInt32>.stride)
        self.candValBuf = try buf((B - 1) * 16, "candVal",
                                  stride: MemoryLayout<Float>.stride)
        self.hprojBuf = try buf((B - 1) * config.dflash_config.selector_rank, "hproj")
    }

    // MARK: - Target-tap staging

    /// Whether target layer `layerIndex` is one of the drafter's tap layers.
    func isTapLayer(_ layerIndex: Int) -> Bool { tapOrdinal[layerIndex] != nil }

    /// How many trailing prompt rows are worth capturing: the drafter's
    /// sliding context window (matching the reference's `hidden_limit`).
    var contextWindowRows: Int { config.sliding_window - 1 }

    /// Encode a tap capture: `rows` rows of the target's hidden state after
    /// layer `layerIndex` ([rows, D] at `srcOffset`) land in the staging
    /// matrix at the current pending row. Call for each tap layer with the
    /// same `rows`, then `commitTapRows` once per batch.
    func encodeTapCapture(commandBuffer cb: MTLCommandBuffer,
                          layerIndex: Int,
                          src: MTLBuffer, srcOffset: Int = 0,
                          rows: Int) {
        guard let ordinal = tapOrdinal[layerIndex] else { return }
        precondition(pendingTapRows + rows <= tapCapacity,
                     "tap staging overflow: \(pendingTapRows) + \(rows)")
        kernels.encodeTapGather(commandBuffer: cb,
                                src: src, srcOffset: srcOffset,
                                dst: tapStaging,
                                d: config.hidden_size,
                                tapCount: tapOrdinal.count,
                                tapIndex: ordinal,
                                dstRow: pendingTapRows,
                                tokens: rows)
    }

    /// Advance the pending-row cursor after all tap layers captured `rows`.
    /// The verify path stages every verify row, then commits only the
    /// accepted prefix — later captures overwrite the rejected rows.
    func commitTapRows(_ rows: Int) {
        precondition(pendingTapRows + rows <= tapCapacity)
        pendingTapRows += rows
    }

    func reset() {
        ctxKept = 0
        ctxTotal = 0
        ringStart = 0
        pendingTapRows = 0
    }

    /// After a reset, restart position bookkeeping at `base` so appended
    /// rows carry the target stream's true RoPE positions.
    func alignPositionBase(_ base: Int) {
        precondition(ctxKept == 0 && pendingTapRows == 0,
                     "position base can only move on an empty context")
        ctxTotal = base
    }

    // MARK: - Draft round

    /// Runs one draft round: consumes the pending tap rows as new context,
    /// then proposes up to `maxDrafts` tokens continuing after `anchor`.
    func propose(anchor: Int32, maxDrafts: Int) throws -> [Int32] {
        let drafts = min(maxDrafts, config.dflash_config.block_size - 1)
        precondition(drafts >= 1, "dflash2 propose needs draft room")
        let blockTokens = drafts + 1

        let ids = blockIds.contents().bindMemory(to: UInt32.self,
                                                 capacity: blockTokens)
        ids[0] = UInt32(bitPattern: anchor)
        for i in 1..<blockTokens { ids[i] = UInt32(config.dflash_config.mask_token_id) }

        let cb = try commandBuffer()
        try encodeContextAppend(cb)
        let emb = targetEmbedding
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer, tableOffset: Int(emb.offset),
                            scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                            tokens: blockIds,
                            out: blockNormed,
                            t: UInt32(blockTokens), d: UInt32(config.hidden_size),
                            outScale: 1.0)
        kernels.encodeF16ToF32(commandBuffer: cb, src: blockNormed, dst: blockH,
                               count: blockTokens * config.hidden_size)
        try encodeCore(cb, blockTokens: blockTokens, normStartRow: 1)

        // Head: full logits for the mask rows, top-16, selector projection.
        let lm = targetLMHead
        multix.encode(commandBuffer: cb,
                      weights: lm.buffer, weightsOffset: Int(lm.offset),
                      scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                      biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                      x: normedFinal, y: logitsBuf,
                      m: targetVocab, n: config.hidden_size, tokens: drafts)
        kernels.encodeTopK16(commandBuffer: cb,
                             logits: logitsBuf,
                             outIndices: candIdxBuf, outValues: candValBuf,
                             rows: drafts, vocab: targetVocab)
        encodeProjection(cb, name: "sel", tensor: selectorHiddenProj,
                         x: normedFinal, y: hprojBuf,
                         m: config.dflash_config.selector_rank,
                         n: config.hidden_size, tokens: drafts)
        try finish(cb)
        pendingTapRows = 0
        return selectPath(anchor: anchor, drafts: drafts)
    }

    /// Test seam: run the drafter core on pre-staged block hidden rows and
    /// pending taps, final-norming every row. Mirrors the reference
    /// `hidden_states` with the embedding bypassed.
    func runCoreForParity(blockTokens: Int) throws {
        let cb = try commandBuffer()
        try encodeContextAppend(cb)
        try encodeCore(cb, blockTokens: blockTokens, normStartRow: 0)
        try finish(cb)
        pendingTapRows = 0
    }

    var parityBlockHidden: MTLBuffer { blockH }
    var parityNormedOut: MTLBuffer { normedFinal }
    /// Diagnostic access to intermediate scratch (parity bring-up only).
    var parityProbe: [String: MTLBuffer] {
        ["ctxFeat": ctxFeat, "hCtx": hCtx, "ctxKNew": ctxKNew,
         "ctxVNew": ctxVNew, "blockNormed": blockNormed,
         "blockConvX": blockConvX, "blockDyn": blockDyn, "blockQ": blockQ,
         "blockK": blockK, "blockV": blockV, "blockAttnOut": blockAttnOut,
         "blockO": blockO, "blockConvOut": blockConvOut,
         "mlpAct": mlpActBuf, "mlpOut": mlpOutBuf, "blockH": blockH]
    }

    /// Test seam: runs the top-16 + selector chain on caller-provided
    /// final-normed hidden rows and full-vocab logits.
    func selectorParityRun(hidden: [Float16], logits: [Float16],
                           anchor: Int32, rows: Int) throws
        -> (path: [Int32], candidates: [[UInt32]]) {
        precondition(hidden.count == rows * config.hidden_size)
        precondition(logits.count == rows * targetVocab)
        hidden.withUnsafeBytes {
            normedFinal.contents().copyMemory(from: $0.baseAddress!,
                                              byteCount: $0.count)
        }
        logits.withUnsafeBytes {
            logitsBuf.contents().copyMemory(from: $0.baseAddress!,
                                            byteCount: $0.count)
        }
        let cb = try commandBuffer()
        kernels.encodeTopK16(commandBuffer: cb,
                             logits: logitsBuf,
                             outIndices: candIdxBuf, outValues: candValBuf,
                             rows: rows, vocab: targetVocab)
        encodeProjection(cb, name: "sel", tensor: selectorHiddenProj,
                         x: normedFinal, y: hprojBuf,
                         m: config.dflash_config.selector_rank,
                         n: config.hidden_size, tokens: rows)
        try finish(cb)
        let path = selectPath(anchor: anchor, drafts: rows)
        let idx = candIdxBuf.contents().bindMemory(to: UInt32.self,
                                                   capacity: rows * 16)
        let candidates = (0..<rows).map { r in
            (0..<16).map { idx[r * 16 + $0] }
        }
        return (path, candidates)
    }

    // MARK: - Encoding

    /// fc + hidden_norm over the pending tap rows, per-layer K/V projection,
    /// per-head K norm + RoPE at the rows' global positions, ring append.
    private func encodeContextAppend(_ cb: MTLCommandBuffer) throws {
        let S = pendingTapRows
        guard S > 0 else { return }
        let D = config.hidden_size
        let taps = tapOrdinal.count
        let kvDim = config.num_key_value_heads * config.head_dim
        let h = MemoryLayout<Float16>.stride

        encodeProjection(cb, name: "fc", tensor: fc,
                         x: tapStaging, y: ctxFeat,
                         m: D, n: taps * D, tokens: S)
        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: ctxFeat,
                               weight: smallBuf, weightOffset: small(hiddenNorm),
                               out: hCtx,
                               t: UInt32(S), d: UInt32(D),
                               eps: config.rms_norm_eps)

        // Ring compaction happens before the append when the linear tail
        // would overflow; the kept window slides afterwards.
        let window = config.sliding_window
        if ringStart + ctxKept + S > ringCapacity {
            let other = 1 - activeRing
            if ctxKept > 0, let blit = cb.makeBlitCommandEncoder() {
                for l in 0..<layers.count {
                    blit.copy(from: ctxK[l][activeRing],
                              sourceOffset: ringStart * kvDim * h,
                              to: ctxK[l][other], destinationOffset: 0,
                              size: ctxKept * kvDim * h)
                    blit.copy(from: ctxV[l][activeRing],
                              sourceOffset: ringStart * kvDim * h,
                              to: ctxV[l][other], destinationOffset: 0,
                              size: ctxKept * kvDim * h)
                }
                blit.endEncoding()
            }
            activeRing = other
            ringStart = 0
        }

        for (l, layer) in layers.enumerated() {
            encodeProjection(cb, name: "L\(l).k", tensor: layer.k,
                             x: hCtx, y: ctxKNew,
                             m: kvDim, n: D, tokens: S)
            encodeProjection(cb, name: "L\(l).v", tensor: layer.v,
                             x: hCtx, y: ctxVNew,
                             m: kvDim, n: D, tokens: S)
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: ctxKNew,
                                   weight: smallBuf,
                                   weightOffset: small(layer.kNorm),
                                   out: ctxKNew,
                                   t: UInt32(S * config.num_key_value_heads),
                                   d: UInt32(config.head_dim),
                                   eps: config.rms_norm_eps)
            prefillRope.encodeNeoxSubdim(commandBuffer: cb,
                                         data: ctxKNew,
                                         startPosition: UInt32(ctxTotal),
                                         queryCount: UInt32(S),
                                         headDim: UInt32(config.head_dim),
                                         numHeads: UInt32(config.num_key_value_heads),
                                         rotaryDim: UInt32(config.head_dim),
                                         tokenStrideElements: UInt32(kvDim),
                                         theta: config.rope_parameters.rope_theta)
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw Qwen38ForwardRunnerError.commandFailed(
                    "unable to create dflash2 ring blit encoder")
            }
            let dstOffset = (ringStart + ctxKept) * kvDim * h
            blit.copy(from: ctxKNew, sourceOffset: 0,
                      to: ctxK[l][activeRing], destinationOffset: dstOffset,
                      size: S * kvDim * h)
            blit.copy(from: ctxVNew, sourceOffset: 0,
                      to: ctxV[l][activeRing], destinationOffset: dstOffset,
                      size: S * kvDim * h)
            blit.endEncoding()
        }

        ctxTotal += S
        let kept = ctxKept + S
        let keepMax = window - 1
        if kept > keepMax {
            ringStart += kept - keepMax
            ctxKept = keepMax
        } else {
            ctxKept = kept
        }
    }

    /// The five drafter layers over `blockTokens` rows of `blockH`, then the
    /// final norm from `normStartRow` into `normedFinal`.
    private func encodeCore(_ cb: MTLCommandBuffer,
                            blockTokens B: Int,
                            normStartRow: Int) throws {
        let D = config.hidden_size
        let F = config.intermediate_size
        let qDim = config.num_attention_heads * config.head_dim
        let kvDim = config.num_key_value_heads * config.head_dim
        let K = config.dflash_config.conv_kernel_size
        let groupSize = config.dflash_config.conv_group_size
        let h = MemoryLayout<Float16>.stride
        // Block queries/keys sit right after every appended context row.
        let blockBase = ctxTotal
        // Byte stride between the two [K, D] planes of a base_kernel tensor.
        let basePlaneBytes = K * D * 2
        // Bring-up bisector: run only the first N layers.
        let layerLimit = ProcessInfo.processInfo
            .environment["MFERENCE_DFLASH2_LAYER_LIMIT"].flatMap(Int.init)
            ?? layers.count

        for (l, layer) in layers.enumerated() where l < layerLimit {
            // Attention sublayer with the dynamic conv wrap.
            kernels.encodeRMSF32Rows(commandBuffer: cb,
                                     x: blockH,
                                     weight: smallBuf,
                                     weightOffset: small(layer.inputNorm),
                                     out: blockNormed,
                                     rows: B, d: D,
                                     eps: config.rms_norm_eps)
            encodeProjection(cb, name: "L\(l).aconv", tensor: layer.attnConvProj,
                             x: blockNormed, y: blockDyn,
                             m: 2 * K * (D / groupSize), n: D, tokens: B)
            kernels.encodeDynConv(commandBuffer: cb,
                                  x: blockNormed, dynamic: blockDyn,
                                  base: smallBuf,
                                  baseOffset: small(layer.attnConvBase),
                                  out: blockConvX,
                                  tokens: B, hidden: D,
                                  kernelSize: K, groupSize: groupSize, plane: 0)
            encodeProjection(cb, name: "L\(l).q", tensor: layer.q,
                             x: blockConvX, y: blockQ,
                             m: qDim, n: D, tokens: B)
            encodeProjection(cb, name: "L\(l).k", tensor: layer.k,
                             x: blockConvX, y: blockK,
                             m: kvDim, n: D, tokens: B)
            encodeProjection(cb, name: "L\(l).v", tensor: layer.v,
                             x: blockConvX, y: blockV,
                             m: kvDim, n: D, tokens: B)
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: blockQ,
                                   weight: smallBuf,
                                   weightOffset: small(layer.qNorm),
                                   out: blockQ,
                                   t: UInt32(B * config.num_attention_heads),
                                   d: UInt32(config.head_dim),
                                   eps: config.rms_norm_eps)
            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: blockK,
                                   weight: smallBuf,
                                   weightOffset: small(layer.kNorm),
                                   out: blockK,
                                   t: UInt32(B * config.num_key_value_heads),
                                   d: UInt32(config.head_dim),
                                   eps: config.rms_norm_eps)
            prefillRope.encodeNeoxSubdim(commandBuffer: cb,
                                         data: blockQ,
                                         startPosition: UInt32(blockBase),
                                         queryCount: UInt32(B),
                                         headDim: UInt32(config.head_dim),
                                         numHeads: UInt32(config.num_attention_heads),
                                         rotaryDim: UInt32(config.head_dim),
                                         tokenStrideElements: UInt32(qDim),
                                         theta: config.rope_parameters.rope_theta)
            prefillRope.encodeNeoxSubdim(commandBuffer: cb,
                                         data: blockK,
                                         startPosition: UInt32(blockBase),
                                         queryCount: UInt32(B),
                                         headDim: UInt32(config.head_dim),
                                         numHeads: UInt32(config.num_key_value_heads),
                                         rotaryDim: UInt32(config.head_dim),
                                         tokenStrideElements: UInt32(kvDim),
                                         theta: config.rope_parameters.rope_theta)
            kernels.encodeBlockAttention(
                commandBuffer: cb,
                q: blockQ,
                ctxK: ctxK[l][activeRing], ctxV: ctxV[l][activeRing],
                ctxByteOffset: ringStart * kvDim * h,
                blkK: blockK, blkV: blockV,
                out: blockAttnOut,
                tokens: B, ctxLen: ctxKept,
                window: config.sliding_window,
                numQHeads: config.num_attention_heads,
                numKVHeads: config.num_key_value_heads,
                scale: 1.0 / Float(config.head_dim).squareRoot())
            encodeProjection(cb, name: "L\(l).o", tensor: layer.o,
                             x: blockAttnOut, y: blockO,
                             m: D, n: qDim, tokens: B,
                             outputFloat32: true)
            kernels.encodeDynConv(commandBuffer: cb,
                                  x: blockO, dynamic: blockDyn,
                                  base: smallBuf,
                                  baseOffset: small(layer.attnConvBase) + basePlaneBytes,
                                  out: blockConvOut,
                                  tokens: B, hidden: D,
                                  kernelSize: K, groupSize: groupSize, plane: 1,
                                  float32IO: true)
            kernels.encodeResidualAddF32(commandBuffer: cb,
                                         hidden: blockH, delta: blockConvOut,
                                         count: B * D)

            // MLP sublayer with its own dynamic conv wrap.
            kernels.encodeRMSF32Rows(commandBuffer: cb,
                                     x: blockH,
                                     weight: smallBuf,
                                     weightOffset: small(layer.postAttnNorm),
                                     out: blockNormed,
                                     rows: B, d: D,
                                     eps: config.rms_norm_eps)
            encodeProjection(cb, name: "L\(l).mconv", tensor: layer.mlpConvProj,
                             x: blockNormed, y: blockDyn,
                             m: 2 * K * (D / groupSize), n: D, tokens: B)
            kernels.encodeDynConv(commandBuffer: cb,
                                  x: blockNormed, dynamic: blockDyn,
                                  base: smallBuf,
                                  baseOffset: small(layer.mlpConvBase),
                                  out: blockConvX,
                                  tokens: B, hidden: D,
                                  kernelSize: K, groupSize: groupSize, plane: 0)
            encodeProjection(cb, name: "L\(l).gate", tensor: layer.mlpGate,
                             x: blockConvX, y: mlpGateBuf,
                             m: F, n: D, tokens: B)
            encodeProjection(cb, name: "L\(l).up", tensor: layer.mlpUp,
                             x: blockConvX, y: mlpUpBuf,
                             m: F, n: D, tokens: B)
            try encodeSiluMul(cb, gate: mlpGateBuf, up: mlpUpBuf,
                              out: mlpActBuf, count: B * F)
            encodeProjection(cb, name: "L\(l).down", tensor: layer.mlpDown,
                             x: mlpActBuf, y: mlpOutBuf,
                             m: D, n: F, tokens: B,
                             outputFloat32: true)
            kernels.encodeDynConv(commandBuffer: cb,
                                  x: mlpOutBuf, dynamic: blockDyn,
                                  base: smallBuf,
                                  baseOffset: small(layer.mlpConvBase) + basePlaneBytes,
                                  out: blockConvOut,
                                  tokens: B, hidden: D,
                                  kernelSize: K, groupSize: groupSize, plane: 1,
                                  float32IO: true)
            kernels.encodeResidualAddF32(commandBuffer: cb,
                                         hidden: blockH, delta: blockConvOut,
                                         count: B * D)
        }

        let rows = B - normStartRow
        kernels.encodeRMSF32Rows(commandBuffer: cb,
                                 x: blockH,
                                 xOffset: normStartRow * D * MemoryLayout<Float>.stride,
                                 weight: smallBuf, weightOffset: small(finalNorm),
                                 out: normedFinal,
                                 rows: rows, d: D,
                                 eps: config.rms_norm_eps)
    }

    /// One projection through the active precision's kernel: quantized
    /// multi-x GEMV from the slab, or the BF16 reference GEMV.
    private func encodeProjection(_ cb: MTLCommandBuffer,
                                  name: String, tensor: DFlash2Weights.Tensor,
                                  x: MTLBuffer, xOffset: Int = 0,
                                  y: MTLBuffer, yOffset: Int = 0,
                                  m: Int, n: Int, tokens: Int,
                                  outputFloat32: Bool = false) {
        if let slab {
            let view = slab.view(name)
            var remaining = tokens
            var xByte = xOffset
            var yByte = yOffset
            let yStride = outputFloat32 ? MemoryLayout<Float>.stride
                                        : MemoryLayout<Float16>.stride
            while remaining > 0 {
                let t = min(remaining, DequantInt4GEMVMultiX.maxTokens)
                multix.encode(commandBuffer: cb,
                              weights: slab.buffer, weightsOffset: view.weightsOffset,
                              scales: slab.buffer, scalesOffset: view.scalesOffset,
                              biases: slab.buffer, biasesOffset: view.biasesOffset,
                              x: x, xOffset: xByte,
                              y: y, yOffset: yByte,
                              m: m, n: n, tokens: t,
                              outputFloat32: outputFloat32)
                remaining -= t
                xByte += t * n * MemoryLayout<Float16>.stride
                yByte += t * m * yStride
            }
        } else {
            kernels.encodeGEMV(commandBuffer: cb,
                               weights: weightsGPU!, weightsOffset: tensor.offset,
                               x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                               m: m, n: n, tokens: tokens,
                               outputFloat32: outputFloat32)
        }
    }

    /// Offset of a small tensor's anonymous GPU copy in `smallBuf`.
    private func small(_ t: DFlash2Weights.Tensor) -> Int { smallOffsets[t.offset]! }

    private func encodeSiluMul(_ cb: MTLCommandBuffer,
                               gate: MTLBuffer, up: MTLBuffer, out: MTLBuffer,
                               count: Int) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create dflash2 activation encoder")
        }
        encoder.setComputePipelineState(siluMulPSO)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: 4, index: 3)
        let width = min(siluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - Selector (CPU)

    /// Greedy path trace: score = top-16 logit + low-rank bilinear edge from
    /// the previous committed token, chained across the block.
    private func selectPath(anchor: Int32, drafts: Int) -> [Int32] {
        let rank = config.dflash_config.selector_rank
        let topK = 16
        let candIdx = candIdxBuf.contents().bindMemory(to: UInt32.self,
                                                       capacity: drafts * topK)
        let candVal = candValBuf.contents().bindMemory(to: Float.self,
                                                       capacity: drafts * topK)
        let hproj = hprojBuf.contents().bindMemory(to: Float16.self,
                                                   capacity: drafts * rank)
        let pred = weights.cpuPointer(of: predecessorCodebook)
        let succ = weights.cpuPointer(of: successorCodebook)

        func bf16(_ v: UInt16) -> Float { Float(bitPattern: UInt32(v) << 16) }

        var predecessor = Int(anchor)
        var path: [Int32] = []
        path.reserveCapacity(drafts)
        for position in 0..<drafts {
            let predRow = pred + predecessor * rank
            var bestScore = -Float.infinity
            var bestToken = Int(candIdx[position * topK])
            for c in 0..<topK {
                let cand = Int(candIdx[position * topK + c])
                let succRow = succ + cand * rank
                var edge: Float = 0
                for r in 0..<rank {
                    edge += bf16(predRow[r]) * Float(hproj[position * rank + r])
                        * bf16(succRow[r])
                }
                let score = candVal[position * topK + c] + edge
                if score > bestScore {
                    bestScore = score
                    bestToken = cand
                }
            }
            predecessor = bestToken
            path.append(Int32(bestToken))
        }
        return path
    }

    // MARK: - Utilities

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw Qwen38ForwardRunnerError.commandFailed(
                "unable to create dflash2 command buffer")
        }
        return cb
    }

    private func finish(_ cb: MTLCommandBuffer) throws {
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw Qwen38ForwardRunnerError.commandFailed(
                "dflash2 command buffer failed: \(error)")
        }
    }
}
