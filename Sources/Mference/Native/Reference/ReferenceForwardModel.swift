import Foundation
import Metal

extension Native {
    /// Builds the reference `qwen35` runner from any `ModelComponents` mix.
    /// The runner calls only the components it is given, so swapping one
    /// component (a Phase 1 kernel, store or weight reader) changes only that
    /// component's work.
    public final class ReferenceForwardModelFactory: ForwardModelFactory {
        let context: MetalContext

        public init(context: MetalContext) {
            self.context = context
        }

        public func activationBytes(config: Qwen35Config, maxPrefillChunk: Int, maxVerifyRows: Int) -> UInt64 {
            UInt64(ReferenceForwardModel.ScratchLayout(config: config, maxRows: max(maxPrefillChunk, maxVerifyRows),
                                                       logitRows: max(1, maxVerifyRows)).totalBytes)
        }

        public func make(_ components: sending ModelComponents, plan: MemoryPlan) throws -> sending any ForwardModel {
            try ReferenceForwardModel(components: components, plan: plan)
        }
    }

    /// The reference `qwen35` runner (`docs/NATIVE_ENGINE.md`, "Data flow of
    /// one step"). One shared scratch buffer holds the activations; host
    /// tokens and logits use two-slot rings so two steps may be in flight.
    /// Mixer and FFN outputs accumulate straight into the residual unless a
    /// tap asks for them, in which case they go to scratch, are copied out,
    /// and are added with `GlueOps.encodeAdd` (the same F32 add).
    public final class ReferenceForwardModel: ForwardModel {
        public let config: Qwen35Config
        public let kv: any KVStore
        public let maxPrefillChunk: Int
        public let maxVerifyRows: Int
        public var tap: (any TensorTap)?

        let components: ModelComponents
        let layout: ScratchLayout
        let scratch: MTLBuffer
        private var stepCounter = 0

        struct AttentionBlock {
            var index: Int
            var q, k, v, output: QuantTensor
            var weights: AttentionLayerWeights
        }

        struct GDNBlock {
            var index: Int
            var qkv, gate, alpha, beta, output: QuantTensor
            var weights: GDNLayerWeights
        }

        struct Block {
            var attnNorm, postNorm, ffnGate, ffnUp, ffnDown: QuantTensor
            var attention: AttentionBlock?
            var gdn: GDNBlock?
        }

        let embedding: QuantTensor
        let outputNorm: QuantTensor
        let lmHead: QuantTensor
        let blocks: [Block]

        /// Byte offsets of every scratch region (256-byte aligned).
        struct ScratchLayout {
            var x = 0, h = 0, mix = 0, qGate = 0, k = 0, v = 0, attn = 0, qkv = 0, z = 0, alpha = 0, beta = 0
            var gdn = 0, ffnGate = 0, ffnUp = 0, swiglu = 0, tokens = 0, logits = 0
            var tokenSlotBytes = 0, logitSlotBytes = 0, totalBytes = 0
            let maxRows: Int
            let logitRows: Int

            init(config c: Qwen35Config, maxRows: Int, logitRows: Int) {
                self.maxRows = maxRows
                self.logitRows = logitRows
                var cursor = 0
                func region(_ floats: Int) -> Int {
                    let at = cursor
                    cursor += (floats * 4 + 255) / 256 * 256
                    return at
                }
                x = region(maxRows * c.hiddenSize)
                h = region(maxRows * c.hiddenSize)
                mix = region(maxRows * c.hiddenSize)
                qGate = region(maxRows * c.headCount * c.keyLength * 2)
                k = region(maxRows * c.kvHeadCount * c.keyLength)
                v = region(maxRows * c.kvHeadCount * c.valueLength)
                attn = region(maxRows * c.headCount * c.valueLength)
                qkv = region(maxRows * c.ssmConvChannels)
                z = region(maxRows * c.ssmInnerSize)
                alpha = region(maxRows * c.ssmTimeStepRank)
                beta = region(maxRows * c.ssmTimeStepRank)
                gdn = region(maxRows * c.ssmInnerSize)
                ffnGate = region(maxRows * c.ffnSize)
                ffnUp = region(maxRows * c.ffnSize)
                swiglu = region(maxRows * c.ffnSize)
                tokenSlotBytes = (maxRows * 4 + 255) / 256 * 256
                tokens = region(2 * tokenSlotBytes / 4)
                logitSlotBytes = (logitRows * c.vocabSize * 4 + 255) / 256 * 256
                logits = region(2 * logitSlotBytes / 4)
                totalBytes = cursor
            }
        }

        init(components: ModelComponents, plan: MemoryPlan) throws {
            let w = components.weights
            let c = w.config
            config = c
            kv = components.kv
            self.components = components
            maxPrefillChunk = plan.maxPrefillChunk
            maxVerifyRows = max(1, plan.maxVerifyRows)
            layout = ScratchLayout(config: c, maxRows: max(maxPrefillChunk, maxVerifyRows), logitRows: maxVerifyRows)
            guard let buffer = components.device.makeBuffer(length: layout.totalBytes, options: .storageModeShared) else {
                throw NativeError.resource("reference scratch (\(layout.totalBytes) bytes)")
            }
            scratch = buffer
            guard kv.capacity <= plan.contextLength, kv.geometry.attentionLayers == c.fullAttentionLayers.count else {
                throw NativeError.invalidShape("KV store does not match the plan")
            }

            embedding = try w.tensor("token_embd.weight")
            outputNorm = try w.tensor("output_norm.weight")
            lmHead = try c.tiedEmbeddings ? embedding : w.tensor("output.weight")
            var blocks: [Block] = []
            var attnIndex = 0, gdnIndex = 0
            for i in 0..<c.blockCount {
                func t(_ name: String) throws -> QuantTensor { try w.tensor("blk.\(i).\(name)") }
                var block = Block(attnNorm: try t("attn_norm.weight"), postNorm: try t("post_attention_norm.weight"),
                                  ffnGate: try t("ffn_gate.weight"), ffnUp: try t("ffn_up.weight"),
                                  ffnDown: try t("ffn_down.weight"))
                switch c.layerKind(i) {
                case .fullAttention:
                    block.attention = AttentionBlock(
                        index: attnIndex, q: try t("attn_q.weight"), k: try t("attn_k.weight"),
                        v: try t("attn_v.weight"), output: try t("attn_output.weight"),
                        weights: AttentionLayerWeights(qNorm: try t("attn_q_norm.weight"),
                                                       kNorm: try t("attn_k_norm.weight")))
                    attnIndex += 1
                case .gatedDeltaNet:
                    block.gdn = GDNBlock(
                        index: gdnIndex, qkv: try t("attn_qkv.weight"), gate: try t("attn_gate.weight"),
                        alpha: try t("ssm_alpha.weight"), beta: try t("ssm_beta.weight"),
                        output: try t("ssm_out.weight"),
                        weights: GDNLayerWeights(conv1d: try t("ssm_conv1d.weight"), dtBias: try t("ssm_dt.bias"),
                                                 a: try t("ssm_a"), norm: try t("ssm_norm.weight")))
                    gdnIndex += 1
                }
                blocks.append(block)
            }
            self.blocks = blocks

            var types: Set<GGMLType> = [embedding.type, lmHead.type]
            for b in blocks {
                for m in [b.ffnGate, b.ffnUp, b.ffnDown] { types.insert(m.type) }
                if let a = b.attention { for m in [a.q, a.k, a.v, a.output] { types.insert(m.type) } }
                if let g = b.gdn { for m in [g.qkv, g.gate, g.alpha, g.beta, g.output] { types.insert(m.type) } }
            }
            try components.gemm.prepare(types: types)
            try components.attention.prepare(config: c, geometry: kv.geometry)
            try components.gdn.prepare(config: c, layout: kv.recurrentLayout)
            try components.glue.prepare(config: c)
        }

        // MARK: Views

        func view(_ offset: Int, rows: Int, cols: Int, row: Int = 0, dtype: DType = .f32) -> TensorRef {
            TensorRef(buffer: scratch, offset: offset + row * cols * dtype.byteSize, dtype: dtype, rows: rows, cols: cols)
        }

        private func emit(_ point: TapPoint, _ tensor: TensorRef, _ encoder: MTLComputeCommandEncoder) throws {
            guard let tap, let destination = tap.destination(for: point, like: tensor) else { return }
            try components.glue.encodeCopy(encoder, source: tensor, destination: destination)
        }

        private func wants(_ point: TapPoint, like tensor: TensorRef) -> TensorRef? {
            tap?.destination(for: point, like: tensor)
        }

        private func matmul(_ encoder: MTLComputeCommandEncoder, _ kind: StepKind, _ weight: QuantTensor,
                            _ input: TensorRef, _ output: TensorRef, _ mode: GEMMOutputMode) throws {
            let gemm = components.gemm
            if input.rows == 1 {
                try gemm.encodeGEMV(encoder, weight: weight, input: input, output: output, mode: mode)
            } else if kind == .verify {
                try gemm.encodeVerify(encoder, weight: weight, input: input, output: output, mode: mode)
            } else {
                try gemm.encodePrefill(encoder, weight: weight, input: input, output: output, mode: mode)
            }
        }

        /// `x += W · input`, or, when the tap wants `point`, `mix = W · input`,
        /// copy out, `x = x + mix`.
        private func residual(_ encoder: MTLComputeCommandEncoder, _ kind: StepKind, _ weight: QuantTensor,
                              _ input: TensorRef, x: TensorRef, rows: Int, point: TapPoint) throws {
            let mix = view(layout.mix, rows: rows, cols: config.hiddenSize)
            if let destination = wants(point, like: mix) {
                try matmul(encoder, kind, weight, input, mix, .overwrite)
                try components.glue.encodeCopy(encoder, source: mix, destination: destination)
                try components.glue.encodeAdd(encoder, a: x, b: mix, output: x)
            } else {
                try matmul(encoder, kind, weight, input, x, .accumulate)
            }
        }

        // MARK: Step

        public func encodeStep(_ plan: StepPlan, encoder: MTLComputeCommandEncoder) throws -> TensorRef? {
            let rows = plan.rows
            switch plan.kind {
            case .decode:
                guard rows == 1 else { throw NativeError.invalidShape("decode step with \(rows) rows") }
            case .verify:
                guard (2...min(16, maxVerifyRows)).contains(rows) else {
                    throw NativeError.invalidShape("verify step with \(rows) rows (max \(maxVerifyRows))")
                }
            case .prefill:
                guard (1...maxPrefillChunk).contains(rows) else {
                    throw NativeError.invalidShape("prefill step with \(rows) rows (max \(maxPrefillChunk))")
                }
            }
            guard kv.length == plan.startPosition + rows else {
                throw NativeError.stateOutOfRange("KV length \(kv.length), step ends at \(plan.startPosition + rows)")
            }
            if plan.logits == .all && rows > layout.logitRows {
                throw NativeError.invalidShape("all-row logits for \(rows) rows (max \(layout.logitRows))")
            }
            let slot = stepCounter % 2
            stepCounter += 1
            let c = config
            let tokens: TokenSlot
            switch plan.input {
            case .host(let ids):
                guard ids.count == rows else { throw NativeError.invalidShape("\(ids.count) host tokens for \(rows) rows") }
                let offset = layout.tokens + slot * layout.tokenSlotBytes
                let p = scratch.contents().advanced(by: offset).assumingMemoryBound(to: UInt32.self)
                for (i, t) in ids.enumerated() { p[i] = UInt32(bitPattern: t) }
                tokens = TokenSlot(buffer: scratch, offset: offset, count: rows)
            case .device(let slotRef):
                guard slotRef.count == rows else { throw NativeError.invalidShape("token slot of \(slotRef.count)") }
                tokens = slotRef
            }

            let x = view(layout.x, rows: rows, cols: c.hiddenSize)
            let h = view(layout.h, rows: rows, cols: c.hiddenSize)
            let glue = components.glue
            try components.gemm.encodeGetRows(encoder, weight: embedding, tokens: tokens, output: x)
            try emit(.embedding, x, encoder)
            let pageTable = kv.pageTable()

            for (i, block) in blocks.enumerated() {
                try glue.encodeRMSNorm(encoder, input: x, weight: block.attnNorm, output: h)
                try emit(.mixerInput(layer: i), h, encoder)
                if let a = block.attention {
                    let qGate = view(layout.qGate, rows: rows, cols: c.headCount * c.keyLength * 2)
                    let k = view(layout.k, rows: rows, cols: c.kvHeadCount * c.keyLength)
                    let v = view(layout.v, rows: rows, cols: c.kvHeadCount * c.valueLength)
                    let out = view(layout.attn, rows: rows, cols: c.headCount * c.valueLength)
                    try matmul(encoder, plan.kind, a.q, h, qGate, .overwrite)
                    try matmul(encoder, plan.kind, a.k, h, k, .overwrite)
                    try matmul(encoder, plan.kind, a.v, h, v, .overwrite)
                    let op = components.attention
                    switch plan.kind {
                    case .decode:
                        try op.encodeDecode(encoder, attnLayer: a.index, weights: a.weights, qGate: qGate, k: k, v: v,
                                            startPosition: plan.startPosition, kv: pageTable, output: out)
                    case .verify:
                        try op.encodeVerify(encoder, attnLayer: a.index, weights: a.weights, qGate: qGate, k: k, v: v,
                                            startPosition: plan.startPosition, kv: pageTable, output: out)
                    case .prefill:
                        try op.encodePrefill(encoder, attnLayer: a.index, weights: a.weights, qGate: qGate, k: k,
                                             v: v, startPosition: plan.startPosition, kv: pageTable, output: out)
                    }
                    try residual(encoder, plan.kind, a.output, out, x: x, rows: rows, point: .mixerOutput(layer: i))
                } else if let g = block.gdn {
                    let qkv = view(layout.qkv, rows: rows, cols: c.ssmConvChannels)
                    let z = view(layout.z, rows: rows, cols: c.ssmInnerSize)
                    let alpha = view(layout.alpha, rows: rows, cols: c.ssmTimeStepRank)
                    let beta = view(layout.beta, rows: rows, cols: c.ssmTimeStepRank)
                    let out = view(layout.gdn, rows: rows, cols: c.ssmInnerSize)
                    try matmul(encoder, plan.kind, g.qkv, h, qkv, .overwrite)
                    try matmul(encoder, plan.kind, g.gate, h, z, .overwrite)
                    try matmul(encoder, plan.kind, g.alpha, h, alpha, .overwrite)
                    try matmul(encoder, plan.kind, g.beta, h, beta, .overwrite)
                    let state = kv.recurrentState(gdnLayer: g.index)
                    let op = components.gdn
                    switch plan.kind {
                    case .decode:
                        try op.encodeDecode(encoder, weights: g.weights, qkv: qkv, z: z, alpha: alpha, beta: beta,
                                            stateIn: state.input, stateOut: state.output, output: out)
                    case .verify:
                        try op.encodeVerify(encoder, weights: g.weights, qkv: qkv, z: z, alpha: alpha, beta: beta,
                                            stateIn: state.input, stateOut: state.output,
                                            arena: kv.verifyArena(gdnLayer: g.index), output: out)
                    case .prefill:
                        try op.encodePrefill(encoder, weights: g.weights, qkv: qkv, z: z, alpha: alpha, beta: beta,
                                             stateIn: state.input, stateOut: state.output, output: out)
                    }
                    try residual(encoder, plan.kind, g.output, out, x: x, rows: rows, point: .mixerOutput(layer: i))
                }
                try glue.encodeRMSNorm(encoder, input: x, weight: block.postNorm, output: h)
                let fg = view(layout.ffnGate, rows: rows, cols: c.ffnSize)
                let fu = view(layout.ffnUp, rows: rows, cols: c.ffnSize)
                let u = view(layout.swiglu, rows: rows, cols: c.ffnSize)
                try matmul(encoder, plan.kind, block.ffnGate, h, fg, .overwrite)
                try matmul(encoder, plan.kind, block.ffnUp, h, fu, .overwrite)
                try glue.encodeSwiGLU(encoder, gate: fg, up: fu, output: u)
                try residual(encoder, plan.kind, block.ffnDown, u, x: x, rows: rows, point: .ffnOutput(layer: i))
                try emit(.layerOutput(layer: i), x, encoder)
            }

            let headRows: Int
            let headInput: TensorRef
            switch plan.logits {
            case .none:
                return nil
            case .last:
                headRows = 1
                headInput = view(layout.x, rows: 1, cols: c.hiddenSize, row: rows - 1)
            case .all:
                headRows = rows
                headInput = x
            }
            let normed = view(layout.h, rows: headRows, cols: c.hiddenSize)
            try glue.encodeRMSNorm(encoder, input: headInput, weight: outputNorm, output: normed)
            try emit(.finalNorm, normed, encoder)
            let logits = TensorRef(buffer: scratch, offset: layout.logits + slot * layout.logitSlotBytes, dtype: .f32,
                                   rows: headRows, cols: c.vocabSize)
            try matmul(encoder, plan.kind, lmHead, normed, logits, .overwrite)
            try emit(.logits, logits, encoder)
            return logits
        }
    }
}
