import Metal

extension Native {
    /// What one step computes.
    public enum StepKind: Sendable, Equatable {
        /// A prompt chunk.
        case prefill
        /// One generated token.
        case decode
        /// 2...16 draft rows checked in one pass.
        case verify
    }

    /// Where a step's input tokens come from.
    public enum StepInput {
        /// Token ids known on the CPU (prompt chunks, draft rows).
        case host([Int32])
        /// Token ids a previous step wrote on the GPU (pipelined decode).
        case device(TokenSlot)
    }

    /// Which rows' logits a step produces.
    public enum LogitRows: Sendable, Equatable {
        case none
        case last
        case all
    }

    /// One step: `rows` tokens at positions `startPosition ..< startPosition + rows`.
    public struct StepPlan {
        public var kind: StepKind
        public var startPosition: Int
        public var rows: Int
        public var input: StepInput
        public var logits: LogitRows
        public init(kind: StepKind, startPosition: Int, rows: Int, input: StepInput, logits: LogitRows) {
            self.kind = kind
            self.startPosition = startPosition
            self.rows = rows
            self.input = input
            self.logits = logits
        }
    }

    /// Named intermediate tensors, for per-op parity against the golden.
    /// `docs/NATIVE_PARITY.md` maps each to the llama.cpp graph tensor it is
    /// compared with. `layer` is the block index.
    public enum TapPoint: Hashable, Sendable {
        /// Token embeddings (input of block 0).
        case embedding
        /// Normed input of the attention / GDN mixer.
        case mixerInput(layer: Int)
        /// Mixer output after `attn_output` / `ssm_out`, before the residual add.
        case mixerOutput(layer: Int)
        /// FFN output after `ffn_down`, before the residual add.
        case ffnOutput(layer: Int)
        /// Residual stream after the block.
        case layerOutput(layer: Int)
        /// Output of the final RMSNorm.
        case finalNorm
        /// LM-head logits.
        case logits
    }

    /// Receives copies of intermediate tensors while a step is encoded.
    public protocol TensorTap: AnyObject {
        /// Return a destination (same rows, cols, dtype as `tensor`) to have
        /// the model encode a copy of `tensor` into it, or nil to skip.
        func destination(for point: TapPoint, like tensor: TensorRef) -> TensorRef?
    }

    /// A family runner (`qwen35`): encodes one step of the model over a
    /// `WeightStore` and a `KVStore`. The reference runner is the Phase 0
    /// oracle; the fast runner is Phase 2 integration.
    public protocol ForwardModel: AnyObject {
        var config: Qwen35Config { get }
        var kv: any KVStore { get }
        /// Largest `rows` accepted for a prefill step.
        var maxPrefillChunk: Int { get }
        /// Largest `rows` accepted for a verify step (1 when none).
        var maxVerifyRows: Int { get }
        /// Optional per-op capture; nil in production.
        var tap: (any TensorTap)? { get set }
        /// Encode `plan` into `encoder`. The caller has already called
        /// `kv.reserve(rows: plan.rows)`. Returns the logits (F32,
        /// `[n x vocabSize]`, n = 0, 1 or rows per `plan.logits`), valid until
        /// the next `encodeStep`; nil for `.none`. Host tokens are staged by the
        /// model before it returns (the caller may commit right away).
        func encodeStep(_ plan: StepPlan, encoder: MTLComputeCommandEncoder) throws -> TensorRef?
    }
}
