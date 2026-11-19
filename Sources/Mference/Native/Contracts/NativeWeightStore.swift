import Metal

extension Native {
    /// One weight tensor as the kernels see it: quant type, shape, tile
    /// layout and the GPU memory holding its packed bytes.
    ///
    /// `buffer` is a no-copy wrap of the mapped pack (or, in the reference,
    /// any buffer); the tensor's bytes are `buffer[offset ..< offset + byteCount]`.
    /// The store keeps the mapping alive for the store's lifetime.
    public struct QuantTensor {
        public var name: String
        public var type: GGMLType
        /// ggml `ne` order; for matrices `shape[0]` = inputs (cols),
        /// `shape[1]` = outputs (rows).
        public var shape: [Int]
        public var layout: TileLayout
        public var buffer: MTLBuffer
        public var offset: Int
        public var byteCount: Int
        /// SHA-256 of the tensor's bytes in GGUF order.
        public var sourceSHA256: String

        public init(name: String, type: GGMLType, shape: [Int], layout: TileLayout,
                    buffer: MTLBuffer, offset: Int, byteCount: Int, sourceSHA256: String) {
            self.name = name
            self.type = type
            self.shape = shape
            self.layout = layout
            self.buffer = buffer
            self.offset = offset
            self.byteCount = byteCount
            self.sourceSHA256 = sourceSHA256
        }

        /// Input width of a matrix (`shape[0]`).
        public var cols: Int { shape[0] }
        /// Output rows of a matrix (`shape[1]`, 1 for vectors).
        public var rows: Int { shape.count > 1 ? shape[1] : 1 }
    }

    /// Read-only access to a loaded model's weights.
    ///
    /// Bounded memory (CLAUDE.md invariant): weights are mapped read-only and
    /// wrapped with `makeBuffer(bytesNoCopy:)`; no tensor is copied into the
    /// Swift heap. Tensor names are GGUF names (`blk.N.attn_qkv.weight`,
    /// `blk.N.ssm_a`, `token_embd.weight`, ...); `nextn.*` blocks, when
    /// present, are exposed under their GGUF names too.
    public protocol WeightStore: AnyObject {
        var manifest: PackManifest { get }
        var config: Qwen35Config { get }
        /// The tensor called `name`; throws `NativeError.missing` if absent.
        func tensor(_ name: String) throws -> QuantTensor
        /// Whether a tensor called `name` exists.
        func contains(_ name: String) -> Bool
        /// Bytes mapped for weights (for `MemoryPlan`).
        var mappedByteCount: UInt64 { get }
        var tokenizer: TokenizerSource { get }
        var chatTemplate: ChatTemplateSource { get }
    }

    /// Tokenizer data from pack metadata (`tokenizer.json`).
    public protocol TokenizerSource: AnyObject {
        var data: TokenizerData { get }
        /// A tokenizer whose `encode`/`decode` are byte-identical to llama.cpp's
        /// `llama_tokenize` / `llama_token_to_piece` at the pin.
        func makeTokenizer() throws -> any TextTokenizer
    }

    /// llama.cpp-identical tokenization.
    public protocol TextTokenizer: AnyObject {
        var vocabSize: Int { get }
        var eosTokenID: Int32? { get }
        /// Token ids that end generation (EOS and end-of-generation tokens).
        var endOfGenerationTokenIDs: Set<Int32> { get }
        /// `llama_tokenize(text, add_special: addSpecial, parse_special: parseSpecial)`.
        func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [Int32]
        /// `llama_token_to_piece(token, special: special)` bytes.
        func piece(_ token: Int32, special: Bool) -> [UInt8]
    }

    /// Chat template from pack metadata (`chat_template.jinja`).
    public protocol ChatTemplateSource: AnyObject {
        /// The template text, byte-identical to GGUF `tokenizer.chat_template`.
        var template: String { get }
        var bosToken: String? { get }
        var eosToken: String? { get }
    }
}
